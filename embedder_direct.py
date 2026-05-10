"""Direct in-process multi-GPU BGE-M3 embedder via llama-cpp-python + Vulkan.

Per-GPU named workers, each with its own request queue. NO shared queue
(was the GPU 4=99%/GPU 5,6=0% bug in the prior version). Round-robin
counter for retrieve (n=1) calls; explicit fan-out for batch ingestion.

Architecture (codified 2026-04-25):
- One subprocess per GPU, pinned via GGML_VK_VISIBLE_DEVICES.
- Each worker has its own (req_q, resp_q) pair. No shared input queue.
- Profile-driven: warmup calls, load timeout, weight, ctx all per-GPU.
- Weighted round-robin for retrieve; balanced fan-out for batch.
- Per-GPU health: ready / degraded / dead. Pool routes around bad GPUs.
- Hot add/remove for the GPU 2 swap protocol (qwen3 vs embed).
- Live counters: pool.health() returns per-GPU stats.

Env:
  EMBED_GPUS                : override default GPU list (e.g. "0,1,3,4,5,6")
  EMBED_PROFILES            : path to embed_pool_profiles.json (default beside this file)
  EMBED_MODEL_PATH          : override model path
"""
from __future__ import annotations
import os, json, time, uuid, threading, multiprocessing as mp
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Optional

# ──────────────────────────────────────────────────────────────────────
# Profile loading
# ──────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_DEFAULT_PROFILES = _HERE / "embed_pool_profiles.json"
PROFILES_PATH = Path(os.environ.get("EMBED_PROFILES",
    os.environ.get("EMBED_PROFILES_PATH", str(_DEFAULT_PROFILES))))

# Fallback profiles location (deployed file beside api.py is canonical;
# if running standalone, look for /opt/indian-legal-ai/embed_pool_profiles.json)
if not PROFILES_PATH.exists():
    alt = Path("/opt/indian-legal-ai/embed_pool_profiles.json")
    if alt.exists():
        PROFILES_PATH = alt

if not PROFILES_PATH.exists():
    raise FileNotFoundError(
        f"embed_pool_profiles.json not found. Tried: {PROFILES_PATH}. "
        "This file MUST exist — per-GPU adaptive config is mandatory."
    )

with open(PROFILES_PATH) as _f:
    _CFG = json.load(_f)

MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", _CFG["model_path"])
DENSE_DIM = int(_CFG.get("dense_dim", 1024))
DEFAULT_GPUS = [int(x) for x in os.environ.get(
    "EMBED_GPUS", ",".join(str(g) for g in _CFG["default_gpus"])).split(",")]
PROFILES: Dict[int, dict] = {int(k): v for k, v in _CFG["gpus"].items()}
DEGRADED_POLICY = _CFG.get("degraded_policy", {})
MIN_POOL_SIZE = int(_CFG.get("min_pool_size_to_start", 2))
REBALANCE_AFTER_WARMUP = bool(_CFG.get("rebalance_after_warmup", True))

# ──────────────────────────────────────────────────────────────────────
# Worker process — one BGE-M3 instance pinned to one GPU
# ──────────────────────────────────────────────────────────────────────

def _gpu_worker(gpu_id: int, profile: dict, model_path: str,
                req_q, resp_q, ready_q):
    """Run inside a spawned subprocess. Loads BGE-M3, services its own queue."""
    os.environ["GGML_VK_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["GGML_VK_DISABLE_INTEGER_DOT_PRODUCT"] = "1"
    os.environ.setdefault("RADV_DEBUG", "nodcc")
    name = profile.get("name", f"GPU{gpu_id}")
    t0 = time.time()
    try:
        from llama_cpp import Llama
        m = Llama(
            model_path=model_path,
            embedding=True, n_gpu_layers=-1,
            n_ctx=profile.get("n_ctx", 8192),
            n_batch=profile.get("n_batch", 512),
            n_ubatch=profile.get("n_ubatch", 512),
            n_threads=profile.get("n_threads", 2),
            verbose=False, pooling_type=2,
        )
        # Warmup — post-reset cards get more, to stabilize clocks
        warmup_n = int(profile.get("warmup_calls", 1))
        warmup_ms = []
        for i in range(warmup_n):
            ts = time.time()
            m.create_embedding(["warmup probe " + str(i)])
            warmup_ms.append(int((time.time() - ts) * 1000))
        load_s = time.time() - t0
        ready_q.put({
            "gpu": gpu_id, "name": name, "ok": True,
            "load_s": round(load_s, 2),
            "warmup_ms": warmup_ms,
            "warmup_p50_ms": sorted(warmup_ms)[len(warmup_ms)//2] if warmup_ms else None,
        })
    except Exception as e:
        ready_q.put({
            "gpu": gpu_id, "name": name, "ok": False,
            "load_s": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {e}",
        })
        return

    # Service loop — own queue, not shared
    fail_streak = 0
    while True:
        item = req_q.get()
        if item is None:
            return
        req_id, texts = item
        out: List[List[float]] = []
        ok_count = 0
        err_count = 0
        latencies_ms: List[int] = []
        for t in texts:
            ts = time.time()
            try:
                v = m.create_embedding([t])
                data = v.get("data") or []
                if data and "embedding" in data[0]:
                    out.append(list(data[0]["embedding"]))
                    ok_count += 1
                else:
                    out.append([0.0] * DENSE_DIM)
                    err_count += 1
                fail_streak = 0
            except Exception as e:
                err_count += 1
                fail_streak += 1
                out.append([0.0] * DENSE_DIM)
                # Don't crash — let pool route around us via degraded counter.
                resp_q.put({"req_id": req_id, "_partial_err":
                    f"gpu{gpu_id} {type(e).__name__}: {e}"})
            latencies_ms.append(int((time.time() - ts) * 1000))
        resp_q.put({
            "req_id": req_id, "embeddings": out,
            "ok": ok_count, "err": err_count, "fail_streak": fail_streak,
            "lat_ms": latencies_ms, "gpu": gpu_id,
        })

# ──────────────────────────────────────────────────────────────────────
# Pool — per-GPU named workers, round-robin counter, weighted
# ──────────────────────────────────────────────────────────────────────

class _GPUWorkerHandle:
    """Per-GPU handle: own queues, own state, own counters."""
    def __init__(self, gpu_id: int, profile: dict, ctx):
        self.gpu_id = gpu_id
        self.profile = profile
        self.req_q = ctx.Queue()
        self.resp_q = ctx.Queue()
        self.proc: Optional[mp.process.BaseProcess] = None
        self.state = "init"   # init | ready | degraded | dead
        self.weight = float(profile.get("weight", 1.0))
        self.load_s: Optional[float] = None
        self.warmup_ms: List[int] = []
        self.error: Optional[str] = None
        # Live counters
        self.calls = 0
        self.errors = 0
        self.recent_lat = deque(maxlen=200)   # ms per-call
        self.last_err: Optional[str] = None
        self.last_err_ts: Optional[float] = None
        self.recent_fail_ts: deque = deque(maxlen=20)
        self.consecutive_fails = 0
        self._inflight = 0
        self._inflight_lock = threading.Lock()

    def start(self, ctx, model_path: str, ready_q):
        self.proc = ctx.Process(
            target=_gpu_worker,
            args=(self.gpu_id, self.profile, model_path, self.req_q, self.resp_q, ready_q),
            daemon=True,
        )
        self.proc.start()

    def mark_ready(self, info: dict):
        self.state = "ready"
        self.load_s = info.get("load_s")
        self.warmup_ms = info.get("warmup_ms", [])

    def mark_failed(self, info: dict):
        self.state = "dead"
        self.error = info.get("error")

    def record_call(self, lat_ms_list: List[int], err_count: int, fail_streak: int):
        self.calls += len(lat_ms_list)
        self.errors += err_count
        for l in lat_ms_list:
            self.recent_lat.append(l)
        self.consecutive_fails = fail_streak
        if err_count > 0:
            self.recent_fail_ts.append(time.time())
            self.last_err_ts = time.time()
        # Degraded / dead policy
        cutoff = time.time() - 60
        recent_fails = sum(1 for t in self.recent_fail_ts if t >= cutoff)
        max_fails = int(DEGRADED_POLICY.get("max_failures_60s", 3))
        dead_streak = int(DEGRADED_POLICY.get("dead_after_consecutive_failures", 5))
        if self.state == "ready" and recent_fails >= max_fails:
            self.state = "degraded"
        if self.consecutive_fails >= dead_streak:
            self.state = "dead"

    def stats(self) -> dict:
        lats = sorted(self.recent_lat) if self.recent_lat else []
        def pct(p):
            if not lats: return None
            return lats[min(len(lats) - 1, int(len(lats) * p))]
        return {
            "gpu": self.gpu_id, "name": self.profile.get("name"),
            "state": self.state, "weight": round(self.weight, 3),
            "load_s": self.load_s, "warmup_ms": self.warmup_ms,
            "calls": self.calls, "errors": self.errors,
            "p50_ms": pct(0.50), "p95_ms": pct(0.95),
            "consecutive_fails": self.consecutive_fails,
            "last_err": self.last_err, "error": self.error,
        }


class _Pool:
    def __init__(self, gpu_ids: List[int]):
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._ready_q = ctx.Queue()
        self.workers: Dict[int, _GPUWorkerHandle] = {}
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._pending: Dict[str, dict] = {}
        self._pending_cond = threading.Condition()
        self._stop = threading.Event()
        self._routers: List[threading.Thread] = []
        # 2026-04-25: SEQUENTIAL COLD-LOAD is mandatory on this rig.
        # Empirical proof: solo cold-load on GPU 1 = 2.8s, on GPU 6 = 2.7s.
        # 6-card concurrent cold-load = GPU 1 timeout 120s, GPU 6 timeout 300s,
        # surviving cards 45-48s (~16x slowdown). Concurrent Vulkan init fights
        # for shared driver/shader-compiler resources and starves the lemons.
        # Fix: load ONE card at a time; each finishes in ~3s; total ≤30s for 6.
        sequential = bool(_CFG.get("sequential_cold_load", True))
        gap_s = float(_CFG.get("sequential_cold_load_gap_s", 0.5))
        mode = "SEQUENTIAL" if sequential else "PARALLEL"
        print(f"[embed_pool] cold-load mode: {mode}", flush=True)
        for idx, gid in enumerate(gpu_ids):
            if gid not in PROFILES:
                print(f"[embed_pool] WARN: GPU {gid} has no profile, skipping", flush=True)
                continue
            h = _GPUWorkerHandle(gid, PROFILES[gid], ctx)
            h.start(ctx, MODEL_PATH, self._ready_q)
            self.workers[gid] = h
            # Start router for this worker immediately (drains its resp_q for runtime calls)
            t = threading.Thread(target=self._route_one, args=(h,), daemon=True,
                                 name=f"embed-router-gpu{gid}")
            t.start()
            self._routers.append(t)
            if sequential:
                # Block until THIS GPU is ready (or fails) before spawning the next.
                # Eliminates Vulkan-init contention.
                self._await_one(gid)
                if idx + 1 < len(gpu_ids) and gap_s > 0:
                    time.sleep(gap_s)
        if not sequential:
            self._await_ready()
        # Final fleet check
        live = [g for g, h in self.workers.items() if h.state == "ready"]
        if len(live) < MIN_POOL_SIZE:
            raise RuntimeError(
                f"[embed_pool] Only {len(live)} GPU ready (need ≥{MIN_POOL_SIZE}). "
                f"Live: {live}. Pool refusing to start with degraded fleet."
            )
        print(f"[embed_pool] {len(live)}/{len(self.workers)} GPUs ready: {live}", flush=True)
        # Optional rebalance after warmup
        if REBALANCE_AFTER_WARMUP:
            self._rebalance_weights_from_warmup()

    def _await_one(self, target_gid: int):
        """Block until the worker for target_gid emits a ready/fail signal,
        or its load_timeout_s deadline elapses. Other workers' signals are
        consumed and applied; we only return when target_gid is settled."""
        h_target = self.workers[target_gid]
        deadline = time.time() + h_target.profile.get("load_timeout_s", 120)
        while True:
            now = time.time()
            remaining = deadline - now
            if remaining <= 0:
                if h_target.state not in ("ready", "dead"):
                    h_target.state = "dead"
                    h_target.error = f"load timeout ({h_target.profile.get('load_timeout_s')}s)"
                    print(f"[embed_pool] GPU {target_gid} TIMEOUT after {h_target.profile.get('load_timeout_s')}s", flush=True)
                return
            try:
                info = self._ready_q.get(timeout=min(remaining, 5.0))
            except Exception:
                continue
            gid = info["gpu"]
            h = self.workers.get(gid)
            if h is None:
                continue
            if info.get("ok"):
                h.mark_ready(info)
                print(f"[embed_pool] GPU {gid} ({h.profile.get('name')}) ready in {info['load_s']}s, warmup_p50={info.get('warmup_p50_ms')}ms", flush=True)
            else:
                h.mark_failed(info)
                print(f"[embed_pool] GPU {gid} ({h.profile.get('name')}) FAILED: {info.get('error')}", flush=True)
            if gid == target_gid:
                return

    def _await_ready(self):
        """Legacy parallel-await path. Retained for sequential_cold_load=false."""
        deadline = {gid: time.time() + h.profile.get("load_timeout_s", 120)
                    for gid, h in self.workers.items()}
        pending = set(self.workers.keys())
        while pending:
            now = time.time()
            timeout = max(0.5, min(deadline[g] - now for g in pending) if pending else 1.0)
            try:
                info = self._ready_q.get(timeout=timeout)
            except Exception:
                for g in list(pending):
                    if time.time() > deadline[g]:
                        h = self.workers[g]
                        h.state = "dead"
                        h.error = f"load timeout ({h.profile.get('load_timeout_s')}s)"
                        print(f"[embed_pool] GPU {g} TIMEOUT after {h.profile.get('load_timeout_s')}s", flush=True)
                        pending.discard(g)
                continue
            gid = info["gpu"]
            h = self.workers.get(gid)
            if h is None:
                continue
            if info.get("ok"):
                h.mark_ready(info)
                print(f"[embed_pool] GPU {gid} ({h.profile.get('name')}) ready in {info['load_s']}s, warmup_p50={info.get('warmup_p50_ms')}ms", flush=True)
            else:
                h.mark_failed(info)
                print(f"[embed_pool] GPU {gid} ({h.profile.get('name')}) FAILED: {info.get('error')}", flush=True)
            pending.discard(gid)

    def _rebalance_weights_from_warmup(self):
        """Set weights inversely proportional to warmup p50 latency.
        Faster card → more weight. Codifies the heterogeneous-pool reality."""
        live = [h for h in self.workers.values() if h.state == "ready" and h.warmup_ms]
        if len(live) < 2:
            return
        # Use median of warmup_ms as the speed signal (smaller = faster)
        speeds = [(h, sorted(h.warmup_ms)[len(h.warmup_ms)//2]) for h in live]
        # Inverse latency, normalized so mean=1.0
        inv = [(h, 1.0 / max(1, ms)) for h, ms in speeds]
        mean_inv = sum(x for _, x in inv) / len(inv)
        for h, x in inv:
            h.weight = round(x / mean_inv, 3)
        print(f"[embed_pool] rebalanced weights: " + ", ".join(
            f"gpu{h.gpu_id}={h.weight}" for h in live), flush=True)

    def _route_one(self, h: _GPUWorkerHandle):
        while not self._stop.is_set():
            try:
                msg = h.resp_q.get(timeout=1.0)
            except Exception:
                continue
            req_id = msg.get("req_id")
            if "_partial_err" in msg:
                h.last_err = msg["_partial_err"]
                continue
            with h._inflight_lock:
                h._inflight = max(0, h._inflight - 1)
            h.record_call(msg.get("lat_ms", []), msg.get("err", 0),
                          msg.get("fail_streak", 0))
            with self._pending_cond:
                self._pending[req_id] = msg
                self._pending_cond.notify_all()

    # ──────────────────────────────────────────────────────────────
    # Selection — weighted round-robin + degraded-aware
    # ──────────────────────────────────────────────────────────────
    def _live_workers(self) -> List[_GPUWorkerHandle]:
        return [h for h in self.workers.values() if h.state in ("ready", "degraded")]

    def _pick_for_retrieve(self) -> _GPUWorkerHandle:
        """Weighted round-robin. Degraded GPUs get half their weight.
        Prefer least-inflight on tie. Single-call (n=1) path."""
        live = self._live_workers()
        if not live:
            raise RuntimeError("[embed_pool] no live GPUs to serve embed request")
        # Weighted scoring: weight - small-penalty * inflight
        scored = []
        for h in live:
            w = h.weight if h.state == "ready" else h.weight * 0.5
            score = w - 0.25 * h._inflight
            scored.append((score, h))
        scored.sort(key=lambda x: -x[0])
        # Round-robin within top-tier (prevents starvation if weights are equal)
        with self._counter_lock:
            self._counter += 1
            n = self._counter
        top_score = scored[0][0]
        top_tier = [h for s, h in scored if abs(s - top_score) < 0.01]
        return top_tier[n % len(top_tier)]

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Production path. n=1 → round-robin; n>1 → fan out across live GPUs."""
        n = len(texts)
        if n == 0:
            return []
        if n == 1:
            h = self._pick_for_retrieve()
            return self._dispatch_one(h, texts)
        # Batch — split across all live workers, weighted shards
        live = self._live_workers()
        if not live:
            raise RuntimeError("[embed_pool] no live GPUs for batch embed")
        total_w = sum(h.weight for h in live)
        # Assign each worker a share proportional to its weight, sum to n
        shares = [max(1, int(round(n * h.weight / total_w))) for h in live]
        # Adjust last share so sum == n (rounding fixup)
        diff = n - sum(shares)
        shares[-1] += diff
        idx = 0
        rids: List[str] = []
        groups: List[List[str]] = []
        targets: List[_GPUWorkerHandle] = []
        for h, s in zip(live, shares):
            if s <= 0:
                continue
            g = texts[idx:idx + s]
            if not g:
                continue
            idx += s
            rid = uuid.uuid4().hex
            rids.append(rid)
            groups.append(g)
            targets.append(h)
            with h._inflight_lock:
                h._inflight += 1
            h.req_q.put((rid, g))
        return self._collect(rids, targets)

    def embed_on(self, gpu_id: int, texts: List[str]) -> List[List[float]]:
        """Bench/diagnostic path — explicit per-GPU dispatch. Bypasses routing."""
        h = self.workers.get(gpu_id)
        if h is None:
            raise ValueError(f"[embed_pool] no worker for GPU {gpu_id}")
        if h.state == "dead":
            raise RuntimeError(f"[embed_pool] GPU {gpu_id} is dead: {h.error}")
        if not texts:
            return []
        return self._dispatch_one(h, texts)

    def _dispatch_one(self, h: _GPUWorkerHandle, texts: List[str]) -> List[List[float]]:
        rid = uuid.uuid4().hex
        with h._inflight_lock:
            h._inflight += 1
        h.req_q.put((rid, texts))
        return self._collect([rid], [h])

    def _collect(self, rids: List[str], targets: List[_GPUWorkerHandle],
                 timeout_s: float = 60.0) -> List[List[float]]:
        results: Dict[str, dict] = {}
        deadline = time.time() + timeout_s
        with self._pending_cond:
            while len(results) < len(rids):
                for rid in rids:
                    if rid not in results and rid in self._pending:
                        results[rid] = self._pending.pop(rid)
                if len(results) < len(rids):
                    remaining = max(0.1, deadline - time.time())
                    if remaining <= 0:
                        raise TimeoutError(
                            f"[embed_pool] timeout after {timeout_s}s waiting for "
                            f"{len(rids)-len(results)}/{len(rids)} shards"
                        )
                    self._pending_cond.wait(timeout=min(2.0, remaining))
        flat: List[List[float]] = []
        for rid in rids:
            flat.extend(results[rid].get("embeddings", []))
        return flat

    # ──────────────────────────────────────────────────────────────
    # Hot add / remove (GPU 2 swap protocol)
    # ──────────────────────────────────────────────────────────────
    def add_gpu(self, gpu_id: int) -> dict:
        if gpu_id in self.workers and self.workers[gpu_id].state != "dead":
            return {"ok": False, "reason": f"gpu {gpu_id} already in pool"}
        if gpu_id not in PROFILES:
            return {"ok": False, "reason": f"no profile for gpu {gpu_id}"}
        h = _GPUWorkerHandle(gpu_id, PROFILES[gpu_id], self._ctx)
        h.start(self._ctx, MODEL_PATH, self._ready_q)
        self.workers[gpu_id] = h
        # Spawn router for this worker
        t = threading.Thread(target=self._route_one, args=(h,), daemon=True,
                             name=f"embed-router-gpu{gpu_id}")
        t.start()
        self._routers.append(t)
        # Wait briefly for ready
        deadline = time.time() + h.profile.get("load_timeout_s", 120)
        while time.time() < deadline:
            try:
                info = self._ready_q.get(timeout=1.0)
                gid = info["gpu"]
                if gid in self.workers:
                    if info.get("ok"):
                        self.workers[gid].mark_ready(info)
                    else:
                        self.workers[gid].mark_failed(info)
                if gid == gpu_id:
                    return {"ok": h.state == "ready", "state": h.state,
                            "load_s": h.load_s, "error": h.error}
            except Exception:
                continue
        h.state = "dead"
        h.error = "add_gpu timeout"
        return {"ok": False, "reason": "load timeout"}

    def remove_gpu(self, gpu_id: int) -> dict:
        h = self.workers.get(gpu_id)
        if h is None:
            return {"ok": False, "reason": f"gpu {gpu_id} not in pool"}
        h.state = "dead"   # Stop routing to it
        try:
            h.req_q.put(None)  # Signal worker to exit
        except Exception:
            pass
        if h.proc and h.proc.is_alive():
            h.proc.join(timeout=10)
            if h.proc.is_alive():
                h.proc.terminate()
        del self.workers[gpu_id]
        return {"ok": True, "gpu": gpu_id}

    def health(self) -> dict:
        return {
            "pool_size": len(self.workers),
            "live": [g for g, h in self.workers.items() if h.state in ("ready", "degraded")],
            "ready": [g for g, h in self.workers.items() if h.state == "ready"],
            "degraded": [g for g, h in self.workers.items() if h.state == "degraded"],
            "dead": [g for g, h in self.workers.items() if h.state == "dead"],
            "workers": [h.stats() for h in self.workers.values()],
        }


# ──────────────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────────────

_pool: Optional[_Pool] = None
_pool_lock = threading.Lock()


def get_pool() -> _Pool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _Pool(DEFAULT_GPUS)
    return _pool


# CLI smoke test:
#   EMBED_GPUS=0,1,3,4,5,6 python3 embedder_direct.py
if __name__ == "__main__":
    import sys, json as _json
    p = get_pool()
    print(_json.dumps(p.health(), indent=2))
    if "--bench" in sys.argv:
        print("\n[bench] solo per-GPU x 50 calls each:")
        for gid in sorted(p.workers.keys()):
            if p.workers[gid].state != "ready":
                continue
            t0 = time.time()
            for _ in range(50):
                p.embed_on(gid, ["the quick brown fox jumps over the lazy dog"])
            dt = time.time() - t0
            print(f"  gpu{gid}: {dt:.2f}s for 50 calls = {50/dt:.1f} q/s, p50={p.workers[gid].stats()['p50_ms']}ms")
        print("\n[bench] burst 600 round-robin retrieves:")
        t0 = time.time()
        for _ in range(600):
            p.embed(["the quick brown fox jumps over the lazy dog"])
        dt = time.time() - t0
        print(f"  600 calls in {dt:.2f}s = {600/dt:.1f} q/s")
        print("\n[bench] post-burst health:")
        print(_json.dumps(p.health(), indent=2))
