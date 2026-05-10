#!/usr/bin/env python3
"""api_v2_shadow.py (B5) — shadow dual-writer for v1 → v2 cutover.

Mounted alongside api.py. Adds:
  POST /query_v2           — queries cbic_v2 directly (θ-gated via theta_v2.json)
  POST /query_shadow       — queries both cbic_v1 AND cbic_v2, returns v1 as primary,
                              logs divergence. This is what dashboard/Open WebUI hits
                              during the cutover window.
  GET  /shadow/status      — returns divergence stats + kill-switch state.
  POST /shadow/kill        — manually trip kill-switch (admin).
  POST /shadow/resume      — clear kill-switch (admin).

Kill-switch: if rolling divergence > DIVERGENCE_THRESHOLD (2%) over
KILL_WINDOW requests, auto-flip to v1-only and log to JOURNAL.

Divergence metric: 1 - |top_k_v1 ∩ top_k_v2| / k  (Jaccard-style on chunk_ids).

Env:
  QDRANT_URL, QDRANT_COLL_V1=cbic_v1, QDRANT_COLL_V2=cbic_v2
  SHADOW_LOG=/opt/indian-legal-ai/data/shadow_log.jsonl
  THETA_V2_PATH=/opt/indian-legal-ai/reingest_spec/theta_v2.json
  DIVERGENCE_THRESHOLD=0.02  KILL_WINDOW=200
"""
from __future__ import annotations
import os, json, time, threading, collections
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6343")
COLL_V1 = os.environ.get("QDRANT_COLL_V1", "cbic_v1")
COLL_V2 = os.environ.get("QDRANT_COLL_V2", "cbic_v2")
SHADOW_LOG = Path(os.environ.get(
    "SHADOW_LOG", "/opt/indian-legal-ai/data/shadow_log.jsonl"))
THETA_PATH = Path(os.environ.get(
    "THETA_V2_PATH", "/opt/indian-legal-ai/reingest_spec/theta_v2.json"))
DIVERGENCE_THRESHOLD = float(os.environ.get("DIVERGENCE_THRESHOLD", "0.02"))
KILL_WINDOW = int(os.environ.get("KILL_WINDOW", "200"))


# --- State (process-local; dashboard reads via /shadow/status) ---------------

_state_lock = threading.Lock()
_recent = collections.deque(maxlen=KILL_WINDOW)  # list[float divergence]
_kill_switch = {"tripped": False, "reason": None, "ts": None}
_stats = {"shadow_n": 0, "v2_only_n": 0, "errors_v1": 0, "errors_v2": 0}


def _load_theta() -> Optional[float]:
    if not THETA_PATH.exists():
        return None
    try:
        return json.loads(THETA_PATH.read_text()).get("theta")
    except Exception:
        return None


def _jaccard_divergence(v1_ids: List[str], v2_ids: List[str]) -> float:
    if not v1_ids and not v2_ids:
        return 0.0
    s1, s2 = set(v1_ids), set(v2_ids)
    denom = max(len(s1), len(s2), 1)
    return 1.0 - (len(s1 & s2) / denom)


def _record_divergence(d: float, payload: dict) -> None:
    with _state_lock:
        _recent.append(d)
        if len(_recent) >= KILL_WINDOW:
            avg = sum(_recent) / len(_recent)
            if avg > DIVERGENCE_THRESHOLD and not _kill_switch["tripped"]:
                _kill_switch.update({
                    "tripped": True,
                    "reason": f"avg_divergence={avg:.4f} over {KILL_WINDOW} req",
                    "ts": time.time(),
                })
                print(f"[shadow] KILL-SWITCH TRIPPED {_kill_switch['reason']}")
    try:
        SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SHADOW_LOG.open("a") as fh:
            fh.write(json.dumps({"ts": time.time(), "divergence": d, **payload}) + "\n")
    except Exception as e:
        print(f"[shadow] log-write failed: {e}")


def _call_collection(question: str, k: int, collection: str) -> dict:
    """Call the in-process retriever against a specific collection.
    H1 fix: pass collection through retriever.retrieve()'s new `collection`
    kwarg instead of mutating os.environ["QDRANT_COLL"] — the env-mutation
    approach was racy under concurrent /query_shadow calls (j1 and j2 threads
    would stomp each other's collection setting)."""
    # Import lazily to avoid pulling heavy deps at module load.
    from retriever import retrieve, rerank  # type: ignore
    from hyde import hyde  # type: ignore
    try:
        hyde_q = hyde(question) if os.environ.get("USE_HYDE", "1") == "1" else question
        hits = retrieve(hyde_q, k=k * 3, collection=collection)
        hits = rerank(question, hits)[:k]
        return {"hits": [{"chunk_id": h.get("chunk_id"),
                          "score": h.get("score", 0),
                          "payload": {k2: v for k2, v in h.items()
                                      if k2 in ("doc_id", "section_ref", "text")}}
                         for h in hits]}
    except Exception as e:
        return {"error": str(e), "hits": []}


# --- Router -----------------------------------------------------------------

router = APIRouter()


class ShadowReq(BaseModel):
    question: str
    k: int = 8


@router.post("/query_v2")
def query_v2(req: ShadowReq) -> Dict[str, Any]:
    theta = _load_theta()
    resp = _call_collection(req.question, req.k, COLL_V2)
    if theta is not None:
        resp["theta"] = theta
        top = resp["hits"][0]["score"] if resp["hits"] else 0
        resp["refused"] = top < theta
        if resp["refused"]:
            resp["hits"] = []
    with _state_lock:
        _stats["v2_only_n"] += 1
        if "error" in resp:
            _stats["errors_v2"] += 1
    return resp


@router.post("/query_shadow")
def query_shadow(req: ShadowReq) -> Dict[str, Any]:
    """Primary: v1. Shadow: v2. Returns v1 to the caller but logs divergence."""
    if _kill_switch["tripped"]:
        # v1-only mode after kill
        r1 = _call_collection(req.question, req.k, COLL_V1)
        with _state_lock:
            _stats["shadow_n"] += 1
        return {"primary": "v1", "kill_switch": True, **r1}

    # Parallel calls
    r1_holder: Dict[str, Any] = {}
    r2_holder: Dict[str, Any] = {}

    def j1(): r1_holder.update(_call_collection(req.question, req.k, COLL_V1))
    def j2(): r2_holder.update(_call_collection(req.question, req.k, COLL_V2))
    t1 = threading.Thread(target=j1); t2 = threading.Thread(target=j2)
    t1.start(); t2.start(); t1.join(timeout=45); t2.join(timeout=45)

    v1_ids = [h.get("chunk_id") for h in r1_holder.get("hits", [])]
    v2_ids = [h.get("chunk_id") for h in r2_holder.get("hits", [])]
    d = _jaccard_divergence(v1_ids, v2_ids)

    with _state_lock:
        _stats["shadow_n"] += 1
        if "error" in r1_holder: _stats["errors_v1"] += 1
        if "error" in r2_holder: _stats["errors_v2"] += 1

    _record_divergence(d, {"q": req.question[:200],
                           "v1_ids": v1_ids, "v2_ids": v2_ids})

    return {
        "primary": "v1",
        "divergence": d,
        "kill_switch": False,
        "v1": r1_holder,
        "v2_shadow": r2_holder,
    }


@router.get("/shadow/status")
def shadow_status() -> Dict[str, Any]:
    with _state_lock:
        n = len(_recent)
        avg = (sum(_recent) / n) if n else 0.0
        return {
            "kill_switch": dict(_kill_switch),
            "window_size": n,
            "window_capacity": KILL_WINDOW,
            "rolling_avg_divergence": round(avg, 4),
            "threshold": DIVERGENCE_THRESHOLD,
            "stats": dict(_stats),
            "theta_v2": _load_theta(),
        }


@router.get("/shadow/recent")
def shadow_recent(n: int = 20) -> Dict[str, Any]:
    """Tail the last n divergence records from SHADOW_LOG."""
    n = max(1, min(int(n), 200))
    items: List[dict] = []
    try:
        if SHADOW_LOG.exists():
            # Efficient tail: read last ~64KB, split lines, take last n.
            with SHADOW_LOG.open("rb") as fh:
                try:
                    fh.seek(0, 2)
                    size = fh.tell()
                    read_bytes = min(size, 64 * 1024)
                    fh.seek(size - read_bytes)
                    tail = fh.read().decode("utf-8", errors="replace")
                except Exception:
                    fh.seek(0)
                    tail = fh.read().decode("utf-8", errors="replace")
            lines = [ln for ln in tail.splitlines() if ln.strip()]
            for ln in lines[-n:]:
                try:
                    items.append(json.loads(ln))
                except Exception:
                    continue
    except Exception as e:
        return {"ok": False, "error": str(e), "items": []}
    return {"ok": True, "items": items}


@router.post("/shadow/kill")
def shadow_kill(reason: str = "manual") -> Dict[str, Any]:
    with _state_lock:
        _kill_switch.update({"tripped": True, "reason": reason, "ts": time.time()})
    return dict(_kill_switch)


@router.post("/shadow/resume")
def shadow_resume() -> Dict[str, Any]:
    with _state_lock:
        _kill_switch.update({"tripped": False, "reason": None, "ts": None})
        _recent.clear()
    return dict(_kill_switch)


# Convenience: mount on a FastAPI app
def attach(app) -> None:
    app.include_router(router)
