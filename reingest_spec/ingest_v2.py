#!/usr/bin/env python3
"""
ingest_v2.py — Orchestrator for CBIC RAG v2 re-ingestion.

EXTENDS the existing cbic_rag/ingest.py pipeline. Does NOT re-implement.

Changes from v1:
- Swaps old `chunker.chunk_doc` for `reingest_spec.chunker_v2.classify_and_chunk`
  (TWO-PASS: Pass-1 Claude CLI chunking_plan → Pass-2 R1–R7 rules).
- Pipes chunks through `reingest_spec.dedupe_chunks.ChunkDeduper` (V21 PASS,
  31.7% savings) before upsert.
- Tags every chunk with `reingest_spec.topic_tagger` multi-label topics (V20 PASS).
- Targets new collection `cbic_v2` (keeps `cbic_v1` as rollback per D10).
- Resumable per phase via manifest v2 sqlite: ingest_manifest_v2.sqlite.

Phases implemented (driven by --phase flag or all):
  phase1  Build manifest v2 + bilingual twin linking
  phase2  Chunk (two-pass) + dedupe + topic-tag → emit chunk dicts to sqlite
  phase3  Dense embed (BGE-M3 pool, reuse embedder.embed_dense_bulk)
  phase4  Sparse embed (fastembed BM25, reuse embedder.embed_sparse_batch)
  phase5  Upsert to Qdrant cbic_v2

Run (on rig):
    python3 ingest_v2.py --phase all --resume
    python3 ingest_v2.py --phase phase2 --limit 50   # debug single phase

Env (inherits from cbic_rag/ingest.py):
  CBIC_MANIFEST=/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite
  MANIFEST_V2=/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite
  QDRANT_URL=http://127.0.0.1:6343
  QDRANT_COLL_V2=cbic_v2
  LLM_URL=http://127.0.0.1:9082          # qwen3-14b for Pass-1 classify (D1 primary)
  EMBED_GPUS=0,1,4,5,6                   # skip GPU 2 (qwen3) and GPU 3 (SMU fault)
  PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag

HARD PRE-FLIGHT (added 2026-04-23 after Ollama-trap incident):
  Before ANY phase runs, this script now:
    1. Asserts embedder.py is the llama-cpp Vulkan facade, NOT Ollama-based
       (checks for the sentinel `_FACADE_VERSION = "direct-v1"` in embedder module).
    2. Asserts EMBED_GPUS is set and excludes GPU 2 (qwen3) + GPU 3 (SMU-faulted).
    3. Pings qwen3 at LLM_URL with a tiny completion; fails if no response or
       HTTP != 200. (Phase 2 will hit it 14,925× in production — no point starting
       a 25-hour run if the service is dead.)
    4. Runs a hello-world embed (1 short text) through the real pool on the real
       GPUs and verifies: shape==(1,1024), values are finite and non-zero-variance.
       Catches the "Ollama fell back to CPU" class of failure BEFORE Phase 3.
    5. At end of phase3_4_5, asserts qdrant points_count == canonical_chunks
       submitted. Catches silent-success (embed returned 0 rows without error).

  Bypass (for component-only smoke): --no-preflight. NEVER bypass on a real run.
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cbic_rag"))

# v2 components (all proven)
from chunker_v2 import classify_and_chunk, ChunkingPlan, Chunk  # noqa: E402
from dedupe_chunks import ChunkDeduper                          # noqa: E402
import topic_tagger                                              # noqa: E402

# Reuse v1 infrastructure unchanged
try:
    import embedder                                              # noqa: E402
except ImportError:
    embedder = None  # only needed for phase3/4/5 on rig

MANIFEST_V1 = os.environ.get("CBIC_MANIFEST", "/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite")
MANIFEST_V2 = os.environ.get("MANIFEST_V2", "/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6343")
QDRANT_COLL_V2 = os.environ.get("QDRANT_COLL_V2", "cbic_v2")
BATCH = int(os.environ.get("EMBED_BATCH", "48"))
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:9082")

# Hardened default: skip GPU 2 (qwen3-14b host), GPU 3 (SMU-faulted),
# and GPUs 0/1 (documented D-state hangs in known_good_configs.md).
# Only proven-stable embed GPUs remain: 4, 5, 6.
# Preflight REJECTS if caller puts a forbidden GPU in EMBED_GPUS.
# (Changed 2026-04-24 per user directive: use ones that work, not ones that hang.)
EMBED_GPUS_DEFAULT = "1,3,4,5,6"
EMBED_GPUS_FORBIDDEN = {"0", "2"}  # 2026-04-25: GPU 0=reranker, GPU 2=qwen3
if not os.environ.get("EMBED_GPUS"):
    os.environ["EMBED_GPUS"] = EMBED_GPUS_DEFAULT


# --- Hard preflight (added 2026-04-23) --------------------------------------
# Catches the class of failures that burned 12h on 2026-04-23:
#   - embedder.py silently regressed to Ollama (CPU fallback → 1024-d zeros)
#   - qwen3 service not running → Phase 2 would 14,925× HTTP timeout
#   - EMBED_GPUS clashes with qwen3 host GPU
#   - Pool comes up but embed returns no-op (silent success)
# NEVER bypass --no-preflight on a real run.


def _preflight_python_stack():
    """Gap-4 fix: fail-fast with an actionable message if the rig's Python stack
    is wrong. Without this, a missing llama_cpp gives a cryptic ImportError deep
    in pool init; this check surfaces the real cause (wrong PYTHONPATH or wrong
    python binary) in the first second of preflight."""
    import sys as _sys
    try:
        import llama_cpp as _lc
    except ImportError as e:
        raise RuntimeError(
            f"[preflight FAIL] llama_cpp not importable: {e}\n"
            f"  python={_sys.executable}\n"
            f"  PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}\n"
            f"  Fix: export PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:"
            f"/opt/indian-legal-ai/rag/cbic_rag\n"
            f"  Ensure you are using /usr/bin/python3 (not a venv python)."
        )
    if not _lc.llama_supports_gpu_offload():
        raise RuntimeError(
            "[preflight FAIL] llama_cpp is importable but has no GPU offload support. "
            "Rebuild with Vulkan via /opt/indian-legal-ai/scripts/cbic_ingest/build-llamacpp-vulkan.sh")
    print(f"[preflight OK] llama_cpp {getattr(_lc, '__version__', '?')} with GPU offload")


def _preflight_embedder_facade():
    """Assert embedder module is the llama-cpp Vulkan facade, not Ollama.
    The facade defines `_FACADE_VERSION = "direct-v1"` at module scope and
    re-exports `embed_dense_batch` from `embedder_direct`. If that sentinel
    is missing OR `ollama` / `11434` appears in embedder source, raise."""
    if embedder is None:
        raise RuntimeError("embedder not importable — cannot preflight. "
                           "Check PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag")
    version = getattr(embedder, "_FACADE_VERSION", None)
    if version != "direct-v1":
        raise RuntimeError(
            f"[preflight FAIL] embedder._FACADE_VERSION={version!r}, expected 'direct-v1'. "
            f"embedder.py has been REGRESSED (probably to Ollama HTTP). "
            f"Restore via: cp /opt/indian-legal-ai/scripts/cbic_ingest/embedder.py.direct.ref "
            f"/opt/indian-legal-ai/rag/cbic_rag/embedder.py")
    # Belt and braces: scan source for the Ollama HTTP-embed smell. Use port
    # numbers (11434-11439 are Ollama-specific) not the word "ollama" — our own
    # regression-history docstring mentions the word, and that's fine. A real
    # regression imports `requests`/`httpx` and hits those ports.
    import inspect
    try:
        src = inspect.getsource(embedder)
        for bad in ("11434", "11435", "11436", "11437", "11438", "11439"):
            if bad in src:
                raise RuntimeError(
                    f"[preflight FAIL] embedder source references port {bad} — "
                    f"Ollama HTTP has leaked back in. Restore from .direct.ref.")
    except (OSError, TypeError):
        pass  # source not available (compiled?) — version check is the real gate
    print(f"[preflight OK] embedder facade version={version}")


def _preflight_embed_gpus():
    gpus = [g.strip() for g in os.environ.get("EMBED_GPUS", "").split(",") if g.strip()]
    if not gpus:
        raise RuntimeError("[preflight FAIL] EMBED_GPUS is empty")
    bad = [g for g in gpus if g in EMBED_GPUS_FORBIDDEN]
    if bad:
        raise RuntimeError(
            f"[preflight FAIL] EMBED_GPUS includes forbidden GPUs {bad}: "
            f"GPU 2 hosts qwen3-14b (:9082 — Phase 2 uses it), "
            f"GPU 3 has SMU fault (hangs under load). "
            f"Use EMBED_GPUS={EMBED_GPUS_DEFAULT}")
    print(f"[preflight OK] EMBED_GPUS={gpus}")


def _preflight_qwen3(timeout=10):
    """Ping qwen3 at LLM_URL with a trivial completion. If Phase 2 is about to
    hit this service ~14,925 times, we want to know in 10s not 25h whether it's
    alive. Accepts llama-server or OpenAI-compatible endpoints."""
    import urllib.request
    import urllib.error
    # llama-server health endpoint (proven by V2b)
    for path in ("/health", "/v1/models", "/"):
        try:
            req = urllib.request.Request(LLM_URL.rstrip("/") + path)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    print(f"[preflight OK] qwen3 reachable at {LLM_URL}{path}")
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            continue
    raise RuntimeError(
        f"[preflight FAIL] qwen3-14b not reachable at {LLM_URL}. "
        f"Phase 2 Pass-1 classification depends on it. "
        f"Start it or set LLM_URL to a live host.")


def _preflight_qwen3_warmup(timeout=180):
    """Send a real-shape completion call so qwen3 compiles its Vulkan shaders
    BEFORE phase2 starts. Without this, the very first classify hits a
    cold-shader path that takes 60-120s and frequently crashes the server.
    Origin: 2026-04-24 GST50 run — first classify timed out at 60s and the
    crash cascaded all 43 docs to FAIL until systemd restarted qwen3."""
    import json as _json, urllib.request, urllib.error, time as _t
    body = {
        "prompt": "Reply with the JSON object {\"ok\": true} only.",
        "max_tokens": 16, "temperature": 0.0,
    }
    req = urllib.request.Request(
        LLM_URL.rstrip("/") + "/v1/completions",
        method="POST",
        data=_json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = _t.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = _json.loads(resp.read())
        text = payload["choices"][0]["text"]
        print(f"[preflight OK] qwen3 warmup completion in {_t.time()-t0:.1f}s "
              f"(text={text!r:.40})")
    except Exception as e:
        # Non-fatal: warmup is best-effort. If qwen3 was reachable in the prior
        # ping, phase2 will also succeed but the first doc may be slow.
        print(f"[preflight WARN] qwen3 warmup failed in {_t.time()-t0:.1f}s: "
              f"{type(e).__name__}: {e} — proceeding, but expect first-doc slowness")


def _preflight_classify_latency_slo(max_tokens: int = 512, p95_budget_s: float = 15.0,
                                    timeout: int = 30):
    """HARD SLO: classify-shape qwen3 call must finish in <= p95_budget_s.

    Codified 2026-04-26 after Set 5 chunkfix re-ingest stalled at 130min for
    100 docs. Root cause: max_tokens was bumped 200->1024->2048->4096 over
    multiple sessions to fix unrelated truncation defects. /no_think prefix
    only takes effect on /v1/chat/completions (with chat template); on raw
    /v1/completions it is literal text qwen3 ignores. Result: qwen3 burned
    80s/call generating 2400 tokens of reasoning. Nobody measured wall-clock
    per-call so the regression hid behind "it works".

    Standing rule: phase 2 budget ~5min for 100 docs = 3s/call median.
    p95 SLO of 15s is a safety ceiling. If preflight exceeds 15s, FAIL —
    do NOT proceed into a multi-hour serial loop with a degraded classifier.
    """
    import json as _json, urllib.request, time as _t
    prompt = ("/no_think You are a JSON classifier. Reply with ONLY this JSON "
              "object and nothing else: {\"doc_type\": \"notification\", "
              "\"primary_splitter\": \"section\", \"target_chunk_size\": 3500}")
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "stop": ["\n\n\n"]}
    req = urllib.request.Request(
        LLM_URL.rstrip("/") + "/v1/completions",
        method="POST",
        data=_json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = _t.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = _json.loads(resp.read())
        elapsed = _t.time() - t0
        text = payload["choices"][0]["text"]
        n_tokens = len(text.split())
    except Exception as e:
        elapsed = _t.time() - t0
        raise RuntimeError(
            f"[preflight FAIL] qwen3 classify-shape call failed in {elapsed:.1f}s: "
            f"{type(e).__name__}: {e}. Phase 2 will hit qwen3 N-times — refusing "
            f"to proceed with a degraded classifier."
        ) from e
    if elapsed > p95_budget_s:
        raise RuntimeError(
            f"[preflight FAIL] qwen3 classify latency {elapsed:.1f}s > SLO {p95_budget_s}s. "
            f"At this rate, phase 2 for 100 docs takes {elapsed*100/60:.0f}min "
            f"(SLO target ~5min). Likely causes: max_tokens too high (check chunker_v2 "
            f"classify_doc_qwen, target <=512), qwen3 in thinking mode (/no_think only "
            f"works on /v1/chat/completions), or qwen3 service degraded. Fix and re-run. "
            f"Tokens emitted: {n_tokens}."
        )
    print(f"[preflight OK] qwen3 classify latency {elapsed:.2f}s "
          f"(SLO <= {p95_budget_s}s, tokens={n_tokens})")




def _preflight_hello_world_embed():
    """Real embed on real hardware. Catches the class 'pool inits but returns
    no-op' (Ollama CPU fallback, etc). The probe: embed one non-trivial sentence,
    verify (1) shape, (2) finite, (3) non-zero variance."""
    import math
    t0 = time.time()
    try:
        vecs = embedder.embed_dense_batch(
            ["Central Board of Indirect Taxes and Customs governs GST in India."]
        )
    except Exception as e:
        raise RuntimeError(f"[preflight FAIL] hello-world embed raised: {e}")
    if not vecs or len(vecs) != 1:
        raise RuntimeError(f"[preflight FAIL] embed returned {len(vecs) if vecs else 0} vectors, want 1")
    v = list(vecs[0])
    dim = getattr(embedder, "DENSE_DIM", 1024)
    if len(v) != dim:
        raise RuntimeError(f"[preflight FAIL] embed dim={len(v)}, want {dim}")
    if not all(math.isfinite(x) for x in v):
        raise RuntimeError("[preflight FAIL] embed vector has non-finite values")
    mean = sum(v) / len(v)
    var = sum((x - mean) ** 2 for x in v) / len(v)
    if var < 1e-8:
        raise RuntimeError(
            f"[preflight FAIL] embed vector variance={var:.2e} ≈ 0. "
            f"Pool is almost certainly returning zeros (Ollama CPU fallback?). "
            f"Sample: {v[:6]}")
    print(f"[preflight OK] hello-world embed: dim={len(v)} var={var:.4f} in {time.time()-t0:.1f}s")


def _preflight_vulkan_env():
    """Mandatory Vulkan/Navi 10 env vars. Codified 2026-04-24 after GST50
    embed boot hung for 30 minutes because RADV_DEBUG=nodcc and
    GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 were missing from the launch env.
    Both are listed as non-negotiable in known_good_configs.md (MEMORY.md).

    RADV_DEBUG=nodcc              — disable delta color compression (shader
                                    compile hangs on Navi 10 without it)
    GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 — Navi 10 int-dot path is broken;
                                    GGML crashes / returns zeros without it

    Auto-sets defaults if missing (so single-invocation CLI runs still work),
    but LOGS the fact we did — silently setting env would hide future
    regressions.
    """
    required = {
        "RADV_DEBUG": "nodcc",
        "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1",
    }
    patched = []
    for k, v in required.items():
        if not os.environ.get(k):
            os.environ[k] = v
            patched.append(f"{k}={v}")
    if patched:
        print(f"[preflight WARN] auto-set missing Vulkan env: {', '.join(patched)} "
              f"(both are MANDATORY per known_good_configs.md — systemd/shell "
              f"launcher should set these explicitly)")
    else:
        print(f"[preflight OK] Vulkan env: RADV_DEBUG=nodcc, "
              f"GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1")


def _preflight_fitz():
    """Fitz (PyMuPDF) MUST be installed. Pre-2026-04-24, _read_pdf_text fell
    back silently to pdfminer which returned page_offsets=[0,total_len] (2
    entries regardless of page count), poisoning Pass-1 table_regions
    corpus-wide. G1 recall dropped to 0.51 because of this single silent
    regression. Check before any phase that reads PDFs."""
    try:
        import fitz  # noqa: F401
        print(f"[preflight OK] PyMuPDF (fitz) version={fitz.__doc__.splitlines()[0] if fitz.__doc__ else 'present'}")
    except ImportError as e:
        raise RuntimeError(
            "[preflight FAIL] PyMuPDF (fitz) not installed. The pdfminer "
            "fallback was REMOVED — it silently returned "
            "page_offsets=[0, total_len] and poisoned Pass-1 plans corpus-wide. "
            "Install with: /opt/indian-legal-ai/rag/cbic_rag/venv/bin/pip install pymupdf"
        ) from e


def run_preflight():
    """Top-to-bottom preflight. Raises on any failure. Called from main() before
    any phase runs unless --no-preflight is passed."""
    print("[preflight] starting hard preflight checks (added 2026-04-23)")
    _preflight_vulkan_env()        # 2026-04-24: RADV/GGML_VK env vars mandatory on Navi 10
    _preflight_fitz()              # 2026-04-24: fitz mandatory — no silent pdfminer fallback
    _preflight_python_stack()      # gap 4: actionable errors for wrong python/PYTHONPATH
    _preflight_embedder_facade()
    _preflight_embed_gpus()
    _preflight_qwen3()
    _preflight_qwen3_warmup()      # 2026-04-24: compile Vulkan shaders + alloc KV BEFORE phase2
    _preflight_classify_latency_slo()  # 2026-04-26: HARD SLO on classify wall-clock — fails fast on /no_think regression / max_tokens bloat
    _preflight_hello_world_embed()
    print("[preflight] all checks PASS — safe to run phases")


# --- Manifest v2 schema -----------------------------------------------------


def init_manifest_v2(path: str = MANIFEST_V2):
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS docs (
      doc_id TEXT PRIMARY KEY,
      category TEXT, subcategory TEXT, title TEXT,
      path_en TEXT, path_hi TEXT,
      lang TEXT, has_twin INTEGER DEFAULT 0,
      twin_doc_id TEXT,
      phase1_done INTEGER DEFAULT 0,
      phase2_done INTEGER DEFAULT 0,
      plan_json TEXT,
      plan_confidence REAL
    );
    CREATE TABLE IF NOT EXISTS chunks (
      chunk_id TEXT PRIMARY KEY,      -- canonical SHA256
      doc_id TEXT,
      payload_json TEXT,
      is_canonical INTEGER DEFAULT 1, -- 0 if dup_of_chunk_id is set
      dup_of_chunk_id TEXT,
      embedded INTEGER DEFAULT 0,
      upserted INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks(embedded);
    CREATE INDEX IF NOT EXISTS idx_chunks_upserted ON chunks(upserted);
    """)
    # 2026-04-24 FAILURE-REPORTING MIGRATION (A-to-Z silent-drop audit fix).
    # Adds per-phase error columns so a doc that drops out of phase2/phase3-5
    # records WHY and WHEN in the manifest itself — no more reliance on
    # ephemeral stdout prints. Idempotent via try/except on duplicate-column.
    # A run that ends with any phase2_status='failed' or phase3_status='failed'
    # MUST exit non-zero (enforced in phase2/phase3_4_5 end-of-phase guards).
    for stmt in (
        "ALTER TABLE docs ADD COLUMN phase2_status TEXT",
        "ALTER TABLE docs ADD COLUMN phase2_error TEXT",
        "ALTER TABLE docs ADD COLUMN phase2_failed_at REAL",
        "ALTER TABLE docs ADD COLUMN phase3_status TEXT",
        "ALTER TABLE docs ADD COLUMN phase3_error TEXT",
        "ALTER TABLE docs ADD COLUMN phase3_failed_at REAL",
    ):
        try:
            c.execute(stmt)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
    c.commit()
    return c


def _record_phase_failure(c2, doc_id: str, phase: str, reason: str) -> None:
    """Record a per-doc phase failure to the manifest. Phase is 'phase2' or 'phase3'.
    Truncates reason to 4kB to keep the manifest compact. Never swallows itself —
    if the UPDATE fails the exception propagates so the caller sees it."""
    import time as _t
    reason = (reason or "")[:4096]
    col_status = f"{phase}_status"
    col_error  = f"{phase}_error"
    col_ts     = f"{phase}_failed_at"
    c2.execute(
        f"UPDATE docs SET {col_status}=?, {col_error}=?, {col_ts}=? WHERE doc_id=?",
        ("failed", reason, _t.time(), doc_id),
    )
    c2.commit()


def _write_failure_report(path: str, records: list) -> None:
    """Emit a JSON failure artifact next to the ingest manifest. Run scripts
    MUST call this at end-of-phase and MUST exit non-zero if records is
    non-empty. Structured so downstream tooling can parse + alert."""
    import time as _t
    doc = {"ts": _t.time(), "count": len(records), "records": records}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


# --- Phase 1: manifest + twin linking --------------------------------------


def _doc_id_in_clause(doc_ids):
    """Build a safe SQL IN-clause fragment for a list of doc_ids. Returns ''
    if doc_ids is falsy. Quotes each id. Used by --doc-ids to scope any phase
    to a fixed subset (e.g. 5-doc end-to-end smoke)."""
    if not doc_ids:
        return ""
    safe = ",".join("'" + d.replace("'", "''") + "'" for d in doc_ids)
    return f" AND doc_id IN ({safe})"


def phase1(limit=None, resume=True, doc_ids=None):
    """Build v2 manifest from v1. Also links bilingual twins (closes
    LESSONS_APPLIED row 6 — twin_doc_id was NULL).

    Twin linking rules:
      - If a row has BOTH path_en AND path_hi, lang=bilingual and
        twin_doc_id=doc_id (self-twin; downstream knows this doc is its own HI
        counterpart — common case in CBIC where a single PDF contains both
        language versions side-by-side).
      - If a row has ONLY path_en (or only path_hi), we search the v1 manifest
        for another row with the same (category, subcategory, title) but the
        opposite language path populated. If found, twin_doc_id=that_doc_id.
      - Otherwise twin_doc_id stays NULL (monolingual doc, no twin exists).
    """
    print(f"[phase1] copying v1 manifest → v2 at {MANIFEST_V2}")
    c2 = init_manifest_v2()
    c1 = sqlite3.connect(MANIFEST_V1)
    c1.row_factory = sqlite3.Row
    q = "SELECT doc_id, category, subcategory, title, path_en, path_hi FROM docs WHERE (path_en IS NOT NULL OR path_hi IS NOT NULL)"
    if doc_ids:
        q += _doc_id_in_clause(doc_ids)
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = list(c1.execute(q))
    print(f"[phase1] {len(rows)} docs in v1 manifest")

    # Pre-index v1 by (category, subcategory, title) for cross-doc twin lookup.
    # Only built if doc_ids is not set — for scoped smoke runs, cross-doc
    # linking is irrelevant (the smoke set is a closed list).
    title_index: dict[tuple, list] = {}
    if not doc_ids:
        for tr in c1.execute(
            "SELECT doc_id, category, subcategory, title, path_en, path_hi FROM docs "
            "WHERE title IS NOT NULL"
        ):
            key = (tr["category"] or "", tr["subcategory"] or "", (tr["title"] or "").strip().lower())
            title_index.setdefault(key, []).append(dict(tr))

    ins = twinned = 0
    for r in rows:
        has_en, has_hi = bool(r["path_en"]), bool(r["path_hi"])
        if has_en and has_hi:
            lang = "bilingual"
            twin = r["doc_id"]           # self-twin: both languages in one doc
        elif has_en:
            lang = "en"
            twin = _find_twin(title_index, r, want_hi=True) if title_index else None
        else:
            lang = "hi"
            twin = _find_twin(title_index, r, want_hi=False) if title_index else None
        if twin:
            twinned += 1
        c2.execute("""INSERT OR REPLACE INTO docs
                      (doc_id,category,subcategory,title,path_en,path_hi,lang,has_twin,twin_doc_id,phase1_done)
                      VALUES (?,?,?,?,?,?,?,?,?,1)""",
                   (r["doc_id"], r["category"], r["subcategory"], r["title"],
                    r["path_en"], r["path_hi"], lang,
                    1 if twin else 0, twin))
        ins += 1
    c2.commit()
    print(f"[phase1] wrote {ins} docs to v2 manifest ({twinned} twin-linked)")


def _find_twin(title_index: dict, row, *, want_hi: bool):
    """Look up a bilingual twin of `row` in the title index. `want_hi=True`
    means we have an EN doc and are looking for the HI counterpart."""
    key = ((row["category"] or ""), (row["subcategory"] or ""),
           ((row["title"] or "").strip().lower()))
    for cand in title_index.get(key, []):
        if cand["doc_id"] == row["doc_id"]:
            continue
        if want_hi and cand.get("path_hi"):
            return cand["doc_id"]
        if not want_hi and cand.get("path_en"):
            return cand["doc_id"]
    return None


# --- Phase 2: chunking (two-pass) + dedupe + topic-tag ----------------------


def _read_pdf_text(path: str) -> tuple[str, list[int]]:
    """Extract full text + per-page char offsets.

    HARD REQUIREMENT (codified 2026-04-24): PyMuPDF (fitz) MUST be installed.
    Prior versions fell back silently to pdfminer, which returned
    page_offsets=[0, total_len] (only 2 entries regardless of page count).
    That poisoned Pass-1 table_regions corpus-wide — every "page 1-1" region
    collapsed to the whole document, starving the section splitter.
    Failure mode: dense G1 recall 0.51 on GST50 Customs Act.
    See reingest_spec/JOURNAL.md 2026-04-24 entry.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(
            "[_read_pdf_text] PyMuPDF (fitz) is not installed. Pre-2026-04-24 "
            "pdfminer fallback is REMOVED — it silently returned "
            "page_offsets=[0, total_len] and poisoned Pass-1 table_regions. "
            "Install with: /opt/indian-legal-ai/rag/cbic_rag/venv/bin/pip install pymupdf"
        ) from e
    doc = fitz.open(path)
    texts = []
    offsets = [0]
    for page in doc:
        t = page.get_text()
        texts.append(t)
        offsets.append(offsets[-1] + len(t))
    return "".join(texts), offsets


# G3 fix: OCR detection via text-density probe.
# If extracted text has <TEXT_DENSITY_MIN chars/page, the PDF is image-only.
# We tag text_source="ocr" (and skip unless OCR pipeline is live) rather than
# silently ingest image-only docs as "born" — which would poison retrieval per
# ocr_research_cbic.md (471 image-only PDFs are currently pending OCR thaw).
#
# 2026-05-07 amendment: introduce "low_density" tier. Some valid customs/CE
# instructions have 1k-2.5k chars across many pages (density < 200/pg) yet are
# real text, not image-only. Previously: dropped silently when no OCR cache hit.
# Now: emit chunks AND mark ocr_pending in payload so future OCR run can supersede.
# Truly image-only PDFs (total text < LOW_DENSITY_TOTAL_MIN) still defer.
TEXT_DENSITY_MIN = 200       # chars/page threshold — below this is sparse
LOW_DENSITY_TOTAL_MIN = 500  # total chars threshold — below this is truly OCR-needed

def detect_text_source(full_text: str, page_offsets: list[int]) -> str:
    n_pages = max(1, len(page_offsets) - 1)
    total = len(full_text.strip())
    density = total / n_pages
    if density >= TEXT_DENSITY_MIN:
        return "born"
    if total >= LOW_DENSITY_TOTAL_MIN:
        return "low_density"
    return "ocr"


def phase2(limit=None, resume=True, use_qwen_first=False, doc_ids=None):
    c2 = init_manifest_v2()
    q = ("SELECT doc_id,category,subcategory,title,path_en,path_hi,lang,plan_json "
         "FROM docs WHERE phase1_done=1")
    if resume:
        q += " AND phase2_done=0"
    if doc_ids:
        q += _doc_id_in_clause(doc_ids)
    if limit:
        q += f" LIMIT {int(limit)}"
    docs = list(c2.execute(q))
    print(f"[phase2] {len(docs)} docs to chunk")

    deduper = ChunkDeduper()
    t0 = time.time()
    total_raw = total_canonical = total_dup = 0

    # 2026-04-24 A-to-Z failure-reporting: no more silent continues. Every
    # exit-before-phase2_done=1 path records WHY to manifest + in-memory list,
    # and the end-of-phase guard raises if any doc failed unaccounted.
    failures: list = []  # list of {"doc_id","stage","reason"}
    for i, r in enumerate(docs):
        doc_id, cat, sub, title, path_en, path_hi, lang, plan_json = r
        # 2026-04-24: if plan_json already exists from a prior Pass-1, reuse it
        # so we don't re-run qwen3 classify (~20s/doc saved on re-chunk passes).
        reuse_plan = None
        if plan_json:
            try:
                import json as _json
                reuse_plan = _json.loads(plan_json)
            except Exception:
                reuse_plan = None
        path = path_en or path_hi
        # 2026-05-07: D-2a (NO_PDF) ingest path. If no PDF exists on disk but the OCR
        # cache has a doc_id-keyed entry, ingest the OCR text directly. Otherwise fall
        # through to the original missing-path failure branch.
        full_text = None
        page_offsets = None
        if not path or not os.path.exists(path):
            from pathlib import Path as _Path
            _docid_cache = _Path("/opt/indian-legal-ai/data/ocr_cache") / f"{doc_id.replace(':','_')}.txt"
            if _docid_cache.exists() and _docid_cache.stat().st_size > 500:
                full_text = _docid_cache.read_text(encoding="utf-8", errors="replace")
                page_offsets = [0]
                _pos = 0
                for _line in full_text.split(chr(10)):
                    if _line.startswith("--- PAGE "):
                        page_offsets.append(_pos)
                    _pos += len(_line) + 1
                if not page_offsets: page_offsets = [0]
                print(f"  [{i}] {doc_id} OCR-ONLY-INGEST (no PDF, doc_id-keyed cache) len={len(full_text)}")
            else:
                reason = f"missing_path_no_ocr: path_en={path_en!r} path_hi={path_hi!r} no doc_id-keyed cache hit"
                print(f"  [{i}] {doc_id} PHASE2-FAIL missing-path — {reason}")
                _record_phase_failure(c2, doc_id, "phase2", reason)
                failures.append({"doc_id": doc_id, "stage": "missing_path", "reason": reason})
                continue
        if full_text is None:
            try:
                full_text, page_offsets = _read_pdf_text(path)
            except Exception as e:
                reason = f"pdf_read_fail: {type(e).__name__}: {e}"
                print(f"  [{i}] {doc_id} PHASE2-FAIL pdf-read — {reason}")
                _record_phase_failure(c2, doc_id, "phase2", reason)
                failures.append({"doc_id": doc_id, "stage": "pdf_read", "reason": reason})
                continue

        text_source = detect_text_source(full_text, page_offsets)
        if text_source == "ocr":
            # Try OCR cache (Gemini/Claude pre-processed) before deferring.
            # Added 2026-04-24: many docs already have cached OCR output
            # keyed by SHA256 of PDF bytes in /opt/indian-legal-ai/data/ocr_cache/.
            import hashlib as _hl
            _cache_dir = Path("/opt/indian-legal-ai/data/ocr_cache")
            _cached_text = None
            for _cand in (path_en, path_hi):
                if not _cand or not os.path.exists(_cand):
                    continue
                try:
                    _h = _hl.sha256(open(_cand, "rb").read()).hexdigest()
                except Exception:
                    continue
                _ctxt = _cache_dir / f"{_h}.txt"
                if _ctxt.exists() and _ctxt.stat().st_size > 500:
                    _cached_text = _ctxt.read_text(encoding="utf-8", errors="replace")
                    break
            # 2026-05-07 fallback: OCR cache files keyed by doc_id (with `:` → `_`).
            # Discovered ~700 such files orphaned in cache because original lookup was sha-only.
            if not _cached_text:
                _docid_cache = _cache_dir / f"{doc_id.replace(':','_')}.txt"
                if _docid_cache.exists() and _docid_cache.stat().st_size > 500:
                    _cached_text = _docid_cache.read_text(encoding="utf-8", errors="replace")
                    print(f"  [{i}] {doc_id} OCR-CACHE-HIT via doc_id fallback len={len(_cached_text)}")
            if _cached_text:
                full_text = _cached_text
                # Rebuild page_offsets from '--- PAGE N ---' markers.
                page_offsets = []
                _pos = 0
                for _line in full_text.split(chr(10)):
                    if _line.startswith("--- PAGE "):
                        page_offsets.append(_pos)
                    _pos += len(_line) + 1
                if not page_offsets:
                    page_offsets = [0]
                text_source = "ocr_cached"
                print(f"  [{i}] {doc_id} PHASE2-OCR-CACHE-HIT len={len(full_text)}")
            else:
                reason = f"ocr_deferred: text_density below {TEXT_DENSITY_MIN}/pg, no ocr_cache hit"
                print(f"  [{i}] {doc_id} PHASE2-DEFER ocr — {reason}")
                c2.execute(
                    "UPDATE docs SET phase2_status=?, phase2_error=?, phase2_failed_at=? WHERE doc_id=?",
                    ("ocr_deferred", reason, time.time(), doc_id),
                )
                c2.commit()
                continue
        meta = {
            "doc_id": doc_id, "source": os.path.basename(path),
            "category": cat, "subcategory": sub, "title": title,
            "lang": lang, "text_source": text_source,
            "parent_hierarchy_text": title or "",
            # 2026-05-07: low-density docs (sparse-but-real text) are ingested with
            # ocr_pending=True so a future OCR run can supersede them.
            "ocr_pending": (text_source == "low_density"),
            # D8 amendment metadata placeholders — populated from Pass-1 plan below
            "notification_id": None, "as_of_date": None, "superseded_by": None,
            "effective_date": None,
        }
        try:
            plan, chunks = classify_and_chunk(
                full_text, meta, page_offsets=page_offsets,
                use_qwen_first=use_qwen_first,
                reuse_plan=reuse_plan,
            )
        except Exception as e:
            reason = f"classify_fail: {type(e).__name__}: {e}"
            print(f"  [{i}] {doc_id} PHASE2-FAIL classify — {reason}")
            _record_phase_failure(c2, doc_id, "phase2", reason)
            failures.append({"doc_id": doc_id, "stage": "classify", "reason": reason})
            continue

        # Even if classify_and_chunk returned cleanly it can produce ZERO
        # chunks on pathological input (empty plan, no splits). That is still
        # a silent-drop. Treat zero-chunks as a failure unless the doc was
        # explicitly marked empty by the plan (future extension).
        if not chunks:
            reason = "classify_empty: classify_and_chunk returned 0 chunks"
            print(f"  [{i}] {doc_id} PHASE2-FAIL empty-chunks — {reason}")
            _record_phase_failure(c2, doc_id, "phase2", reason)
            failures.append({"doc_id": doc_id, "stage": "empty_chunks", "reason": reason})
            continue

        # Topic-tag each chunk
        for ch in chunks:
            primary, scores = topic_tagger.tag_chunk(ch.text, category=cat)
            ch.topic_tags = list(scores.keys())

        # Dedupe via canonical SHA256.
        # NOTE: ChunkDeduper.add() expects a DICT with text/chunk_id/doc_id, not a bare
        # string, and returns (canonical_chunk_dict, is_new). The bare-string call was a
        # smoke-ingest bug (AttributeError on chunk.get). Fixed 2026-04-23.
        for ch in chunks:
            ch_dict = {"text": ch.text, "chunk_id": ch.chunk_id, "doc_id": ch.doc_id}
            canonical_chunk, is_new = deduper.add(ch_dict)
            total_raw += 1
            if is_new:
                total_canonical += 1
                _insert_chunk(c2, ch, dup_of=None)
            else:
                total_dup += 1
                # canonical_chunk["chunk_id"] is the v2 sha256 id of the first occurrence
                _insert_chunk(c2, ch, dup_of=canonical_chunk.get("chunk_id"))

        c2.execute("UPDATE docs SET phase2_done=1, phase2_status='ok', plan_json=?, plan_confidence=? WHERE doc_id=?",
                   (json.dumps(asdict(plan)), plan.confidence, doc_id))
        c2.commit()

        if (i + 1) % 10 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [phase2] {i+1}/{len(docs)} docs  raw={total_raw} canonical={total_canonical} "
                  f"dup={total_dup} rate={rate:.2f} doc/s")

    # 2026-04-24 FAILURE REPORTING — end-of-phase2 guard. Silent-drop audit:
    # compare input set to successful set and RAISE with structured summary
    # if any doc is unaccounted. Each failed doc must be in either:
    #   - phase2_status='ok' with phase2_done=1 (success)
    #   - phase2_status='failed' with a reason (explicit failure, recorded)
    #   - phase2_status='ocr_deferred' (expected skip for OCR queue)
    # Any other state (e.g. phase1_done=1, phase2_done=0, phase2_status IS NULL)
    # is a silent drop — that IS the bug that produced cbic_v2_smoke 2-of-10.
    input_doc_ids = [r[0] for r in docs]
    # Re-query final state for this scoped run
    if input_doc_ids:
        q_in = ",".join("?" * len(input_doc_ids))
        state_rows = list(c2.execute(
            f"SELECT doc_id, phase2_done, phase2_status FROM docs WHERE doc_id IN ({q_in})",
            input_doc_ids,
        ))
    else:
        state_rows = []
    ok_n        = sum(1 for _, d, s in state_rows if d == 1 and s == "ok")
    failed_docs = [(d_id, s) for d_id, d, s in state_rows if s == "failed"]
    defer_n     = sum(1 for _, d, s in state_rows if s == "ocr_deferred")
    unaccounted = [d_id for d_id, d, s in state_rows if d == 0 and s is None]

    print(f"[phase2 DONE] raw={total_raw} canonical={total_canonical} dup={total_dup} "
          f"savings={total_dup/max(1,total_raw)*100:.1f}% elapsed={time.time()-t0:.1f}s")
    print(f"[phase2 SUMMARY] input={len(input_doc_ids)} ok={ok_n} failed={len(failed_docs)} "
          f"ocr_deferred={defer_n} unaccounted={len(unaccounted)}")

    # Write structured failure report artifact beside the manifest.
    if failures or unaccounted:
        report_path = str(MANIFEST_V2) + ".phase2_failures.json"
        full_records = list(failures) + [
            {"doc_id": d, "stage": "unaccounted",
             "reason": "phase2 exited with phase2_done=0 and no phase2_status — silent drop"}
            for d in unaccounted
        ]
        _write_failure_report(report_path, full_records)
        print(f"[phase2 FAILURES] {len(full_records)} records written to {report_path}")
        for rec in full_records[:20]:
            print(f"  - {rec['doc_id']}  stage={rec['stage']}  reason={rec['reason'][:200]}")
        # HARD FAIL unless caller passed --allow-phase2-failures with a
        # high-enough budget. Unaccounted silent drops always raise (those are
        # bugs, not edge-case docs).
        _allow = globals().get("ALLOW_PHASE2_FAILURES", 0)
        if len(unaccounted) > 0 or len(failed_docs) > _allow:
            raise RuntimeError(
                f"phase2 FAILED: {len(failed_docs)} explicit failures + "
                f"{len(unaccounted)} unaccounted silent drops "
                f"(budget --allow-phase2-failures={_allow}). See {report_path}"
            )
        else:
            print(f"[phase2 SOFT-PASS] {len(failed_docs)} failures within "
                  f"--allow-phase2-failures={_allow} budget; proceeding")


def _canonical_id_of(deduper: ChunkDeduper, canonical_text: str) -> str:
    import hashlib
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def _insert_chunk(c2, ch: Chunk, dup_of: str | None):
    """Defect D fix (2026-04-25): if dup_of is set, do NOT INSERT OR REPLACE
    (which would overwrite the canonical row's doc_id, destroying ingest
    integrity for shared-PDF docs). Instead UPDATE the canonical row's
    payload_json to append this doc_id to linked_doc_ids. The dup is still
    recorded as a row so reconciliation/audit works, but with a doc-id-prefixed
    chunk_id to avoid PK collision with the canonical."""
    payload = ch.to_payload()
    if dup_of:
        # Append this doc_id to canonical's linked_doc_ids
        row = c2.execute('SELECT payload_json FROM chunks WHERE chunk_id=?',
                         (dup_of,)).fetchone()
        if row:
            cp = json.loads(row[0])
            link = set(cp.get('linked_doc_ids') or [])
            if ch.doc_id and ch.doc_id != cp.get('doc_id'):
                link.add(ch.doc_id)
                cp['linked_doc_ids'] = sorted(link)
                c2.execute('UPDATE chunks SET payload_json=? WHERE chunk_id=?',
                           (json.dumps(cp), dup_of))
        # Insert dup row with prefixed chunk_id to avoid PK collision
        dup_row_id = f'{ch.doc_id}::{ch.chunk_id}'
        c2.execute("""INSERT OR REPLACE INTO chunks
                      (chunk_id,doc_id,payload_json,is_canonical,dup_of_chunk_id)
                      VALUES (?,?,?,?,?)""",
                   (dup_row_id, ch.doc_id, json.dumps(payload), 0, dup_of))
    else:
        c2.execute("""INSERT OR REPLACE INTO chunks
                      (chunk_id,doc_id,payload_json,is_canonical,dup_of_chunk_id)
                      VALUES (?,?,?,?,?)""",
                   (ch.chunk_id, ch.doc_id, json.dumps(payload), 1, None))


# --- Phase 3-5: delegate to existing cbic_rag.ingest machinery --------------


def phase3_4_5(batch=BATCH, limit=None, doc_ids=None):
    """Embed + upsert canonical chunks using existing cbic_rag/embedder.py
    and ingest.py functions. Only canonical (is_canonical=1) chunks upsert."""
    # G4 fix: set QDRANT_COLL BEFORE importing ingest, because ingest.py reads
    # QCOLL at module-import time (line 38). Setting it after import is a no-op.
    os.environ["QDRANT_COLL"] = QDRANT_COLL_V2
    if embedder is None:
        raise RuntimeError("embedder.py not importable — run on rig")
    from ingest import ensure_collection, embed_batch, upsert_chunks, QCOLL  # reuse!
    from qdrant_client import QdrantClient
    assert QCOLL == QDRANT_COLL_V2, f"QCOLL override failed: got {QCOLL}, want {QDRANT_COLL_V2}"

    c2 = init_manifest_v2()
    qc = QdrantClient(url=QDRANT_URL, timeout=120)
    ensure_collection(qc, dim=embedder.DENSE_DIM)

    q = "SELECT chunk_id,payload_json FROM chunks WHERE is_canonical=1 AND upserted=0"
    if doc_ids:
        q += _doc_id_in_clause(doc_ids)
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = list(c2.execute(q))
    print(f"[phase3-5] {len(rows)} canonical chunks to embed+upsert")

    buf = []
    done = 0
    t0 = time.time()
    for chunk_id, payload_json in rows:
        p = json.loads(payload_json)
        buf.append(p)
        if len(buf) >= batch:
            _flush_batch(c2, qc, buf, embed_batch, upsert_chunks)
            done += len(buf)
            buf = []
            rate = done / max(1, time.time() - t0)
            if done % (batch * 10) == 0:
                print(f"  [phase3-5] done={done}/{len(rows)} rate={rate:.1f} ch/s")
    if buf:
        _flush_batch(c2, qc, buf, embed_batch, upsert_chunks)
        done += len(buf)
    # Sanity: confirm Qdrant actually has the points before claiming DONE.
    try:
        pts = qc.count(QCOLL, exact=True).count
    except Exception:
        pts = -1
    print(f"[phase3-5 DONE] {done} chunks submitted, qdrant points_count={pts} "
          f"in {time.time()-t0:.1f}s")
    if done > 0 and pts == 0:
        raise RuntimeError(f"phase3-5: submitted {done} chunks but Qdrant reports 0 — "
                           f"embed or upsert silently dropped everything")
    # Silent-success guard (added 2026-04-23): the stricter assertion. If we
    # submitted N but Qdrant has fewer than N-in-collection NEW points, something
    # dropped on the floor. `pts` is total count; for a fresh --doc-ids smoke
    # where we expect pts == done exactly, enforce. For incremental runs (resume,
    # append) pts >= done (since prior runs contributed). The rule below catches
    # both cases.
    if done > 0 and pts >= 0 and pts < done:
        _drift_budget = globals().get("ALLOW_UPSERT_DRIFT", 0)
        _drift = done - pts
        if _drift > _drift_budget:
            raise RuntimeError(
                f"phase3-5 SILENT-SUCCESS: submitted {done} but Qdrant has only {pts} "
                f"points in collection {QCOLL} (drift={_drift}, budget={_drift_budget}). "
                f"Some batches were dropped. Check embed output variance and Qdrant upsert response.")
        else:
            print(f"[phase3-5 SOFT-PASS] drift={_drift} within "
                  f"--allow-upsert-drift={_drift_budget} budget; proceeding")

    # 2026-04-24 PER-DOC RECONCILIATION — the original silent-success guard
    # operated at chunk-count granularity only. If 3 docs' chunks vanished but
    # other docs' chunks happened to sum to the same total, the old guard would
    # pass. Now: for every doc that contributed canonical chunks in this scoped
    # run, verify every canonical chunk of that doc ended with upserted=1.
    # Emit per-doc failure artifact + raise on any shortfall.
    scope_ids = doc_ids or [r[0] for r in c2.execute(
        "SELECT DISTINCT doc_id FROM chunks WHERE is_canonical=1").fetchall()]
    if scope_ids:
        q_in = ",".join("?" * len(scope_ids))
        recon = list(c2.execute(
            f"""SELECT doc_id,
                       SUM(CASE WHEN is_canonical=1 THEN 1 ELSE 0 END) AS expected,
                       SUM(CASE WHEN is_canonical=1 AND upserted=1 THEN 1 ELSE 0 END) AS upserted
                FROM chunks WHERE doc_id IN ({q_in}) GROUP BY doc_id""",
            scope_ids,
        ))
        phase3_failures = []
        for d_id, expected, upserted in recon:
            if expected is None:
                expected = 0
            if upserted is None:
                upserted = 0
            if expected > 0 and upserted < expected:
                reason = f"reconcile_fail: expected={expected} canonical chunks, upserted={upserted}"
                _record_phase_failure(c2, d_id, "phase3", reason)
                phase3_failures.append({"doc_id": d_id, "stage": "reconcile",
                                        "reason": reason,
                                        "expected": expected, "upserted": upserted})
        if phase3_failures:
            report_path = str(MANIFEST_V2) + ".phase3_failures.json"
            _write_failure_report(report_path, phase3_failures)
            print(f"[phase3-5 FAILURES] {len(phase3_failures)} docs with shortfalls "
                  f"— see {report_path}")
            for rec in phase3_failures[:20]:
                print(f"  - {rec['doc_id']}  expected={rec['expected']} upserted={rec['upserted']}")
            raise RuntimeError(
                f"phase3-5 FAILED reconciliation: {len(phase3_failures)} docs have "
                f"fewer upserted chunks than canonical. See {report_path}"
            )
        else:
            print(f"[phase3-5 RECONCILE] all {len(scope_ids)} scoped docs have "
                  f"expected == upserted canonical chunks.")


def _flush_batch(c2, qc, buf, embed_batch, upsert_chunks):
    """Embed + upsert a batch. RAISES on embed failure (do NOT swallow).
    On Qdrant 400 (malformed body, e.g. NaN/Inf vector slipping past sanitiser),
    halve-and-retry to isolate the bad point, log+skip it, continue.
    Codified 2026-04-26 after batch 1 crash on chunk 2880/4242 (NaN in vector)."""
    texts = [p.get("embed_text") or p.get("text") for p in buf]
    dense, sparse = embed_batch(texts)          # raises on embed failure
    ids = [p["chunk_id"] for p in buf]

    def _try_upsert(sub_buf, sub_dense, sub_sparse):
        """Recursive halve-and-retry. Returns (n_ok, skipped_chunk_ids)."""
        from qdrant_client.http.exceptions import UnexpectedResponse
        try:
            n = upsert_chunks(qc, sub_buf, sub_dense, sub_sparse)
            return n, []
        except UnexpectedResponse as e:
            status = getattr(e, 'status_code', None)
            if status != 400:
                raise
            if len(sub_buf) == 1:
                bad_id = sub_buf[0].get('chunk_id')
                bad_doc = sub_buf[0].get('doc_id')
                print(f'[_flush_batch BAD POINT skipped] chunk_id={bad_id} doc_id={bad_doc} '
                      f'qdrant_400={str(e)[:200]}')
                return 0, [bad_id]
            mid = len(sub_buf) // 2
            n_a, sk_a = _try_upsert(sub_buf[:mid], sub_dense[:mid], sub_sparse[:mid])
            n_b, sk_b = _try_upsert(sub_buf[mid:], sub_dense[mid:], sub_sparse[mid:])
            return n_a + n_b, sk_a + sk_b

    n, skipped = _try_upsert(buf, dense, sparse)
    if skipped:
        skipped_set = set(skipped)
        ok_ids = [i for i in ids if i not in skipped_set]
        print(f'[_flush_batch] upserted={n} skipped={len(skipped)}/{len(buf)} (Qdrant 400 isolates)')
    else:
        ok_ids = ids
    c2.executemany("UPDATE chunks SET embedded=1, upserted=1 WHERE chunk_id=?",
                   [(i,) for i in ok_ids])
    c2.commit()
    return n


# --- CLI --------------------------------------------------------------------


def main():
    # === NO-CPU PREFLIGHT (added 2026-04-24 per user hard rule) ===
    # ingest.py:embed_batch silently falls through to CPU fastembed BM25
    # unless SKIP_SPARSE=1 or DENSE_ONLY=1 is set. Enforced here, not in shell.
    import os as _os
    if _os.environ.get('DENSE_ONLY') != '1' and _os.environ.get('SKIP_SPARSE') != '1':
        raise SystemExit(
            '[preflight FAIL] no-CPU rule: export DENSE_ONLY=1 (or SKIP_SPARSE=1) '
            'before running ingest_v2. Without this, phase4 sparse embed runs on CPU via fastembed. '
            'Origin: 2026-04-24 user directive + audit caveat 1.'
        )
    # === END NO-CPU PREFLIGHT ===
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["phase1", "phase2", "phase3_4_5", "phase6_pairs", "all"], default="all")
    ap.add_argument("--scope", default=None, help="Cohort name for phase6_pairs (e.g. set6, full)")
    ap.add_argument("--collection", default=None, help="Qdrant collection for phase6_pairs (e.g. cbic_v2_set6)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    # D1 FLIPPED 2026-04-23: qwen3-14b is now PRIMARY after Claude CLI delivered
    # 40% invalid-JSON rate on the first production smoke (2/5 docs: trailing-comma
    # parse errors). qwen3-14b.service on GPU 2 port 9082 is proven (V2b: 30/30
    # parse) and already running. `--claude-first` is the legacy fallback for
    # when qwen3 is unavailable.
    ap.add_argument("--qwen-first", action="store_true", default=True,
                    help="Use qwen3-14b for Pass-1 (default; D1 primary as of 2026-04-23)")
    ap.add_argument("--claude-first", dest="qwen_first", action="store_false",
                    help="Legacy: use Claude CLI for Pass-1 (retired after 40%% JSON fail)")
    ap.add_argument("--doc-ids", type=str, default=None,
                    help="Comma-separated list of doc_ids to scope this run to (for targeted smoke)")
    ap.add_argument("--no-preflight", action="store_true", default=False,
                    help="Skip hard preflight (embedder facade / qwen3 / hello-world embed). "
                         "NEVER use on a real run. Component-smoke only.")
    # 2026-04-24: scale-test slack flags. Default posture stays strict (0).
    # Origin: GST50 run had 1/43 phase2 failures (bilingual annexure, qwen3
    # truncated JSON) and 2/1775 phase3-5 chunk drift. At 14,925 docs we expect
    # ~350 such edge-case docs, so the strict raise blocks every scale-up.
    ap.add_argument("--allow-phase2-failures", type=int, default=0,
                    help="Allow up to N phase2 doc failures before raising. "
                         "Default 0 (strict). Set to e.g. 5%% of scope for scale tests.")
    ap.add_argument("--allow-upsert-drift", type=int, default=0,
                    help="Allow Qdrant points_count to lag chunks_submitted by up "
                         "to N points before raising. Default 0 (strict).")
    args = ap.parse_args()
    # Stash on a global so phase2/phase3_4_5 can read without changing their
    # signatures (existing code calls them positionally from many places).
    global ALLOW_PHASE2_FAILURES, ALLOW_UPSERT_DRIFT
    ALLOW_PHASE2_FAILURES = args.allow_phase2_failures
    ALLOW_UPSERT_DRIFT = args.allow_upsert_drift

    doc_ids = [d.strip() for d in args.doc_ids.split(",") if d.strip()] if args.doc_ids else None
    if doc_ids:
        print(f"[main] --doc-ids active: {len(doc_ids)} ids  {doc_ids}")

    # HARD PREFLIGHT — runs unless explicitly disabled. See run_preflight() docstring.
    # Phase 1 alone (manifest copy) doesn't need embed/qwen3, but anything that
    # touches Phase 2 or 3-5 does. We gate preflight on those phases.
    needs_preflight = (args.phase in ("phase2", "phase3_4_5", "phase6_pairs", "all")) and not args.no_preflight
    if needs_preflight:
        run_preflight()
    elif args.no_preflight:
        print("[main] WARNING: --no-preflight set. You are running without the 2026-04-23 safety checks.")

    if args.phase in ("phase1", "all"):
        phase1(limit=args.limit, resume=args.resume, doc_ids=doc_ids)
    if args.phase in ("phase2", "all"):
        phase2(limit=args.limit, resume=args.resume, use_qwen_first=args.qwen_first, doc_ids=doc_ids)
    if args.phase in ("phase3_4_5", "all"):
        phase3_4_5(limit=args.limit, doc_ids=doc_ids)
    if args.phase in ("phase6_pairs",):
        if not args.scope or not args.collection:
            raise SystemExit("phase6_pairs requires --scope and --collection")
        import phase6_pairs as _p6
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        _p6.run(args.scope, args.collection, args.limit, gemini_key)


if __name__ == "__main__":
    main()
