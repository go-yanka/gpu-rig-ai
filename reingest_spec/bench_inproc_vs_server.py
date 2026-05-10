"""Empirical bench: llama-server (HTTP) vs llama-cpp-python (in-process)
for qwen3-14b classifier + bge-reranker cross-encoder.

Tests:
  1. Cold-load time
  2. Per-call latency (sequential)
  3. Concurrent throughput (parallel calls)
  4. VRAM usage

Outputs: /tmp/bench_inproc_vs_server.json
Run: python3 bench_inproc_vs_server.py
"""
import json, time, os, subprocess, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

QWEN3_MODEL = "/opt/ai-models/qwen3-14b-q4_k_m.gguf"
RERANKER_MODEL = "/opt/indian-legal-ai/models/bge-reranker-v2-m3-Q4_K_M.gguf"
CLASSIFY_PROMPT = "You are a CBIC document classifier. Reply only with the category from {act, rules, notification, circular, instruction, form, other}.\n\nDoc title: 'Notification 12/2017 — exemption to certain services under GST'\nCategory:"
RERANK_QUERY = "GST refund procedure for SEZ unit"
RERANK_DOCS = ["Procedure for GST refund involves filing", "The election manifesto outlines policy", "SEZ units may claim refund of input tax", "Weather forecast for tomorrow shows rain", "Customs duty payment process"] * 4  # 20 docs

def http_call(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def vram_mb(gpu_id):
    try:
        f = f"/sys/class/drm/card{gpu_id}/device/mem_info_vram_used"
        if os.path.exists(f): return int(open(f).read()) / 1024 / 1024
    except: pass
    return None

# ----- Test 1: qwen3-14b classifier -----
def bench_qwen3_server(n=10):
    """HTTP llama-server :9082 (current architecture)."""
    print(f"\n=== qwen3-14b via llama-server (HTTP :9082) ===")
    print(f"VRAM GPU 2 before: {vram_mb(2):.0f} MB")
    body = {"prompt": CLASSIFY_PROMPT, "max_tokens": 12, "temperature": 0}
    # Warmup
    http_call("http://127.0.0.1:9082/v1/completions", body)
    # Sequential
    t0 = time.time()
    for _ in range(n):
        http_call("http://127.0.0.1:9082/v1/completions", body)
    seq_total = time.time() - t0
    seq_per_call = seq_total / n * 1000
    # Concurrent (single-slot --parallel 1, so this serializes)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda _: http_call("http://127.0.0.1:9082/v1/completions", body), range(n)))
    par_total = time.time() - t0
    return {"variant":"http_server","cold_load_s":"persistent","seq_ms_per_call":seq_per_call,
            "par4_total_s":par_total, "par4_per_call_ms":par_total/n*1000}

def bench_qwen3_inproc(n=10):
    """In-process llama-cpp-python on GPU 2."""
    print(f"\n=== qwen3-14b via llama-cpp-python (in-process, GPU 2) ===")
    os.environ["GGML_VK_VISIBLE_DEVICES"] = "2"
    os.environ["RADV_DEBUG"] = "nodcc"
    sys.path.insert(0, "/home/user/.local/lib/python3.10/site-packages")
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(model_path=QWEN3_MODEL, n_gpu_layers=-1, n_ctx=2048, n_batch=2048, verbose=False)
    cold = time.time() - t0
    print(f"cold-load {cold:.1f}s, VRAM GPU 2 after: {vram_mb(2):.0f} MB")
    # Warmup
    llm(CLASSIFY_PROMPT, max_tokens=12, temperature=0, echo=False)
    # Sequential
    t0 = time.time()
    for _ in range(n):
        llm(CLASSIFY_PROMPT, max_tokens=12, temperature=0, echo=False)
    seq_total = time.time() - t0
    seq_per_call = seq_total / n * 1000
    # Concurrent — GIL means this serializes
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda _: llm(CLASSIFY_PROMPT, max_tokens=12, temperature=0, echo=False), range(n)))
    par_total = time.time() - t0
    del llm
    return {"variant":"inproc","cold_load_s":cold,"seq_ms_per_call":seq_per_call,
            "par4_total_s":par_total, "par4_per_call_ms":par_total/n*1000}

# ----- Test 2: bge-reranker -----
def bench_reranker_server(n=10):
    """HTTP llama-server :9085 with --parallel 8."""
    print(f"\n=== bge-reranker via llama-server (HTTP :9085, --parallel 8) ===")
    body = {"model":"bge","query":RERANK_QUERY,"documents":RERANK_DOCS}
    http_call("http://127.0.0.1:9085/v1/rerank", body)
    t0 = time.time()
    for _ in range(n):
        http_call("http://127.0.0.1:9085/v1/rerank", body)
    seq_total = time.time() - t0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: http_call("http://127.0.0.1:9085/v1/rerank", body), range(n)))
    par_total = time.time() - t0
    return {"variant":"http_server_parallel8","seq_ms_per_call":seq_total/n*1000,
            "par8_total_s":par_total, "par8_per_call_ms":par_total/n*1000,
            "speedup_vs_seq": (seq_total/par_total) if par_total>0 else 0}

def bench_reranker_inproc(n=10):
    """In-process: hard. llama-cpp-python doesn't natively support cross-encoder rerank API."""
    print(f"\n=== bge-reranker in-process — N/A ===")
    return {"variant":"inproc","skip":"llama-cpp-python lacks --rerank --pooling rank API; HTTP llama-server is the canonical interface for the GGUF cross-encoder"}

if __name__ == "__main__":
    results = {"ts": time.time(), "tests": []}
    try: results["tests"].append({"name":"qwen3_server", **bench_qwen3_server(10)})
    except Exception as e: results["tests"].append({"name":"qwen3_server","error":str(e)})
    try: results["tests"].append({"name":"qwen3_inproc", **bench_qwen3_inproc(10)})
    except Exception as e: results["tests"].append({"name":"qwen3_inproc","error":str(e)})
    try: results["tests"].append({"name":"reranker_server", **bench_reranker_server(10)})
    except Exception as e: results["tests"].append({"name":"reranker_server","error":str(e)})
    try: results["tests"].append({"name":"reranker_inproc", **bench_reranker_inproc(10)})
    except Exception as e: results["tests"].append({"name":"reranker_inproc","error":str(e)})
    json.dump(results, open("/tmp/bench_inproc_vs_server.json","w"), indent=2)
    print("\n\n=== SUMMARY ===")
    for t in results["tests"]:
        print(f"\n{t['name']}:")
        for k,v in t.items():
            if k != "name": print(f"  {k}: {v}")
    print("\nFull: /tmp/bench_inproc_vs_server.json")
