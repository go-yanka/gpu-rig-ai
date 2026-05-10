"""Quick bench: HTTP llama-server vs in-process llama-cpp-python.
Prints progress at every step. Smaller N (5) and short max_tokens (12).
Run: PYTHONPATH=/home/user/.local/lib/python3.10/site-packages /usr/bin/python3 bench_quick.py
"""
import json, time, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

QWEN3_MODEL = "/opt/ai-models/qwen3-14b-q4_k_m.gguf"
RERANKER_MODEL = "/opt/indian-legal-ai/models/bge-reranker-v2-m3-Q4_K_M.gguf"
CLASSIFY_PROMPT = "Reply with one word: act, rules, notification, circular, instruction, form, other.\n\nDoc: 'Notification 12/2017 GST exemption'\nCategory:"
RERANK_QUERY = "GST refund procedure for SEZ unit"
RERANK_DOCS = ["GST refund procedure involves filing", "Election manifesto policy", "SEZ units claim refund of input tax", "Weather forecast tomorrow", "Customs duty payment"] * 4
N = 5

def now():
    return time.strftime("%H:%M:%S")

def http_call(url, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def vram(g):
    f = f"/sys/class/drm/card{g}/device/mem_info_vram_used"
    try: return int(open(f).read())/1024/1024
    except: return None

results = {}
print(f"[{now()}] === BENCH START ===")

# ----- qwen3 HTTP -----
print(f"[{now()}] qwen3 HTTP: warmup...")
body = {"prompt": CLASSIFY_PROMPT, "max_tokens": 12, "temperature": 0}
http_call("http://127.0.0.1:9082/v1/completions", body)
print(f"[{now()}] qwen3 HTTP: {N} sequential calls...")
t0 = time.time()
for i in range(N):
    http_call("http://127.0.0.1:9082/v1/completions", body)
    print(f"[{now()}]   call {i+1}/{N} done")
seq_t = (time.time()-t0)/N*1000
results["qwen3_http_seq_ms"] = seq_t
print(f"[{now()}] qwen3 HTTP: seq avg = {seq_t:.0f}ms/call")

# ----- qwen3 in-process -----
print(f"[{now()}] qwen3 in-process: importing llama_cpp...")
os.environ["GGML_VK_VISIBLE_DEVICES"] = "2"
os.environ["RADV_DEBUG"] = "nodcc"
sys.path.insert(0, "/home/user/.local/lib/python3.10/site-packages")
print(f"[{now()}] qwen3 in-process: cold-loading model (this is the bottleneck)...")
t0 = time.time()
from llama_cpp import Llama
llm = Llama(model_path=QWEN3_MODEL, n_gpu_layers=-1, n_ctx=2048, n_batch=2048, verbose=False)
cold = time.time()-t0
results["qwen3_inproc_cold_load_s"] = cold
print(f"[{now()}] qwen3 in-process: cold-load = {cold:.1f}s, VRAM GPU 2 = {vram(2):.0f}MB")
print(f"[{now()}] qwen3 in-process: warmup...")
llm(CLASSIFY_PROMPT, max_tokens=12, temperature=0, echo=False)
print(f"[{now()}] qwen3 in-process: {N} sequential calls...")
t0 = time.time()
for i in range(N):
    llm(CLASSIFY_PROMPT, max_tokens=12, temperature=0, echo=False)
    print(f"[{now()}]   call {i+1}/{N} done")
seq_t = (time.time()-t0)/N*1000
results["qwen3_inproc_seq_ms"] = seq_t
print(f"[{now()}] qwen3 in-process: seq avg = {seq_t:.0f}ms/call")
del llm

# ----- reranker HTTP --parallel 8 (already running) -----
print(f"[{now()}] reranker HTTP: {N} sequential calls (20 docs each)...")
body = {"model":"bge","query":RERANK_QUERY,"documents":RERANK_DOCS}
http_call("http://127.0.0.1:9085/v1/rerank", body)
t0 = time.time()
for _ in range(N): http_call("http://127.0.0.1:9085/v1/rerank", body)
seq_t = (time.time()-t0)/N*1000
results["rerank_http_seq_ms"] = seq_t
print(f"[{now()}] reranker HTTP: seq = {seq_t:.0f}ms/call")
print(f"[{now()}] reranker HTTP: {N*8} concurrent calls (--parallel 8)...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(lambda _: http_call("http://127.0.0.1:9085/v1/rerank", body), range(N*8)))
par_t = time.time()-t0
results["rerank_http_par8_total_s"] = par_t
results["rerank_http_par8_per_call_ms"] = par_t/(N*8)*1000
print(f"[{now()}] reranker HTTP: {N*8} parallel-8 total = {par_t:.2f}s ({par_t/(N*8)*1000:.0f}ms/call)")
print(f"[{now()}] reranker HTTP: parallelism speedup = {seq_t/(par_t/(N*8)*1000):.2f}x")

print(f"[{now()}] === BENCH DONE ===")
print(json.dumps(results, indent=2))
json.dump(results, open("/opt/indian-legal-ai/reingest_spec/bench_quick_result.json","w"), indent=2)
