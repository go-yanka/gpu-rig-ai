"""Test qwen3-14b in-process via llama-cpp-python (direct, NO llama-server).
Same model on same GPU 2. Compare timing vs HTTP llama-server.
"""
import os, sys, time, json
os.environ["GGML_VK_VISIBLE_DEVICES"] = "2"
os.environ["RADV_DEBUG"] = "nodcc"
os.environ["GGML_VK_DISABLE_INTEGER_DOT_PRODUCT"] = "1"
sys.path.insert(0, "/home/user/.local/lib/python3.10/site-packages")

MODEL = "/opt/ai-models/qwen3-14b-q4_k_m.gguf"

def now(): return time.strftime("%H:%M:%S")

print(f"[{now()}] === DIRECT qwen3-14b BENCH (GPU 2, no llama-server) ===")
print(f"[{now()}] importing llama_cpp...")
from llama_cpp import Llama

print(f"[{now()}] cold-loading model {MODEL}...")
t0 = time.time()
llm = Llama(model_path=MODEL, n_gpu_layers=-1, n_ctx=2048, n_batch=2048, verbose=False)
cold = time.time() - t0
print(f"[{now()}] cold-load = {cold:.1f}s")

print(f"[{now()}] warmup call...")
t0 = time.time()
r = llm("hi", max_tokens=5, temperature=0, echo=False)
warm = time.time() - t0
print(f"[{now()}] warmup = {warm:.2f}s, output: {r['choices'][0]['text'][:60]!r}")

print(f"[{now()}] 5 timed calls (10 max_tokens each):")
results = []
for i in range(5):
    t0 = time.time()
    r = llm("Reply with one word: act, rules, notification, circular. Doc: 'Notification 12/2017 GST'", max_tokens=10, temperature=0, echo=False)
    dt = time.time() - t0
    results.append(dt)
    print(f"[{now()}]   call {i+1}: {dt:.2f}s, tokens={r.get('usage',{}).get('completion_tokens', '?')}, text={r['choices'][0]['text'][:40]!r}")

avg = sum(results) / len(results)
print(f"[{now()}] === RESULT ===")
print(f"  cold-load: {cold:.1f}s")
print(f"  warm avg per call (10 tokens): {avg*1000:.0f}ms")
print(f"  tokens/sec at warm: {10/avg:.1f}")
print(f"")
print(f"For comparison, HTTP llama-server is timing out at 30s for similar calls.")
print(f"If this is FAST → llama-server / HTTP path is the bottleneck. RG was right.")
print(f"If this is also slow → hardware issue, not architecture.")

json.dump({"cold_load_s": cold, "warm_avg_ms": avg*1000, "tok_per_s": 10/avg, "raw": results},
          open("/opt/indian-legal-ai/reingest_spec/qwen3_direct_bench.json", "w"), indent=2)
