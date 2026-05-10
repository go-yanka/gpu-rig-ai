"""Solo cold-load test for lemon GPUs 1 and 6.

Loads ONE GPU at a time (not concurrent) with a long deadline. If both come up
solo, the issue is concurrent-load contention. If they fail solo too, marginal cards.

Run as:
  PYTHONPATH=/opt/indian-legal-ai/rag/cbic_rag:/home/user/.local/lib/python3.10/site-packages \
  RADV_DEBUG=nodcc GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 \
  python3 /tmp/solo_lemon_test.py 1
"""
import os, sys, time

GPU_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
MODEL = "/usr/share/ollama/.ollama/models/blobs/sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c"

os.environ["GGML_VK_VISIBLE_DEVICES"] = str(GPU_ID)
print(f"[solo] GPU {GPU_ID} — importing llama_cpp...", flush=True)
t0 = time.time()
from llama_cpp import Llama
print(f"[solo] GPU {GPU_ID} — import done in {time.time()-t0:.1f}s, loading model...", flush=True)
t1 = time.time()
m = Llama(model_path=MODEL, embedding=True, n_gpu_layers=-1,
          n_ctx=8192, n_batch=512, n_ubatch=512, n_threads=2,
          verbose=False, use_mmap=True, use_mlock=False)
print(f"[solo] GPU {GPU_ID} — LOADED in {time.time()-t1:.1f}s", flush=True)
t2 = time.time()
v = m.create_embedding("Section 17(5) of CGST Act blocks ITC on motor vehicles for personal use.")
print(f"[solo] GPU {GPU_ID} — EMBED in {(time.time()-t2)*1000:.0f}ms, dim={len(v['data'][0]['embedding'])}", flush=True)
print(f"[solo] GPU {GPU_ID} — TOTAL {time.time()-t0:.1f}s", flush=True)
