#!/usr/bin/env python3
"""V2: qwen3-14b extraction latency on 4096-token-ish inputs.

Pass: p50 <=2s, p95 <=4s. Drives budget for chunker LLM backstop.
Run on rig: python3 probe_v2_qwen_latency.py
"""
import json, time, urllib.request
from pathlib import Path

QWEN = "http://127.0.0.1:9082/v1/chat/completions"
QDRANT = "http://127.0.0.1:6343"
OUT = Path("/opt/indian-legal-ai/data/probes/v2_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

PROMPT = ("Given the legal text below, extract: section_ref, proviso_ref (if any), "
          "rule_ref (if any). Return strict JSON only, no prose.\n\nTEXT:\n")

def get_long_samples(n=30, min_chars=3000):
    body = {"limit": 200, "with_payload": ["text"], "with_vector": False}
    req = urllib.request.Request(
        f"{QDRANT}/collections/cbic_v1/points/scroll",
        method="POST", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    longs = [p["payload"]["text"] for p in r["result"]["points"]
             if p.get("payload", {}).get("text") and len(p["payload"]["text"]) >= min_chars]
    return longs[:n]

def ask(text):
    body = {
        "model": "qwen3-14b",
        "messages": [{"role": "user", "content": "/no_think\n" + PROMPT + text}],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    req = urllib.request.Request(QWEN, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    dt = time.time() - t0
    tokens_in = r.get("usage", {}).get("prompt_tokens", 0)
    tokens_out = r.get("usage", {}).get("completion_tokens", 0)
    return dt, tokens_in, tokens_out

def main():
    samples = get_long_samples(30, 3000)
    print(f"[V2] got {len(samples)} long samples (>=3000 chars)")
    lats, tok_ins, tok_outs = [], [], []
    for i, s in enumerate(samples):
        try:
            dt, ti, to = ask(s)
            lats.append(dt); tok_ins.append(ti); tok_outs.append(to)
            if i % 5 == 0:
                print(f"  [{i}/{len(samples)}] dt={dt:.2f}s in={ti} out={to}")
        except Exception as e:
            print(f"  [{i}] ERR {e}")
    lats.sort()
    p50 = lats[len(lats)//2]; p95 = lats[int(len(lats)*0.95)]
    summary = {
        "probe": "V2", "n": len(lats),
        "latency_p50": round(p50, 2), "latency_p95": round(p95, 2),
        "latency_max": round(lats[-1], 2) if lats else 0,
        "avg_tokens_in": sum(tok_ins)//max(len(tok_ins),1),
        "avg_tokens_out": sum(tok_outs)//max(len(tok_outs),1),
        "pass_gate": p50 <= 2.0 and p95 <= 4.0,
        "projection_115k_chunks_hours": round(115000 * (sum(lats)/len(lats)) / 3600, 1) if lats else 0,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[V2] p50={p50:.2f} p95={p95:.2f} PASS={summary['pass_gate']} "
          f"proj_115k={summary['projection_115k_chunks_hours']}h")

if __name__ == "__main__":
    main()
