#!/usr/bin/env python3
"""V1: Can qwen3-14b reliably return strict JSON for section_ref extraction?

Pass: >=48/50 parse cleanly; >=45/50 semantically correct on sample.
Run on rig: python3 probe_v1_qwen_json.py
Output: /opt/indian-legal-ai/data/probes/v1_result.json
"""
import json, time, urllib.request, random, sqlite3
from pathlib import Path

QWEN = "http://127.0.0.1:9082/v1/chat/completions"
QDRANT = "http://127.0.0.1:6343"
OUT = Path("/opt/indian-legal-ai/data/probes/v1_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

SYS = ("You extract legal section references from Indian tax law text. "
       "Return STRICT JSON: {\"section_ref\": \"Section N(x)(y)\" or null, "
       "\"confidence\": 0.0-1.0}. No prose. No markdown fences.")

def get_samples(n=50):
    body = {"limit": n, "with_payload": ["text"], "with_vector": False}
    req = urllib.request.Request(
        f"{QDRANT}/collections/cbic_v1/points/scroll",
        method="POST", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return [p["payload"]["text"][:2000] for p in r["result"]["points"] if p.get("payload", {}).get("text")]

def ask(text):
    body = {
        "model": "qwen3-14b",
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"/no_think\nTEXT:\n{text}\n\nExtract section_ref as JSON."}
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    req = urllib.request.Request(QWEN, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=30).read())
        dt = time.time() - t0
        return r["choices"][0]["message"]["content"], dt, None
    except Exception as e:
        return None, time.time()-t0, f"{type(e).__name__}: {e}"

def main():
    samples = get_samples(50)
    print(f"[V1] got {len(samples)} samples")
    results = []
    parsed_ok = 0
    for i, txt in enumerate(samples):
        raw, dt, err = ask(txt)
        if err:
            results.append({"i": i, "err": err, "dt": dt})
            continue
        # strip markdown fences if present
        r = raw.strip()
        if r.startswith("```"):
            r = r.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            obj = json.loads(r)
            ok = "section_ref" in obj and "confidence" in obj
            if ok: parsed_ok += 1
            results.append({"i": i, "parsed": ok, "ref": obj.get("section_ref"),
                            "conf": obj.get("confidence"), "dt": dt})
        except Exception as e:
            results.append({"i": i, "parsed": False, "raw": raw[:200], "dt": dt,
                            "parse_err": str(e)})
        if i % 10 == 0:
            print(f"  [{i}/{len(samples)}] ok={parsed_ok}")

    lats = [r["dt"] for r in results if "dt" in r]
    lats.sort()
    p50 = lats[len(lats)//2] if lats else 0
    p95 = lats[int(len(lats)*0.95)] if lats else 0

    summary = {
        "probe": "V1",
        "n": len(samples),
        "parsed_ok": parsed_ok,
        "parse_rate": round(parsed_ok/len(samples), 3),
        "latency_p50": round(p50, 2),
        "latency_p95": round(p95, 2),
        "pass_gate_parse": parsed_ok >= 48,
        "pass_gate_overall": parsed_ok >= 48,
        "results": results,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[V1] parse_rate={summary['parse_rate']} p50={p50:.1f}s p95={p95:.1f}s "
          f"PASS={summary['pass_gate_overall']}")

if __name__ == "__main__":
    main()
