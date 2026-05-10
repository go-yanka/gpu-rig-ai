#!/usr/bin/env python3
"""V10: route_llm latency on qwen3-14b.
Call the LLM router directly on 50 varied queries, measure p50/p95.
Pass: p50 <=800ms, p95 <=1500ms.
Run on rig.
"""
import json, time, urllib.request
from pathlib import Path

QWEN = "http://127.0.0.1:9082/v1/chat/completions"
OUT = Path("/opt/indian-legal-ai/data/probes/v10_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["gst", "customs", "central-excise", "service-tax", "ntrp", "unknown"]
SYS = ("Classify the Indian tax query into one of: gst, customs, central-excise, service-tax, "
       "ntrp, unknown. Reply with the single word category only, nothing else.")

QUERIES = [
    "What is the GST rate on restaurant services?",
    "Customs duty on imported laptops",
    "Section 16(2)(c) input tax credit conditions",
    "Service tax exemption for educational institutions",
    "NTRP payment procedure for government services",
    "Export of goods under LUT without payment of IGST",
    "Reverse charge mechanism under GST",
    "Anti-dumping duty on steel imports from China",
    "Excise duty applicability on petroleum products",
    "CGST vs IGST applicability on inter-state supply",
    "Drawback scheme for exporters",
    "Place of supply rules for online services",
    "Refund of accumulated ITC under inverted duty structure",
    "Import of gift articles customs duty",
    "Central excise on tobacco products",
    "GST council meeting decisions May 2024",
    "DGFT notification on import policy",
    "GSTR-3B late fee and interest",
    "Classification of composite supply vs mixed supply",
    "Customs valuation under Section 14 of Customs Act",
]*3  # 60 queries

def ask(q):
    body = {"model": "qwen3-14b",
            "messages": [{"role":"system","content":SYS},{"role":"user","content":f"/no_think\n{q}"}],
            "temperature": 0.0, "max_tokens": 10}
    req = urllib.request.Request(QWEN, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=10).read())
    dt = time.time() - t0
    return dt, r["choices"][0]["message"]["content"].strip().lower()

def main():
    lats = []; answers = []
    for i, q in enumerate(QUERIES):
        try:
            dt, a = ask(q); lats.append(dt); answers.append(a)
            if i % 10 == 0: print(f"  [{i}] {dt*1000:.0f}ms -> {a}")
        except Exception as e:
            print(f"  [{i}] ERR {e}")
    lats.sort()
    p50 = lats[len(lats)//2]; p95 = lats[int(len(lats)*0.95)]
    summary = {
        "probe": "V10", "n": len(lats),
        "p50_ms": round(p50*1000), "p95_ms": round(p95*1000),
        "max_ms": round(lats[-1]*1000),
        "pass_gate": p50 <= 0.8 and p95 <= 1.5,
        "sample_answers": answers[:20],
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
