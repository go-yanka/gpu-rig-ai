#!/usr/bin/env python3
"""V2b: qwen3-14b Pass-1 classification latency (replaces V2).

Rationale: Pass-1 chunker classification uses first 2000 + last 1500 chars
as input and emits short JSON (<150 tokens). V2 tested a different regime
(3000+ char bodies, max_tokens=300) that does not match how we call qwen3
in the actual plan. This probe matches the Pass-1 call shape exactly.

Pass: p50 <=3s, p95 <=5s on ~3500-char synthetic Pass-1 inputs.
Run on rig: python3 probe_v2b_qwen_pass1.py
"""
import json, time, urllib.request
from pathlib import Path

QWEN = "http://127.0.0.1:9082/v1/chat/completions"
QDRANT = "http://127.0.0.1:6343"
OUT = Path("/opt/indian-legal-ai/data/probes/v2b_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "/no_think\n"
    "You are classifying an Indian CBIC legal document to produce a chunking "
    "plan. Do NOT chunk — only describe structure. Respond with STRICT JSON: "
    '{"doc_type":"act|rules|notification|circular|form|faq|judgment|schedule|press_release|mixed",'
    '"structure":"hierarchical_sections|flat_paragraphs|tabular|form_fields|list_of_items|mixed",'
    '"primary_splitter":"section|rule|chapter|heading|paragraph|table_row|page",'
    '"has_tables":true|false,"language":"en|hi|bilingual","confidence":0.0-1.0}\n\n'
    "DOC HEAD + TAIL:\n"
)

def get_pass1_inputs(n=30):
    """Build synthetic Pass-1 inputs: first 2000 + last 1500 chars of long docs."""
    body = {"limit": 300, "with_payload": ["text"], "with_vector": False}
    req = urllib.request.Request(
        f"{QDRANT}/collections/cbic_v1/points/scroll",
        method="POST", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    # Take longest chunks, simulate doc head+tail
    texts = sorted(
        [p["payload"]["text"] for p in r["result"]["points"]
         if p.get("payload", {}).get("text")],
        key=len, reverse=True)
    inputs = []
    for t in texts[:n*2]:
        if len(t) < 2500:
            continue
        head = t[:2000]
        tail = t[-1500:] if len(t) > 3500 else t[2000:]
        inputs.append(head + "\n...\n" + tail)
        if len(inputs) >= n:
            break
    return inputs

def ask(text):
    body = {
        "model": "qwen3-14b",
        "messages": [{"role": "user", "content": PROMPT + text}],
        "temperature": 0.0,
        "max_tokens": 150,
    }
    req = urllib.request.Request(QWEN, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    dt = time.time() - t0
    content = r["choices"][0]["message"]["content"]
    tokens_in = r.get("usage", {}).get("prompt_tokens", 0)
    tokens_out = r.get("usage", {}).get("completion_tokens", 0)
    parsed_ok = False
    try:
        s = content.find("{"); e = content.rfind("}")
        if s >= 0 and e > s:
            json.loads(content[s:e+1])
            parsed_ok = True
    except Exception:
        pass
    return dt, tokens_in, tokens_out, parsed_ok

def main():
    samples = get_pass1_inputs(30)
    print(f"[V2b] got {len(samples)} Pass-1 shaped inputs (~3500 chars each)")
    lats, tok_ins, tok_outs, parse_ok = [], [], [], 0
    for i, s in enumerate(samples):
        try:
            dt, ti, to, ok = ask(s)
            lats.append(dt); tok_ins.append(ti); tok_outs.append(to)
            parse_ok += int(ok)
            if i % 5 == 0:
                print(f"  [{i}/{len(samples)}] dt={dt:.2f}s in={ti} out={to} ok={ok}")
        except Exception as e:
            print(f"  [{i}] ERR {e}")
    lats.sort()
    p50 = lats[len(lats)//2]; p95 = lats[int(len(lats)*0.95)]
    summary = {
        "probe": "V2b", "n": len(lats),
        "latency_p50": round(p50, 2), "latency_p95": round(p95, 2),
        "latency_max": round(lats[-1], 2) if lats else 0,
        "avg_tokens_in": sum(tok_ins)//max(len(tok_ins),1),
        "avg_tokens_out": sum(tok_outs)//max(len(tok_outs),1),
        "parse_rate": round(parse_ok/max(len(lats),1), 2),
        "pass_gate": p50 <= 3.0 and p95 <= 5.0 and parse_ok/max(len(lats),1) >= 0.9,
        "projection_851_docs_minutes": round(851 * (sum(lats)/len(lats)) / 60, 1) if lats else 0,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[V2b] p50={p50:.2f} p95={p95:.2f} parse={parse_ok}/{len(lats)} "
          f"PASS={summary['pass_gate']} proj_851docs={summary['projection_851_docs_minutes']}min")

if __name__ == "__main__":
    main()
