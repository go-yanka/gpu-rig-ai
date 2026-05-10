#!/usr/bin/env python3
"""V9: Gemini daily quota vs worst-case re-OCR + extraction usage.
Static calculation + a single probe request to confirm key still works.
Runs from laptop.
"""
import json, urllib.request
from pathlib import Path

ENV = Path("D:/_gpu_rig_ai/.env")
for line in ENV.read_text().splitlines():
    if line.startswith("GEMINI_API_KEY="):
        API_KEY = line.split("=",1)[1].strip(); break

OUT = Path("D:/_gpu_rig_ai/reingest_spec/v9_result.json")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Worst case estimates
TABLES_REOCR_PAGES = 500         # estimate: 500 table-heavy pages need re-OCR
REGEX_MISS_CHUNKS = 20000         # estimate: 20k chunks that regex can't parse, fall back to LLM
PER_CALL_TOKENS_IN = 2000
PER_CALL_TOKENS_OUT = 300
DAILY_FREE_TIER_REQUESTS = 1500   # Flash free tier; adjust if paid

def main():
    # Probe: single call to confirm key works
    body = {"contents":[{"parts":[{"text":"ping"}]}],
            "generationConfig":{"maxOutputTokens":5}}
    req = urllib.request.Request(URL, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=30).read())
        key_ok = "candidates" in r
    except Exception as e:
        key_ok = False; err = str(e)

    total_calls = TABLES_REOCR_PAGES + REGEX_MISS_CHUNKS
    PRICE_IN = 0.10/1_000_000; PRICE_OUT = 0.40/1_000_000
    est_cost = total_calls * (PER_CALL_TOKENS_IN * PRICE_IN + PER_CALL_TOKENS_OUT * PRICE_OUT)

    summary = {
        "probe": "V9",
        "key_reachable": key_ok,
        "estimated_total_calls": total_calls,
        "breakdown": {
            "table_reocr": TABLES_REOCR_PAGES,
            "regex_miss_backstop": REGEX_MISS_CHUNKS,
        },
        "estimated_cost_usd": round(est_cost, 2),
        "days_required_at_free_tier": round(total_calls / DAILY_FREE_TIER_REQUESTS, 1),
        "pass_gate": key_ok and est_cost < 50,
        "note": ("Free tier is rate-limited, not cost-limited; paid tier removes rate limit. "
                 "If using paid: ~$" + str(round(est_cost,2)) + " total budget. "
                 "If free: ~" + str(round(total_calls/DAILY_FREE_TIER_REQUESTS,1)) + " days."),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
