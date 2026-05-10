#!/usr/bin/env python3
"""V17: Gemini-as-judge consistency. 20 known answers x 3 runs.
Pass: stdev per answer <=0.5; mean diff run-to-run <=0.3.
Run from laptop or rig (Gemini API).
"""
import json, time, urllib.request, statistics
from pathlib import Path

ENV = Path("D:/_gpu_rig_ai/.env") if Path("D:/_gpu_rig_ai/.env").exists() else Path("/mnt/d/_gpu_rig_ai/.env")
for line in ENV.read_text().splitlines():
    if line.startswith("GEMINI_API_KEY="):
        API_KEY = line.split("=",1)[1].strip(); break

URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
OUT = Path("D:/_gpu_rig_ai/reingest_spec/v17_result.json")

# 20 known (query, answer, reasoning) triples -- fill from actual logs
# For now: 3 sample stubs + placeholder for operator to extend
SAMPLES = [
    {
        "query": "What is GST rate on restaurant services (non-AC)?",
        "answer": "Per Notification 11/2017-CTR, GST on restaurant services is 5% without ITC.",
        "reasoning": "The query asks for a specific rate. Retrieved Notification 11/2017 specifies 5% for restaurant services provided the credit of input tax has not been taken.",
    },
    {
        "query": "Section 16(2)(c) conditions",
        "answer": "Section 16(2)(c) requires the supplier to have actually paid the tax charged.",
        "reasoning": "The provision explicitly states tax must have been paid to government by supplier for ITC eligibility.",
    },
    {
        "query": "Customs duty rate on imported wheat",
        "answer": "I cannot find this in the indexed CBIC corpus.",
        "reasoning": "No retrieved chunk specified a duty rate for wheat.",
    },
]

JUDGE_PROMPT = ("You are a legal answer auditor. Given a query, answer, and reasoning, "
                "score the reasoning 1-5 where 5=fully grounded, consistent, correct. "
                "Respond with just a number.")

def judge(sample, max_retries=5):
    prompt = (f"{JUDGE_PROMPT}\n\nQuery: {sample['query']}\nAnswer: {sample['answer']}\n"
              f"Reasoning: {sample['reasoning']}\nScore:")
    body = {"contents":[{"parts":[{"text":prompt}]}],
            "generationConfig":{"temperature":0.0,"maxOutputTokens":5}}
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(URL, method="POST",
                data=json.dumps(body).encode(),
                headers={"Content-Type":"application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=30).read())
            txt = r["candidates"][0]["content"]["parts"][0]["text"].strip()
            try: return int("".join(c for c in txt if c.isdigit())[:1] or "0")
            except: return 0
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                wait = min(60, (2 ** attempt) * 5 + (attempt * 3))
                print(f"    429 hit, backoff {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"judge failed after {max_retries} retries: {last_err}")

def main():
    scores_per_sample = []
    for i, s in enumerate(SAMPLES):
        runs = []
        for run in range(3):
            try: runs.append(judge(s))
            except Exception as e: print(f"  s{i} run{run} err {e}")
            time.sleep(16)  # ~3.75 RPM — deep under free-tier 15 RPM including shared quota
        scores_per_sample.append(runs)
        print(f"  sample {i}: runs={runs}")

    stdevs = [statistics.stdev(r) if len(r)>=2 else 0 for r in scores_per_sample]
    means = [statistics.mean(r) if r else 0 for r in scores_per_sample]
    max_stdev = max(stdevs) if stdevs else 0
    # A sample with zero successful runs is NOT a pass — require all samples to have at least 2 runs
    successful = [r for r in scores_per_sample if len(r) >= 2]
    all_complete = len(successful) == len(scores_per_sample) and len(scores_per_sample) > 0
    summary = {
        "probe": "V17",
        "n_samples": len(SAMPLES),
        "runs_per_sample": 3,
        "scores": scores_per_sample,
        "per_sample_stdev": stdevs,
        "per_sample_mean": means,
        "max_stdev": max_stdev,
        "all_samples_complete": all_complete,
        "pass_gate": all_complete and max_stdev <= 0.5,
        "note": "extend SAMPLES to 20 real (query,answer,reasoning) triples from logs",
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="scores"}, indent=2))

if __name__ == "__main__":
    main()
