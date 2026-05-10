# CBIC RAG Evaluation Harness

A gold-standard Q/A set plus a scoring runner for the CBIC RAG system at
`http://192.168.1.107:9500`. Run this after every backend deploy to catch
silent regressions.

## Files

- `gold_set.yaml` — 50 curated Q/A pairs across 5 categories (20 GST, 10 Customs,
  8 Central Excise, 6 Service Tax, 6 Others). 5 items marked `complex` mirror
  real user pain points (multi-part scenarios like Quantum Tech).
- `run_eval.py` — runs the gold set against a RAG endpoint, scores each answer,
  writes JSON + markdown to `runs/<timestamp>/`.
- `diff_runs.py` — diffs two runs to surface regressions/improvements.
- `runs/` — per-run output directory (git-ignore this if needed).

## Install

```
pip install pyyaml requests
```

Python 3.10+.

## How to run

```
python run_eval.py --gold gold_set.yaml \
    --api http://192.168.1.107:9500 \
    --out runs/2026_04_21_b22b
```

Faster smoke test (first 5 items, no judge):

```
python run_eval.py --out runs/smoke --limit 5 --no-judge
```

Just one category:

```
python run_eval.py --out runs/gst_only --only gst
```

Parallel (up to 4 concurrent — API handles it post-B16):

```
python run_eval.py --out runs/fast --workers 4
```

## How to diff runs

```
python diff_runs.py --base runs/2026_04_20_base --new runs/2026_04_21_b22b
```

Prints a per-item delta table + aggregate delta + regression gate flag.

## Scoring — how to read it

Each item contributes up to `max_points` total:

| Component                         | Per item                    |
|-----------------------------------|-----------------------------|
| `expected_sections` hit           | +1 per section found         |
| `expected_conclusion_keywords` hit| +1 per keyword found         |
| `must_not_say` hit                | -1 each                      |
| `must_cite_verbatim` gate pass    | +1 if `verified_quotes >= 1` |
| LLM-as-judge (0..3 scale)         | +0..+3                       |

Aggregate printed as `total_points / max_points` and percentage. 100% is not
realistic — target 70-85% on a healthy backend. Individual items worth 5-10
points each depending on how many expected refs/keywords they have.

### Regression gate rule

After any backend change:

1. Run `run_eval.py` -> new run dir.
2. `diff_runs.py --base <last_good> --new <this_run>`.
3. If aggregate percentage drops by **more than 5 percentage points**,
   the diff marks `GATE: REGRESSION`. Investigate before accepting the deploy.
4. Per-item regressions with delta <= -0.5 are listed separately — useful
   for spotting a specific question type that broke.

## How to add a new gold question

Append to `items:` in `gold_set.yaml`:

```yaml
- id: gst_pos_006               # <cat>_<subcat>_<nnn>, must be unique
  category: gst                 # gst | customs | central_excise | service_tax | others
  subcategory: place_of_supply  # free-form
  difficulty: basic             # basic | intermediate | complex
  question: "..."
  expected_sections: ["10(1)(a) IGST"]          # any-of; 1pt each if present in answer
  expected_rules: []                            # metadata only
  expected_notifications: []                    # metadata only
  expected_conclusion_keywords: ["IGST", "..."] # 1pt each if present
  must_not_say: ["CGST and SGST"]               # -1 each if present
  must_cite_verbatim: true                      # +1 if verified_quotes non-empty
  notes: "Why this question matters."
```

Guidelines:

- Keep `expected_conclusion_keywords` to 3-6 terms that MUST appear in any
  correct answer. Too many keywords inflate scores; too few make scoring noisy.
- `must_not_say` should flag wrong conclusions (e.g. "CGST+SGST" for an
  inter-state query). Don't use it for mere word avoidance.
- For legacy (pre-GST) questions, include `"pre-GST"` or similar in
  `expected_conclusion_keywords` so the harness rewards the model for
  correctly flagging legacy context.
- `must_cite_verbatim: true` for any factual/statutory question — forces
  the model to pass the verified-quote gate (relevant post-B16).

## Notes & gotchas

- **Latency**: full 50-item run with judge enabled takes roughly
  50 * (RAG latency + judge latency). At ~15-25s RAG + ~5s judge, expect
  **20-30 minutes serial**, ~7-10 min with `--workers 4`.
- **Judge**: uses `qwen3-14b` on `http://192.168.1.107:9082` (same model
  as the RAG backend). If you change the RAG LLM, also change the judge
  or set `--no-judge` to avoid the same model grading itself biased.
  Judge at temp=0 with tight system prompt; scores 0-3 integer.
- **Judge cost**: zero — runs on local GPU. Only latency cost.
- **Rig restart**: if the rig is cold, the first query takes 30s for
  embed warmup. Toss a `--limit 1` warm-up run before the real eval.
- **API schema**: runner expects `POST /v1/query {"question": "..."}` to
  return `{"answer": "...", "verified_quotes": [...]}`. If the schema
  changes, update `call_rag` in `run_eval.py`.
- **Case-insensitive substring match**: "Karnataka" in the answer matches
  "karnataka", "KARNATAKA", etc. Whitespace is normalized, punctuation is not.
