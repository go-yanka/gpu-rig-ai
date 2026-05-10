# CBIC RAG v2 — Stage D–H Evaluators

Five standalone scripts that validate the freshly-built `cbic_v2` Qdrant
collection. Each:

- accepts `--collection cbic_v2` (default) and `--gold` / `--adv` path overrides
- writes a `*_result.json` next to itself
- prints a JSON summary to stdout
- exits **0 on pass, 2 on fail** (hard gate — CI/run orchestrator should stop)

All talk to the live API at `http://127.0.0.1:9500/query` (same as
`benchmarks/probes/probe_v16_theta.py`) so they must be run **on the rig**
(or via SSH port-forward).

## Run order

| # | Stage | Script | Prereq |
|---|-------|--------|--------|
| 1 | D | `probe_v2_runner.py`     | `cbic_v2` fully upserted |
| 2 | — | (run `../theta_tune.py --collection cbic_v2`) | V16 baseline from step 1 looks separable |
| 3 | E | `gate_g1_recall.py`      | `eval/v2_gold.json` exists (Stage C output) |
| 4 | F | `gate_g2_dual_judge.py`  | `GEMINI_API_KEY` + `ANTHROPIC_API_KEY` in env |
| 5 | G | `gate_g3_levenshtein.py` | `eval/v2_gold.json` (uses `expected_text` if present) |
| 6 | H | `gate_g4_adversarial.py` | `eval/v2_adversarial.json` + `../theta_v2.json` |

G1 must pass before G2 is meaningful; G3 recovers G1 near-misses so run it
regardless; G4 depends on `theta_v2.json` from `theta_tune.py`.

## Expected input schemas (Stage C must produce)

`eval/v2_gold.json`:
```json
{"version": 2, "queries": [
  {"id": "q001", "query": "...", "category": "gst", "subcategory": "itc",
   "expected_doc_ids": ["cbic-act-msts:1000006"],
   "expected_section_refs": ["Section 16(2)(c)"],
   "expected_chunk_ids": [],          // optional, strictest match
   "expected_text": "...",            // optional, used by G3 Levenshtein
   "expected_terms": ["ITC", "blocked credit"],
   "must_cite_verbatim": false
  }
]}
```

`eval/v2_adversarial.json`:
```json
{"version": 2, "queries": [
  {"id": "adv_001", "query": "What is Section 999 of CGST Act?",
   "attack_class": "fake_section"}
]}
```

Plain string entries in `queries[]` are also tolerated by G4.

## One-shot driver

```bash
cd /mnt/d/_gpu_rig_ai/reingest_spec/evaluators
python3 probe_v2_runner.py --collection cbic_v2        || exit 2
python3 ../theta_tune.py  --collection cbic_v2          || exit 2
python3 gate_g1_recall.py                               || exit 2
python3 gate_g3_levenshtein.py                          || exit 2
python3 gate_g4_adversarial.py                          || exit 2
python3 gate_g2_dual_judge.py                           || exit 2  # slowest; run last
echo "ALL GATES PASS"
```

## Dependencies

Standard library only (`urllib.request`, `json`, `unicodedata`, `re`,
`hashlib`). No `requests`, no `anthropic` / `google-generativeai` SDK — calls
are direct HTTP. G2 reads `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` from env;
update the Claude model string in `gate_g2_dual_judge.py` if the rig's Max
plan binds to a different ID.
