# Judge audit: qwen3-14B vs Claude (Sonnet 4.5 via Claude CLI)

**Date:** 2026-04-22
**Gold set:** 170 items
**RAG state:** P1.2 baseline (validator-simplified, two-pass OFF)
**Hypothesis under test:** external consult claimed "qwen3-14B judge is biased low; Claude/GPT-4o would reveal true baseline of 55-60%."

## Headline

| Judge | Mean score | Dist (0/1/2/3) | Final pass rate |
|---|---:|---|---:|
| qwen3-14B | **2.15** | 0 / 43 / 58 / 69 | **40.44%** |
| Claude | **0.91** | 18 / 151 / 0 / 1 | **24.86%** |

**The consult hypothesis is falsified.** Claude is *harsher*, not more lenient.

## Agreement

- Exact agreement: 37/170 (22%)
- Within 1 point: 99/170 (58%)
- qwen3 scored higher on 133 items; Claude higher on 0.

### Confusion matrix

| qwen3 ↓ / Claude → | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 1 | 7 | 36 | 0 | 0 |
| 2 | 3 | 55 | 0 | 0 |
| 3 | 8 | 60 | 0 | 1 |

Note: qwen3 never gave a 0 on this run; Claude gave a 2 exactly zero times. Claude's output distribution is nearly bimodal (0 or 1).

## Interpretation

1. **Claude is under-discriminating on this rubric.** 151/170 items got score=1 ("partially correct"). The rubric is a single-line scale with no examples. Strong models default toward conservative middle values when the rubric is ambiguous; that's what we're seeing.

2. **Disagreements cluster at qwen3=3 → Claude=1 (60 items).** qwen3 calls these "fully correct"; Claude calls them "partially correct". Without inspecting the answers, we can't tell which judge is right — but the systematic directionality (Claude always ≤ qwen3) suggests Claude is applying a stricter bar, not that qwen3 is biased low.

3. **The consult's predicted +15pp from judge swap does not materialize.** If anything the swap makes our baseline look worse (24.86% vs 40.44%).

## Caveats

- Minimal 0-3 rubric may under-calibrate Claude. A richer rubric with few-shot examples could change distribution.
- Single-judge replacement is weaker evidence than dual-judge consensus. The "right" metric might be `min(qwen3, claude)` or an independent third model breaking ties.
- Our pass-rate metric is compound: deterministic keyword matching + judge × 3 points. Swapping only the judge component shifts the numerator but may not reveal the true quality picture.

## Recommendation

**Abandon the "judge is biased" line of investigation.** The consult's strongest frame-shift claim doesn't hold up. The 40.44% number is the operational baseline; don't chase it.

**Continue with the other two consult recommendations** which are independent of judge choice:
- Q3 payload enrichment (cited_entities arrays via in-place scroll)
- Q1 single-pass CoT replacing two-pass
- Gemini's BGE-M3 contrastive tuning (if retrieval recall@5 turns out to be the bottleneck — test next)

## Files

- qwen3 judge run: `D:\_gpu_rig_ai\eval\runs\p1_2_baseline170_20260422_050751\`
- Claude judge run: `D:\_gpu_rig_ai\eval\runs\p1_2_claude_judge_20260422_104728\`
- Comparison script: inline above (can be re-run from the two per_item.jsonl files)
