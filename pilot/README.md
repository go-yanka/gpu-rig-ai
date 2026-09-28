# CBIC Verify — closed pilot

**Verdict (2026-09-28): ready for a closed pilot with 5–10 invited practitioners once the checklist below is done. Not ready for open or public testing.**

It runs **entirely on your Windows desktop**:
- a local model through Ollama (or LM Studio);
- local search over an exported copy of the corpus;
- the rig is needed once, to export the corpus (read-only).

Not public yet because:
- the local model's answer quality on this corpus is unmeasured (last score 40% on v1, with qwen3-14b);
- local search has no cross-encoder reranker, so its recall is unmeasured (the rig reached ~84%);
- answers will take ~20–90 s, depending on your GPU;
- there is no case law in the corpus.

A closed pilot is how we learn which of these matter to real users. Every tester correction becomes a real, human-labelled gold question.

## What testers get

| Tab | What it does |
|---|---|
| Ask | Question (English or Hindi, short or a full business scenario). The answer quotes CBIC text word for word, each quote re-checked against its source. A banner shows confidence: verified / partly / not supported / not found / out of scope. |
| Notification status | Type `50/2017-Customs`, get "in force as amended / superseded by / rescinded by", with the sentence from the notification that proves it. |
| Test missions | 29 guided questions in 6 tracks (`test_pack.json`) plus "your own questions". Each has a checklist of what a good answer includes. |
| Rating | Correct? Right references? The correct reference if we missed it. Would you rely on it (1–5)? |

## What's new compared with the rig's API

1. **Issue-spotting query rewrite + fusion** (`rag/cbic_rag/query_rewrite.py`). The model names the legal issue and restates the question in statute language, up to 3 queries. Results are merged by reciprocal-rank fusion. Legal-retrieval studies report +6–10 pts recall@10 from this. It is not the HyDE variant that hurt us (L2).
2. **Amendment chains** (`rag/cbic_rag/amendment_graph.py`). Deterministic regexes over notification text ("amends…", "rescinds…", "in supersession of…", "last amended by…"). Each link keeps its proving sentence. On the 536 sample chunks in the repo it found 189 links across 260 notifications; the full corpus will give many more. Tests: `rag/cbic_rag/test_amendment_graph.py`.
3. **Made-up notifications are caught in code.** If a question cites a notification that is in neither the amendment index nor any retrieved text (e.g. "154/2026-Customs"), the app shows a red warning, tells the model not to describe it, and caps confidence at "low". This is the failure behind the 2026 Gujarat HC ruling.
4. **Honest confidence.** "High" needs 2+ verified quotes and none unverified. If the sources don't answer, the model must reply `NOT_FOUND` rather than use general knowledge.
5. **Hindi.** Hindi questions are rewritten into English search queries and answered in Hindi; quotes stay verbatim.
6. **Local hybrid search** (`pilot/local/local_retrieve.py`). SQLite full-text BM25 plus optional BGE-M3 "meaning" search through Ollama, fused. Same `/retrieve` contract as the rig, so the G1 evaluator can score it.
7. **Feedback that fixes the gold-set problem.** Every rating is logged with the tester's correct reference, and exported as CSV.

## Set up on your desktop (about half a day, mostly waiting)

1. **Export the corpus — the only step that touches the rig.** It reads the manifest; nothing is written on the rig's own disk.
   ```bash
   python3 pilot/local/export_chunks.py --manifest /opt/indian-legal-ai/data/ingest_manifest_v2.sqlite --out /mnt/d/_gpu_rig_ai/pilot_data
   ```
   Parts are small and checksummed, because of the rig's write-corruption fault. The desktop verifies every part. If one fails, re-export just that part with `--only N`.
2. **Install [Ollama](https://ollama.com/download) and Python 3.10+** on the desktop. Pick a model for your GPU:

   | GPU memory | Model |
   |---|---|
   | ≥ 12 GB | `qwen3:14b` |
   | 8 GB | `qwen3:8b` (default) |
   | No GPU | `qwen3:4b` (slow) |
3. **Start everything:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File pilot\local\run_local.ps1 -Model qwen3:8b -Dense
   ```
   On first run it:
   - installs Python packages;
   - pulls the model;
   - creates a 16K-context copy of the model (Ollama's default context would silently cut off sources);
   - builds the search index and the amendment index;
   - creates your owner key and prints your link.

   `-Dense` adds meaning search. It is better recall, but embedding ~50K chunks takes roughly 30–60 min on a GPU and hours on CPU. Leave it off for a first look.
4. **Measure search before inviting anyone**, with the same scorer and gold set as the rig:
   ```powershell
   $env:G1_RETRIEVE_API="http://127.0.0.1:9601/retrieve"; python reingest_spec\evaluators\gate_g1_recall.py --retrieve-only --out g1_local.json
   ```
5. **Dry run:** `python pilot\local\dry_run.py --token <owner key>`. Go only if:
   - there are no errors;
   - the F-track traps are refused or answered "not found";
   - median answer time is under 60 s.

   Then rate the missions yourself in the app.
6. **Add testers** to `pilot_data\pilot_tokens.json`: one line `"name@firm": "<random key>"` each. Never commit this file.
7. **Share.** Run `cloudflared tunnel --url http://localhost:9600` and send each tester `https://<url>/?t=<their key>`. Quick-tunnel URLs change on restart; a named tunnel with a domain gives a stable one. Your desktop must stay on while testers use it.
8. Invite 5–10 testers: 2–3 customs brokers, 2–3 CAs with customs/excise work, 1–2 importers. Run a 2-week window and review the export weekly (the launcher prints the command).

## What the pilot should tell us

From the export:
- % of answers rated correct;
- % with the right references;
- % "would rely" ≥ 4;
- how often "not found" was right versus a miss;
- which tracks fail.

The corrected references become the new, human-labelled gold set. That is the re-baselining step in the project review.

## Other ways to run it

- **On the rig instead:** set `UPSTREAM_URL=http://127.0.0.1:9500`, `LLM_URL=http://127.0.0.1:9082`, and `CBIC_RAG_DIR=/opt/indian-legal-ai/rag/cbic_rag`. `cbic-pilot.service` is a ready systemd unit. Run the CLAUDE.md preflight first.
- **Hosted model:** point `LLM_URL` / `LLM_MODEL` / `LLM_KEY` at any OpenAI-compatible endpoint. Questions and passages then leave your machine.
- **Plumbing test without any model** (answers are fake):
  ```bash
  python3 pilot/dev/mock_rig.py --port 9555 &
  python3 rag/cbic_rag/amendment_graph.py build --jsonl eval/training_pairs/pairs_2000_20260422.jsonl --out /tmp/g.sqlite
  cd pilot/backend && CBIC_RAG_DIR=../../rag/cbic_rag:../../cbic_rag UPSTREAM_URL=http://127.0.0.1:9555 \
  LLM_URL=http://127.0.0.1:9555 GRAPH_DB=/tmp/g.sqlite PILOT_OPEN=1 python3 -m uvicorn pilot_api:app --port 9600
  ```

Useful settings:
- `USE_REWRITE=0` turns the rewrite off, for an A/B comparison.
- `PILOT_RATE_PER_MIN` (default 6) and `PILOT_MAX_CONCURRENT` (1 on the desktop) protect a single local model.
