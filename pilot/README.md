# CBIC Verify — closed pilot

**Verdict (2026-09-28): ready for a closed pilot with 5–10 invited practitioners once the go-live checklist below is done. Not ready for open or public testing.**

Not public yet because:
- the answer model has never been measured on the v2 collection (last score 40% on v1);
- answers take ~30–60 s on the rig;
- retrieval finds the right document for ~84% of gold questions;
- the rig has a hardware fault;
- there is no case law in the corpus.

A closed pilot is how we find out which of these matter to real users, and every tester correction becomes a real, human-labelled gold question.

## What testers get

| Tab | What it does |
|---|---|
| Ask | Question (English or Hindi, short or a full business scenario). The answer quotes CBIC text word for word, each quote re-checked against its source. A banner shows confidence: verified / partly / not supported / not found / out of scope. |
| Notification status | Type `50/2017-Customs`, get "in force as amended / superseded by / rescinded by", with the sentence from the notification that proves it. |
| Test missions | 29 guided questions in 6 tracks (`test_pack.json`) plus "your own questions". Each has a checklist of what a good answer includes. |
| Rating | Correct? Right references? The correct reference if we missed it. Would you rely on it (1–5)? |

## What's new compared with the current API

All of these run in `backend/pilot_api.py`, which only calls the existing `/retrieve`. `cbic-rag-api` is not modified.

1. **Issue-spotting query rewrite + fusion** (`rag/cbic_rag/query_rewrite.py`). The LLM names the legal issue and restates the question in statute language, up to 3 queries. Results are merged with reciprocal-rank fusion. Legal-retrieval studies report +6–10 pts recall@10 from this. It is not the HyDE variant that hurt us (L2).
2. **Amendment chains** (`rag/cbic_rag/amendment_graph.py`). Deterministic regexes over notification text ("amends…", "rescinds…", "in supersession of…", "last amended by…"). Each link keeps its proving sentence. On the 1,908 sample chunks in the repo it found 183 links across 259 notifications; the full manifest will give many more. Tests: `rag/cbic_rag/test_amendment_graph.py`.
3. **Honest confidence.** "High" needs 2+ verified quotes and none unverified. If the sources don't answer, the model must say `NOT_FOUND` rather than use general knowledge.
4. **Hindi.** Hindi questions are rewritten into English search queries and answered in Hindi; quotes stay verbatim.
5. **Feedback that fixes the gold-set problem.** Every rating is logged with the tester's correct reference. `GET /api/export.csv` (admin token) exports it.

## Go-live checklist (about 1 day on the rig)

- [ ] **Rotate the leaked tester tokens.** `cloudflare_access_setup.md` has three live tokens committed to git. Make new ones in `/opt/cbic-auth/pilot_tokens.json` (`{"name@firm": "<token>"}`, chmod 600) and never commit that file.
- [ ] **Run the CLAUDE.md preflight** (`/preflight` with keywords `pilot`, `retrieval`, `service`). This adds a new service and a new retrieval path.
- [ ] Copy `rag/cbic_rag/amendment_graph.py` and `query_rewrite.py` to `/opt/indian-legal-ai/rag/cbic_rag/`, and `pilot/` to `/opt/indian-legal-ai/pilot/`. These are small files; the >150 MB SMB rule doesn't apply.
- [ ] Build the amendment index from the manifest (read-only; no Qdrant scroll):
      `python3 /opt/indian-legal-ai/rag/cbic_rag/amendment_graph.py build --manifest /opt/indian-legal-ai/data/ingest_manifest_v2.sqlite --out /opt/indian-legal-ai/data/amendment_graph.sqlite`
- [ ] **Choose the answer model** (your call). The default, local qwen3-14b on :9082, keeps data on the rig but is slower and weaker. A hosted model (Claude / Gemini through an OpenAI-compatible endpoint: set `LLM_URL`, `LLM_MODEL`, `LLM_KEY`) is faster and better, but questions and passages leave the rig. Tell testers which one they are using.
- [ ] Install `cbic-pilot.service` (below), start it, and check `curl localhost:9600/api/health` → all `true`.
- [ ] **Stable URL.** Point a *named* Cloudflare tunnel at `localhost:9600`. Quick tunnels change URL on every restart.
- [ ] **Internal dry run.** Do all 29 missions yourself. Go only if:
    - there are no errors;
    - every F-track trap is refused or answered "not found";
    - the A-track chains match the checklists;
    - median answer time is under 60 s.
- [ ] Invite 5–10 testers: 2–3 customs brokers, 2–3 CAs with customs/excise work, 1–2 importers. Send each one `https://<url>/?t=<token>`. Run a 2-week window and review the export weekly.

## What the pilot should tell us

From the export:
- % of answers rated correct;
- % with the right references;
- % "would rely" ≥ 4;
- how often "not found" was right versus a miss;
- which tracks fail.

The corrected references become the new, human-labelled gold set. That is the re-baselining step in the project review.

## Run it

```bash
# on the rig
cd /opt/indian-legal-ai/pilot/backend
CBIC_RAG_DIR=/opt/indian-legal-ai/rag/cbic_rag UPSTREAM_URL=http://127.0.0.1:9500 \
LLM_URL=http://127.0.0.1:9082 LLM_MODEL=qwen3-14b-q4_k_m.gguf \
GRAPH_DB=/opt/indian-legal-ai/data/amendment_graph.sqlite \
PILOT_TOKENS=/opt/cbic-auth/pilot_tokens.json PILOT_ADMIN_TOKEN=<secret> \
python3 -m uvicorn pilot_api:app --host 127.0.0.1 --port 9600

# anywhere, without the rig (plumbing only — the mock's answers mean nothing)
python3 pilot/dev/mock_rig.py --port 9555 &
python3 rag/cbic_rag/amendment_graph.py build --jsonl eval/training_pairs/pairs_2000_20260422.jsonl --out /tmp/g.sqlite
cd pilot/backend && CBIC_RAG_DIR=../../rag/cbic_rag:../../cbic_rag UPSTREAM_URL=http://127.0.0.1:9555 \
LLM_URL=http://127.0.0.1:9555 GRAPH_DB=/tmp/g.sqlite PILOT_OPEN=1 python3 -m uvicorn pilot_api:app --port 9600
```

Useful settings:
- `USE_REWRITE=0` turns the rewrite off, for an A/B comparison.
- `COLLECTION=cbic_v2` picks the Qdrant collection.
- `PILOT_RATE_PER_MIN` (default 6) and `PILOT_MAX_CONCURRENT` (default 2) protect the single-slot qwen3.
