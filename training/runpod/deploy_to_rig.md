# Deploy fine-tuned BGE-M3 to rig (192.168.1.107)

After `download.ps1` lands the model at `D:\_gpu_rig_ai\training\bge-m3-cbic-v1\`.

---

## CRITICAL DECISION: do we need to re-embed Qdrant?

**Short answer: YES, the full `cbic_v1` collection must be re-embedded.**

Rationale:
- `MultipleNegativesRankingLoss` back-props through the **entire encoder**
  (all 568M params of BGE-M3), not just a head. There is no separate
  "head" to swap — the output vectors of the fine-tuned model are in a
  different subspace than the base model.
- Mixing base-embedded chunks with fine-tuned-embedded queries at search
  time will produce garbage similarity scores. The existing 108k chunks in
  `cbic_v1` were embedded with stock `BAAI/bge-m3`.
- Dimensionality stays at 1024 (BGE-M3 untouched structurally), so Qdrant
  schema does not change — only the vectors inside.

Plan: embed into a new collection `cbic_v2`, cut over, keep `cbic_v1` as
rollback until recall audit passes.

The user's note in the task asks us to flag this. **Flagged: re-embed IS
required. Budget ~2-4 hrs for 108k chunks at 5.2-9.6 ch/s** (per
`ingest_playbook_cbic.md`), on single GPU 5 Vulkan. No RunPod needed for
re-embed — rig handles it fine.

---

## 1. Ship model to rig

```powershell
# from laptop
scp -r D:\_gpu_rig_ai\training\bge-m3-cbic-v1 `
       rig:/tmp/bge-m3-cbic-v1
ssh rig 'sudo mv /tmp/bge-m3-cbic-v1 /opt/ai-models/bge-m3-cbic-v1 && \
         sudo chown -R ai-models:ai-models /opt/ai-models/bge-m3-cbic-v1'
```

Canonical location (matches existing convention from `cbic_ingest_runbook.md`):
`/opt/ai-models/bge-m3-cbic-v1/`

---

## 2. Two serving options

The existing pipeline uses **GGUF F16 on llama.cpp Vulkan** (see
`cbic_ingest_runbook.md`). The fine-tune outputs a **sentence-transformers
safetensors dir** — these are incompatible.

### Option A: parallel sentence-transformers HTTP endpoint (FAST)
Spin up FastAPI + `SentenceTransformer.encode()` on GPU 5 (or any free GPU).
Swap ingest + query paths to point at it. 1-2 hours of work; no conversion
risk. RECOMMENDED for v1.

```bash
ssh rig
cat >/opt/ai-models/bge-m3-cbic-v1/serve.py <<'PY'
import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
os.environ.setdefault("HIP_VISIBLE_DEVICES", "5")
m = SentenceTransformer("/opt/ai-models/bge-m3-cbic-v1", device="cuda")
app = FastAPI()
class Req(BaseModel): texts: list[str]
@app.post("/embed")
def embed(r: Req): return {"vectors": m.encode(r.texts, normalize_embeddings=True).tolist()}
PY
# Serve on port 9580 (next free port per rig convention)
# uvicorn serve:app --host 0.0.0.0 --port 9580 --workers 1
```

Wire via new systemd unit `bge-m3-cbic-ft.service`.

### Option B: convert to GGUF (SLOW, reuses existing path)
`python convert-hf-to-gguf.py /opt/ai-models/bge-m3-cbic-v1 --outtype f16`
then plug into the existing `embedder_direct.py` pool. Defer to v2.

---

## 3. Re-embed into `cbic_v2`

```bash
ssh rig
cd /opt/indian-legal-ai/rag/cbic_rag
python tools/reembed_collection.py \
    --src cbic_v1 \
    --dst cbic_v2 \
    --embedder http://127.0.0.1:9580/embed
# ~2-4 hours for 108k chunks
```

If `tools/reembed_collection.py` does not exist yet, spin it up: scroll
Qdrant, batch 64, POST to /embed, upsert into `cbic_v2` with same payload.

---

## 4. Flip the RAG API to `cbic_v2`

Edit `/opt/indian-legal-ai/rag/cbic_rag/config.yaml` (or equivalent env var):
```yaml
qdrant_collection: cbic_v2
embedder_url: http://127.0.0.1:9580/embed
```

Restart:
```bash
ssh rig sudo systemctl restart cbic-rag-api.service
ssh rig sudo systemctl status  cbic-rag-api.service --no-pager
# tail the logs to confirm "embedder=bge-m3-cbic-v1" and "collection=cbic_v2"
ssh rig 'journalctl -u cbic-rag-api.service -n 50 --no-pager'
```

Smoke test:
```bash
curl -s http://192.168.1.107:9500/query -d '{"q":"ITC reversal under rule 42"}' \
     -H 'content-type: application/json' | jq .
```

---

## 5. Re-run recall audit

```powershell
cd D:\_gpu_rig_ai\eval
python recall_audit_rig.py --collection cbic_v2 --out recall_audit_cbic_v2.jsonl
python run_eval.py --run recall_audit_cbic_v2.jsonl
```

Target: recall@5 jumps from 12.35% to **>=40%**. If not, mine harder negatives
and re-train (v2 roadmap in `training/README.md`).

---

## 6. Rollback plan

If recall REGRESSES or the new endpoint is flaky:
```bash
# flip config back to cbic_v1 + old embedder URL
ssh rig sudo systemctl restart cbic-rag-api.service
# cbic_v2 can stay on disk; drop later with `qdrant delete-collection cbic_v2`
```
