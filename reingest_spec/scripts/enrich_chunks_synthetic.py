#!/usr/bin/env python3
"""Synthetic scenario-question enrichment for cbic_v2 chunks.

For each chunk, ask Gemini to generate 3 realistic long-scenario business queries
that this chunk would answer. Concatenate onto chunk's embed_text and re-embed
via BGE-M3. Update Qdrant point's dense vector.

Why (codified 2026-05-08): both consultation responses agreed the bottleneck is
asymmetric search — gold queries are long business scenarios, chunk text is formal
statutory language. Enriching chunks with scenario-style questions makes BGE-M3
embeddings of those chunks closer to gold queries in vector space.

Reference: reingest_spec/CONSULT_BRIEF_2026-05-08.md.

Single-pass over all chunks. Parallel Gemini calls (32 workers).
Re-embed via cbic_rag.embedder pool. Update Qdrant in batches.
"""
import os, sys, json, time, urllib.request, urllib.error
# Critical: set PYTHONPATH BEFORE any embedder import — multiprocessing spawn
# workers inherit env from os.environ, not from the launching shell's vars.
_pp = '/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag'
os.environ['PYTHONPATH'] = _pp + (':' + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
sys.path.insert(0, '/opt/indian-legal-ai/rag/cbic_rag')
sys.path.insert(0, '/home/user/.local/lib/python3.10/site-packages')
from concurrent.futures import ThreadPoolExecutor, as_completed
from embedder import embed_dense_bulk
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = 'http://127.0.0.1:6343'
COLLECTION = 'cbic_v2'
GEMINI_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-exp')
GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}'
WORKERS = int(os.environ.get('ENRICH_WORKERS', '32'))
SCROLL_BATCH = 256
EMBED_BATCH = 64
QDRANT_UPDATE_BATCH = 64

ENRICH_PROMPT = """You are an expert Indian indirect-tax consultant familiar with CBIC notifications, GST/CGST/IGST Acts, Customs Act, Central Excise Act, and circulars.

Read the following passage from an Indian tax statute / notification / circular. Generate 3 distinct, realistic client-scenario questions that a tax consultant or business owner would ask, where THIS exact passage provides the key answer.

Style requirements:
- Long, narrative, business-context heavy (50-150 words each)
- Include fictional Indian company names (Pvt Ltd, LLP, Inc), Indian cities (Mumbai, Chennai, Bangalore, Delhi, Hyderabad, Pune, Kolkata, Ahmedabad, Surat, etc.), specific products/materials, exact monetary amounts (rupees lakhs/crores), and specific situations (import, manufacturing, SCN, exemption claim, audit, refund, classification dispute, drawback)
- Make them VARIED in scenario type
- Use professional but conversational tone
- The core legal point each question asks about MUST match what the passage establishes

Passage:
{passage}

Output ONLY a JSON array of 3 strings (the questions). No preamble, no explanation."""


def gemini_call(passage: str, retries: int = 3) -> list:
    """Call Gemini, return list of 3 scenario questions. Empty list on failure."""
    body = {
        "contents": [{"parts": [{"text": ENRICH_PROMPT.format(passage=passage[:3500])}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500, "responseMimeType": "application/json"},
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(GEMINI_URL, method='POST',
                data=json.dumps(body).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=45) as r:
                d = json.loads(r.read())
            txt = d['candidates'][0]['content']['parts'][0]['text']
            arr = json.loads(txt)
            if isinstance(arr, list) and len(arr) >= 1:
                return [str(q).strip() for q in arr if str(q).strip()][:5]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt + 1)
                continue
            return []
        except Exception:
            return []
    return []


def main():
    if not GEMINI_KEY:
        print('FATAL: GEMINI_API_KEY not set'); sys.exit(2)
    qc = QdrantClient(url=QDRANT_URL, timeout=120)
    n_total = qc.get_collection(COLLECTION).points_count
    print(f'cbic_v2 has {n_total} points to enrich')
    print(f'config: workers={WORKERS} model={GEMINI_MODEL}')

    t0 = time.time()
    n_enriched = 0; n_gemini_fail = 0; n_embed_fail = 0
    next_offset = None
    batch_count = 0

    while True:
        scroll = qc.scroll(collection_name=COLLECTION, limit=SCROLL_BATCH,
                           offset=next_offset,
                           with_payload=['text', 'parent_hierarchy_text', 'doc_id'],
                           with_vectors=False)
        points, next_offset = scroll
        if not points: break
        batch_count += 1

        # Step 1: parallel Gemini calls for the whole scroll batch (256 items)
        texts_for_gemini = []
        for p in points:
            pl = p.payload or {}
            text = pl.get('text') or ''
            if len(text) < 100:
                texts_for_gemini.append('')  # skip tiny chunks
            else:
                texts_for_gemini.append(text)

        questions_per_point = [None] * len(points)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(gemini_call, t): i for i, t in enumerate(texts_for_gemini) if t}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    questions_per_point[i] = f.result()
                except Exception:
                    questions_per_point[i] = []

        # Step 2: build new embed_text and re-embed in sub-batches
        # embed_text = parent_hierarchy + "\n\n" + text + "\n\nUser questions answered:\n" + questions
        new_embed_texts = []
        for p, qs in zip(points, questions_per_point):
            pl = p.payload or {}
            text = pl.get('text') or ''
            ph = pl.get('parent_hierarchy_text') or ''
            qs_str = ''
            if qs:
                qs_str = '\n\nUser questions answered by this passage:\n' + '\n'.join(f'- {q}' for q in qs)
            else:
                n_gemini_fail += 1
            new = (ph + '\n\n' if ph else '') + text + qs_str
            new_embed_texts.append(new)

        # Sub-batch re-embed
        upd_pids = []; upd_dense = []
        for i in range(0, len(new_embed_texts), EMBED_BATCH):
            sub = new_embed_texts[i:i+EMBED_BATCH]
            sub_pids = [p.id for p in points[i:i+EMBED_BATCH]]
            try:
                vecs = embed_dense_bulk(sub)
            except Exception as e:
                print(f'  embed fail at i={i}: {type(e).__name__}: {e}', flush=True)
                n_embed_fail += len(sub); continue
            for pid, v in zip(sub_pids, vecs):
                upd_pids.append(pid)
                upd_dense.append([float(x) for x in v])

        # Step 3: update Qdrant dense vectors
        for i in range(0, len(upd_pids), QDRANT_UPDATE_BATCH):
            sub_pids = upd_pids[i:i+QDRANT_UPDATE_BATCH]
            sub_vecs = upd_dense[i:i+QDRANT_UPDATE_BATCH]
            try:
                qc.update_vectors(
                    collection_name=COLLECTION,
                    points=[qm.PointVectors(id=pid, vector={'dense': vec})
                            for pid, vec in zip(sub_pids, sub_vecs)],
                )
                n_enriched += len(sub_pids)
            except Exception as e:
                print(f'  qdrant update fail at i={i}: {type(e).__name__}: {e}', flush=True)
                n_embed_fail += len(sub_pids)

        if batch_count % 1 == 0:
            elapsed = time.time() - t0
            rate = n_enriched / elapsed if elapsed > 0 else 0
            eta = (n_total - n_enriched) / rate if rate > 0 else 0
            print(f'  batch {batch_count} enriched={n_enriched}/{n_total} '
                  f'gemini_fail={n_gemini_fail} embed_fail={n_embed_fail} '
                  f'rate={rate:.1f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

        if next_offset is None: break

    elapsed = time.time() - t0
    print(f'\nfinished: {n_enriched} enriched, {n_gemini_fail} gemini-fail, '
          f'{n_embed_fail} embed-fail in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
