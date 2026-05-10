#!/usr/bin/env python3
"""Gemini-only synthetic enrichment of chunk payloads (no re-embed in this script).

For each canonical chunk in the v2 manifest, calls Gemini 2.5 Flash to generate
3 realistic long-scenario business questions. Updates the chunk's payload_json
to set:
    embed_text = parent_hierarchy + chunk_text + "\\n\\nUser questions answered:\\n" + Qs
    upserted = 0

After this script completes, run:
    /usr/bin/python3 reingest_spec/ingest_v2.py --phase phase3_4_5 --collection cbic_v2

That step uses the existing tested embedder pool to re-embed all chunks with
upserted=0 and write new dense vectors to Qdrant.

Why split into two steps: the embedder pool's multiprocessing.spawn workers
have PYTHONPATH/env issues when launched outside the run_batch_loop context.
Letting ingest_v2 do the embedding (since that path is proven to work) is more
reliable than trying to reproduce its env in a fresh script.

Reference: reingest_spec/CONSULT_BRIEF_2026-05-08.md.
"""
import os, sys, json, time, urllib.request, urllib.error, sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST = '/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite'
GEMINI_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}'
WORKERS = int(os.environ.get('ENRICH_WORKERS', '32'))
COMMIT_BATCH = 256

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


_DBG_FAIL_LOG = []
def gemini_call(passage: str, retries: int = 3) -> list:
    body = {
        "contents": [{"parts": [{"text": ENRICH_PROMPT.format(passage=passage[:3500])}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2000,
            "responseMimeType": "application/json",
            # 2026-05-08: Gemini 2.5 Flash defaults to spending ~1500 thinking tokens
            # before output. With max=1500 we got truncated 234-char responses (failed JSON).
            # thinkingBudget=0 disables thinking, returns answer directly. Verified 1.6s/call.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(GEMINI_URL, method='POST',
                data=json.dumps(body).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=45) as r:
                d = json.loads(r.read())
            cand = d.get('candidates') or []
            if not cand:
                last_err = f'no candidates: finishReason={d.get("promptFeedback")}'
                if attempt < retries - 1:
                    time.sleep(1); continue
                _DBG_FAIL_LOG.append(last_err)
                return []
            txt = cand[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            if not txt:
                last_err = f'empty text: finishReason={cand[0].get("finishReason")}'
                _DBG_FAIL_LOG.append(last_err)
                return []
            arr = json.loads(txt)
            if isinstance(arr, list) and len(arr) >= 1:
                return [str(q).strip() for q in arr if str(q).strip()][:5]
            last_err = f'not list/empty: {str(arr)[:120]}'
        except urllib.error.HTTPError as e:
            err_body = ''
            try: err_body = e.read().decode()[:200]
            except: pass
            last_err = f'HTTP {e.code}: {err_body}'
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt + 1); continue
            _DBG_FAIL_LOG.append(last_err)
            return []
        except Exception as e:
            last_err = f'{type(e).__name__}: {str(e)[:150]}'
            if attempt < retries - 1:
                time.sleep(0.5); continue
            _DBG_FAIL_LOG.append(last_err)
            return []
    if last_err:
        _DBG_FAIL_LOG.append(last_err)
    return []


def main():
    if not GEMINI_KEY:
        print('FATAL: GEMINI_API_KEY not set'); sys.exit(2)

    c = sqlite3.connect(MANIFEST)
    c.execute('PRAGMA journal_mode=WAL')

    # Find chunks needing enrichment: canonical, not yet enriched (we use a marker:
    # if embed_text already contains "[ENRICHED]" we skip).
    rows = list(c.execute(
        "SELECT chunk_id, payload_json FROM chunks WHERE is_canonical=1"
    ))
    print(f'total canonical chunks: {len(rows)}')

    need = []
    for cid, pj in rows:
        try:
            p = json.loads(pj)
        except Exception:
            continue
        if '[ENRICHED]' in (p.get('embed_text') or ''):
            continue  # already enriched
        if len(p.get('text') or '') < 100:
            continue  # skip tiny chunks (no signal)
        need.append((cid, p))
    print(f'chunks to enrich: {len(need)}')
    print(f'config: workers={WORKERS} model={GEMINI_MODEL}')
    if not need:
        print('nothing to do')
        return

    t0 = time.time()
    n_done = 0; n_fail = 0
    pending_updates = []

    def commit_pending():
        nonlocal pending_updates
        if not pending_updates:
            return
        c.executemany(
            "UPDATE chunks SET payload_json=?, upserted=0 WHERE chunk_id=?",
            pending_updates,
        )
        c.commit()
        pending_updates = []

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        # submit all
        futs = {ex.submit(gemini_call, p['text']): (cid, p) for cid, p in need}
        for f in as_completed(futs):
            cid, p = futs[f]
            try:
                qs = f.result()
            except Exception:
                qs = []
            if qs:
                ph = p.get('parent_hierarchy_text') or ''
                txt = p.get('text') or ''
                enriched = (ph + '\n\n' if ph else '') + txt + \
                    '\n\n[ENRICHED] User questions answered by this passage:\n' + \
                    '\n'.join(f'- {q}' for q in qs)
                p['embed_text'] = enriched
                pending_updates.append((json.dumps(p), cid))
                n_done += 1
            else:
                n_fail += 1
            if len(pending_updates) >= COMMIT_BATCH:
                commit_pending()
            if (n_done + n_fail) % 200 == 0:
                elapsed = time.time() - t0
                rate = (n_done + n_fail) / elapsed if elapsed > 0 else 0
                eta = (len(need) - n_done - n_fail) / rate if rate > 0 else 0
                print(f'  progress: done={n_done} fail={n_fail} '
                      f'rate={rate:.1f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)
                if _DBG_FAIL_LOG:
                    from collections import Counter
                    cnt = Counter(_DBG_FAIL_LOG)
                    print(f'  top fail patterns: {cnt.most_common(5)}', flush=True)
                    _DBG_FAIL_LOG.clear()

    commit_pending()
    elapsed = time.time() - t0
    print(f'\nfinished: {n_done} enriched, {n_fail} failed in {elapsed:.0f}s')
    print(f'\nNEXT STEP: run ingest_v2 phase3_4_5 to re-embed:')
    print('  cd /opt/indian-legal-ai && env DENSE_ONLY=1 EMBED_GPUS=0,1,3,4,5,6 \\')
    print('    PYTHONPATH=/home/user/.local/lib/python3.10/site-packages:/opt/indian-legal-ai/rag/cbic_rag \\')
    print('    /usr/bin/python3 reingest_spec/ingest_v2.py --phase phase3_4_5 --collection cbic_v2')


if __name__ == '__main__':
    main()
