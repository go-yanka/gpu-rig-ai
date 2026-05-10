# CBIC RAG — E-suite + Inline Popover — Agent Spec

**Target rig:** `user@192.168.1.107` (SSH key `~/.ssh/id_ed25519`, `-o ControlMaster=no -o ControlPath=none`)
**Files:** `/opt/indian-legal-ai/rag/cbic_rag/{api.py, storyformat.py, static/index.html}`
**Restart pattern:** `pkill -9 -f "api\.py$"; sleep 2; cd /tmp && nohup setsid bash /tmp/start_api.sh > /tmp/api.log 2>&1 < /dev/null & disown`  then wait 45s.

Do everything in **one patch script** named `/tmp/esuite_patch.py`. Backup every file as `.bak.esuite.<ts>` before editing. Fail loudly on missing anchors. Tag sentinel: `// esuite_v1` on the JS side, `# esuite_v1` on Python side. If sentinel already present, `print("ALREADY_PATCHED")` and exit 0.

Verify scope: all 5 categories (gst, customs, central_excise, service_tax, others). Fixes must be category-agnostic — no hard-coded "gst" paths.

---

## 1. `storyformat.py` — enrich citations

In `build_response_payload` (or wherever each citation dict is assembled), add these fields to every citation:

```python
'text_full': c.get('text', ''),
'date': c.get('date') or c.get('doc_date') or c.get('issued_date'),
'number': c.get('number') or c.get('circular_no') or c.get('notification_no'),
'category': c.get('category'),
'subcategory': c.get('subcategory'),
```

Also add a **stable query id** to the top-level response: `query_id: hashlib.blake2b(f"{question}|{time.time()}".encode(), digest_size=8).hexdigest()`. Used by the feedback endpoint and by the frontend popover cache.

---

## 2. `api.py` — new endpoints

### 2.1 `GET /v1/meta`

Returns static-ish facts about the running stack. Used by footer (E11) and the About popover.

```python
@app.get("/v1/meta")
def meta():
    import sqlite3 as _s3
    try:
        con = _s3.connect(f"file:{_MANIFEST}?mode=ro", uri=True, timeout=3)
        total_docs = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        con.close()
    except Exception:
        total_docs = None
    return {
        "llm_model": "qwen3-14B-hermes",
        "llm_url": os.environ.get("LLM_URL", ""),
        "embedder": "BGE-M3 (Vulkan GPU5)",
        "fusion": "RRF (dense + BM25)",
        "reranker": "ColBERT (CPU, lazy)",
        "verifier": "fuzzy (canon + label-strip + 6-gram 0.80)",
        "qdrant_coll": "cbic_v1",
        "top_k_retr": int(os.environ.get("TOP_K_RETR", 12)),
        "top_k_rerank": int(os.environ.get("TOP_K_RERANK", 6)),
        "total_docs": total_docs,
        "corpus": "CBIC (GST, Customs, Central Excise, Service Tax, Others)",
    }
```

### 2.2 `POST /v1/queries/{qid}/feedback`

```python
from pydantic import BaseModel
class FeedbackReq(BaseModel):
    kind: str            # "downvote_citation" | "downvote_quote" | "upvote_answer" | "thumb_down"
    citation_index: int | None = None
    quote: str | None = None
    reason: str | None = None

@app.post("/v1/queries/{qid}/feedback")
def feedback(qid: str, req: FeedbackReq):
    import json, time, pathlib
    rec = {"qid": qid, "ts": time.time(), **req.dict()}
    p = pathlib.Path("/opt/indian-legal-ai/rag/cbic_rag/_feedback.jsonl")
    with p.open("a") as f: f.write(json.dumps(rec) + "\n")
    return {"ok": True}
```

### 2.3 Confirm existing PDF endpoints still live

Three routes — **do not use `:path` converter** on any of them (that was the F2 bug). All use plain `{doc_id}`:
- `GET /pdf/{doc_id}` — full PDF
- `GET /pdf/{doc_id}/page/{page}/image` — page PNG
- `GET /pdf/{doc_id}/page/{page}/snippet?q=...` — cropped PNG

---

## 3. `static/index.html` — UI changes

### 3.1 E1 — Clickable S-badges in Verified/Flagged tabs

In `renderQuote`, the span `<span class="sn">S${sid}</span>` becomes:
```html
<span class="sn cliks" data-s="${sid}" style="cursor:pointer" title="Jump to evidence S${sid}">S${escHtml(sid)}</span>
```

In the renderResult post-render binding block, add:
```js
document.querySelectorAll('.cliks').forEach(el => {
  el.addEventListener('click', e => {
    const sid = e.currentTarget.dataset.s;
    // switch to Evidence tab
    document.querySelector('[data-tab="evidence"]')?.click();
    // scroll to citation S<sid>
    const tgt = document.querySelector(`[data-citation="${sid}"]`);
    if (tgt) { tgt.scrollIntoView({behavior:'smooth', block:'center'});
               tgt.classList.add('flash'); setTimeout(()=>tgt.classList.remove('flash'),1500); }
    // open the inline popover for this citation
    openCitePopover(sid);
  });
});
```

Evidence cards need `data-citation="${c.index}"` attribute added to their root `<div>`.

Add CSS:
```css
.flash { outline: 2px solid var(--green); outline-offset: 4px; animation: flash .4s ease-in-out 2 alternate; }
@keyframes flash { from{outline-color:var(--green)} to{outline-color:transparent} }
```

### 3.2 E4 — Inline PDF region snapshot in evidence card

In `renderCard`, after the main text, insert:
```html
${c.doc_id && c.page ? `
  <details class="snapdet">
    <summary style="cursor:pointer;font-size:12px;color:var(--dim)">📷 Show PDF region (p.${c.page})</summary>
    <img loading="lazy" class="snapimg" data-snipsrc="/pdf/${encodeURIComponent(c.doc_id)}/page/${c.page}/snippet?q=${encodeURIComponent((c.text||'').slice(0,120))}" style="max-width:100%;border:1px solid var(--line);border-radius:4px;margin-top:6px" />
  </details>` : ''}
```

In the binding block:
```js
document.querySelectorAll('details.snapdet').forEach(d => {
  d.addEventListener('toggle', () => {
    if (d.open) {
      const img = d.querySelector('img.snapimg');
      if (img && !img.src && img.dataset.snipsrc) img.src = img.dataset.snipsrc;
    }
  });
});
```

### 3.3 E5 — Bbox highlight (optional, light)

Cropping from F2 already shows only the matched region, so explicit rectangle overlay is redundant. Skip unless user later requests a full-page view with overlay. Mark done.

### 3.4 E6 — Verify-in-PDF link per verified quote

In `renderQuote`, after the existing PDF link, add:
```html
${docId && page ? `<a href="/pdf/${encodeURIComponent(docId)}/page/${page}/snippet?q=${encodeURIComponent(q.text||q.quote||'')}" target="_blank" rel="noopener" style="font-size:11px;margin-left:8px" title="See PDF snippet highlighting this quote">🔍 Verify in PDF</a>` : ''}
```

### 3.5 E8 — Full chunk text expand/collapse

`renderCard` gets:
```html
${c.text_full && c.text_full.length > (c.text||'').length ? `
  <details class="fullt">
    <summary style="font-size:11px;color:var(--dim);cursor:pointer">▸ Show full passage (${c.text_full.length} chars)</summary>
    <div class="ft" style="font-size:13px;white-space:pre-wrap;padding:8px;background:#fafafa;border-radius:4px;margin-top:6px">${escHtml(c.text_full)}</div>
  </details>` : ''}
```

### 3.6 E9 — Date + number badges

In the card header, add inline:
```html
${c.date ? `<span class="badge badge-date">${escHtml(c.date)}</span>` : ''}
${c.number ? `<span class="badge badge-num">${escHtml(c.number)}</span>` : ''}
```

CSS:
```css
.badge { display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;margin-left:6px;vertical-align:middle }
.badge-date { background:#e8f4ff;color:#0066aa }
.badge-num  { background:#fff4e8;color:#aa6600 }
```

### 3.7 E10 — "Not relevant" downvote

In the card footer, add:
```html
<button class="dvote" data-cidx="${c.index}" data-docid="${c.doc_id||''}" title="Report: not relevant" style="background:none;border:none;cursor:pointer;color:var(--dim);font-size:11px">⊯ Not relevant</button>
```

Binding:
```js
document.querySelectorAll('.dvote').forEach(b => {
  b.addEventListener('click', async e => {
    const btn = e.currentTarget;
    if (!window.lastQid) return;
    await fetch(`/v1/queries/${window.lastQid}/feedback`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({kind:'downvote_citation', citation_index: parseInt(btn.dataset.cidx), reason:'not_relevant'})
    });
    btn.textContent = '✓ Reported'; btn.disabled = true; btn.style.opacity = .5;
  });
});
```

In `renderResult`, set `window.lastQid = data.query_id;` at the top.

### 3.8 E11 — Dynamic footer

Replace the stale "Mistral Nemo 12B" string. Add `<div id="footMeta">Loading…</div>` in the footer, and before `</body>`:
```js
fetch('/v1/meta').then(r=>r.json()).then(m=>{
  const el = document.getElementById('footMeta');
  if (el) el.innerHTML = `${m.llm_model} · ${m.embedder} · ${m.fusion} · ${m.reranker} · ${m.total_docs?.toLocaleString()||'?'} chunks`;
}).catch(()=>{});
```

---

## 4. NEW — Inline Citation Popover (highest priority per user)

**Goal:** clicking an S-badge *or* a 👁 icon on any citation opens a slide-in panel that shows the evidence *inline* — no page reload, no new tab unless user asks. Panel has tabs: **Snapshot / Text / PDF**, plus an icon toolbar.

### 4.1 HTML — add at end of `<body>`:

```html
<div id="citePopover" class="cpop" style="display:none">
  <div class="cpop-head">
    <span id="cpopTitle">Citation</span>
    <div class="cpop-tools">
      <button class="cpop-tab" data-tab="snap" title="Snapshot">📷</button>
      <button class="cpop-tab" data-tab="text" title="Text">📄</button>
      <button class="cpop-tab" data-tab="pdf"  title="Full PDF">📘</button>
      <button id="cpopCopy" title="Copy link">🔗</button>
      <button id="cpopOpen" title="Open in new tab">⤢</button>
      <button id="cpopClose" title="Close">✕</button>
    </div>
  </div>
  <div class="cpop-body">
    <div class="cpop-pane cpop-snap"></div>
    <div class="cpop-pane cpop-text" style="display:none"></div>
    <div class="cpop-pane cpop-pdf"  style="display:none"></div>
  </div>
</div>
```

### 4.2 CSS:

```css
.cpop { position:fixed;top:60px;right:20px;width:min(520px,45vw);height:calc(100vh - 80px);
        background:#fff;border:1px solid var(--line);border-radius:8px;
        box-shadow:0 10px 40px rgba(0,0,0,.15);z-index:1000;display:flex;flex-direction:column }
.cpop-head { display:flex;justify-content:space-between;align-items:center;padding:10px 14px;
             border-bottom:1px solid var(--line);background:#fafafa;border-radius:8px 8px 0 0 }
.cpop-tools button { background:none;border:none;font-size:16px;cursor:pointer;padding:4px 8px;
                     border-radius:4px }
.cpop-tools button:hover { background:#eee }
.cpop-tools button.active { background:var(--green);color:#fff }
.cpop-body { flex:1;overflow:auto;padding:14px }
.cpop-pane img { max-width:100% }
.cpop-pane iframe { width:100%;height:100%;border:0 }
.cpop-pane .ft { white-space:pre-wrap;font-size:13px }
```

### 4.3 JS — API:

```js
let _cpopState = { sid:null, cite:null };

function openCitePopover(sid) {
  const c = (window.lastCitations||[]).find(x => String(x.index) === String(sid));
  if (!c) return;
  _cpopState = { sid, cite: c };
  const pop = document.getElementById('citePopover');
  pop.style.display = 'flex';
  document.getElementById('cpopTitle').textContent =
    `S${sid} · ${c.title || c.doc_id} ${c.page?'· p.'+c.page:''}`;
  // default tab: snapshot
  _cpopShowTab('snap');
}

function _cpopShowTab(name) {
  const pop = document.getElementById('citePopover');
  const c = _cpopState.cite;
  if (!c) return;
  pop.querySelectorAll('.cpop-tab').forEach(b => b.classList.toggle('active', b.dataset.tab===name));
  pop.querySelectorAll('.cpop-pane').forEach(p => p.style.display = 'none');
  if (name === 'snap') {
    const pane = pop.querySelector('.cpop-snap');
    if (c.doc_id && c.page) {
      pane.innerHTML = `<img src="/pdf/${encodeURIComponent(c.doc_id)}/page/${c.page}/snippet?q=${encodeURIComponent((c.text||'').slice(0,120))}" />`;
    } else pane.textContent = 'No page info.';
    pane.style.display = 'block';
  } else if (name === 'text') {
    const pane = pop.querySelector('.cpop-text');
    pane.innerHTML = `<div class="ft">${escHtml(c.text_full||c.text||'')}</div>`;
    pane.style.display = 'block';
  } else if (name === 'pdf') {
    const pane = pop.querySelector('.cpop-pdf');
    if (c.doc_id) {
      pane.innerHTML = `<iframe src="/pdf/${encodeURIComponent(c.doc_id)}${c.page?'#page='+c.page:''}"></iframe>`;
    }
    pane.style.display = 'block';
  }
}

// wire tabs
document.getElementById('citePopover').addEventListener('click', e => {
  const tab = e.target.closest('.cpop-tab');
  if (tab) _cpopShowTab(tab.dataset.tab);
  if (e.target.id === 'cpopClose') document.getElementById('citePopover').style.display = 'none';
  if (e.target.id === 'cpopCopy') {
    const c = _cpopState.cite;
    if (c?.doc_id) {
      const url = `${location.origin}/pdf/${encodeURIComponent(c.doc_id)}${c.page?'#page='+c.page:''}`;
      navigator.clipboard?.writeText(url).catch(()=>{
        const t=document.createElement('textarea');t.value=url;document.body.appendChild(t);t.select();
        document.execCommand('copy');document.body.removeChild(t);
      });
    }
  }
  if (e.target.id === 'cpopOpen') {
    const c = _cpopState.cite;
    if (c?.doc_id) window.open(`/pdf/${encodeURIComponent(c.doc_id)}${c.page?'#page='+c.page:''}`,'_blank');
  }
});
```

Store `window.lastCitations = data.citations || [];` in renderResult so popover can look up by sid.

### 4.4 Add 👁 icon to each evidence card header

Beside the existing "Open PDF" link in `renderCard`:
```html
<button class="eyebtn" data-sid="${c.index}" title="View inline" style="background:none;border:none;cursor:pointer;font-size:14px">👁</button>
```

Binding:
```js
document.querySelectorAll('.eyebtn').forEach(b => {
  b.addEventListener('click', e => openCitePopover(e.currentTarget.dataset.sid));
});
```

---

## 5. Verification checklist (agent runs after restart)

After `pkill + restart + sleep 45`, run all of these and report output:

```bash
# 1. meta endpoint live
curl -s http://localhost:9500/v1/meta | jq

# 2. feedback endpoint accepts
curl -s -X POST http://localhost:9500/v1/queries/test123/feedback \
  -H 'Content-Type: application/json' \
  -d '{"kind":"downvote_citation","citation_index":1,"reason":"test"}' | jq

# 3. snippet endpoint returns PNG (not JSON)
curl -sI 'http://localhost:9500/pdf/<SOME_DOC_ID>/page/1/snippet?q=tax' | head -5

# 4. run one query per category; inspect returned citations have text_full, date, number, query_id
for CAT in gst customs central_excise service_tax others; do
  curl -s -X POST http://localhost:9500/query \
    -H 'Content-Type: application/json' \
    -d "{\"question\":\"What are the key rules in $CAT?\",\"category\":\"$CAT\"}" \
    | jq '{qid:.query_id, n_cites:(.citations|length), sample:.citations[0]|{text_full:(.text_full|length), date, number, category}}'
done
```

Pass criteria:
- `/v1/meta` returns 200 with total_docs > 100000
- `/feedback` returns `{"ok":true}` and appends to `_feedback.jsonl`
- `snippet` returns `Content-Type: image/png`
- All 5 categories return `query_id` + citations with `text_full`, and no 5xx

## 6. Stop conditions (agent must halt and report)

- Any anchor-string `assert` fails
- API doesn't warm up after 60s
- `/health` returns non-200
- Any verification curl above fails

---

**Tell user when done:** "E-suite + inline popover deployed. Ctrl+Shift+R to bust browser cache."
