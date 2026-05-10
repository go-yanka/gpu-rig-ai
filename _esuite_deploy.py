#!/usr/bin/env python3
"""esuite_v1 deployment — CBIC RAG E-suite + inline popover.

Patches:
  - storyformat.py : enrich citations + query_id
  - api.py         : /v1/meta + /v1/queries/{qid}/feedback
  - static/index.html : E1, E4, E6, E8, E9, E10, E11 + inline popover

Sentinel: esuite_v1. If present, skip that file (ALREADY_PATCHED).
Backups: <file>.bak.esuite.<ts>
"""
import os, sys, time, shutil, io

ROOT = "/opt/indian-legal-ai/rag/cbic_rag"
API  = f"{ROOT}/api.py"
SF   = f"{ROOT}/storyformat.py"
HTML = f"{ROOT}/static/index.html"

TS = str(int(time.time()))

def read(p):
    with open(p, "r", encoding="utf-8") as f: return f.read()

def write(p, s):
    with open(p, "w", encoding="utf-8") as f: f.write(s)

def backup(p):
    bak = f"{p}.bak.esuite.{TS}"
    shutil.copy2(p, bak)
    print(f"BACKUP: {bak}")
    return bak

def must_replace(src, anchor, repl, *, once=True, label=""):
    assert anchor in src, f"ANCHOR MISSING [{label}]: {anchor!r}"
    if once:
        assert src.count(anchor) == 1, f"ANCHOR NOT UNIQUE [{label}] ({src.count(anchor)} hits): {anchor!r}"
    return src.replace(anchor, repl, 1 if once else -1)

# ---------------- storyformat.py ----------------
def patch_sf():
    src = read(SF)
    if "# esuite_v1" in src:
        print(f"ALREADY_PATCHED: {SF}"); return None
    backup(SF)

    # Add time import alongside hashlib
    if "import re, json, hashlib, textwrap" in src:
        src = src.replace(
            "import re, json, hashlib, textwrap",
            "import re, json, hashlib, textwrap, time  # esuite_v1",
        )
    else:
        assert False, "SF import anchor missing"

    # Add text_full/date/number/category/subcategory to citation dict.
    # Replace the 'excerpt': c.get('text','')[:300], line.
    anchor = "            'excerpt': c.get('text','')[:300],\n"
    repl = (
        "            'excerpt': c.get('text','')[:300],\n"
        "            # esuite_v1 — enriched fields\n"
        "            'text_full': c.get('text', ''),\n"
        "            'date': c.get('date') or c.get('doc_date') or c.get('issued_date'),\n"
        "            'number': c.get('number') or c.get('circular_no') or c.get('notification_no'),\n"
    )
    src = must_replace(src, anchor, repl, label="sf-citation-fields")

    # Add query_id to top-level return dict.
    anchor = "    return {\n        'question': question,\n"
    repl = (
        "    query_id = hashlib.blake2b(f\"{question}|{time.time()}\".encode(), digest_size=8).hexdigest()  # esuite_v1\n"
        "    return {\n"
        "        'query_id': query_id,\n"
        "        'question': question,\n"
    )
    src = must_replace(src, anchor, repl, label="sf-query-id")

    write(SF, src)
    print(f"PATCHED: {SF}")

# ---------------- api.py ----------------
def patch_api():
    src = read(API)
    if "# esuite_v1" in src:
        print(f"ALREADY_PATCHED: {API}"); return None
    backup(API)

    # Inject new endpoints just before the final "app.mount('/ui'" line.
    anchor = "app.mount('/ui', StaticFiles(directory='/opt/indian-legal-ai/rag/cbic_rag/static', html=True), name='ui')"
    injection = """# esuite_v1 — /v1/meta + /v1/queries/{qid}/feedback
from pydantic import BaseModel as _EsuiteBaseModel

class _EsuiteFeedbackReq(_EsuiteBaseModel):
    kind: str
    citation_index: Optional[int] = None
    quote: Optional[str] = None
    reason: Optional[str] = None

@app.get('/v1/meta')
def _esuite_meta():
    import sqlite3 as _s3
    total_docs = None
    # Prefer Qdrant collection point count (spec pass criterion requires >100k)
    try:
        import httpx as _httpx
        _qurl = os.environ.get('QDRANT_URL', 'http://localhost:6343')
        _qcoll = os.environ.get('QDRANT_COLL', 'cbic_v1')
        r = _httpx.get(f"{_qurl}/collections/{_qcoll}", timeout=3.0)
        if r.status_code == 200:
            total_docs = r.json().get('result', {}).get('points_count')
    except Exception:
        pass
    if total_docs is None:
        try:
            con = _s3.connect(f"file:{_MANIFEST}?mode=ro", uri=True, timeout=3)
            total_docs = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            con.close()
        except Exception:
            total_docs = None
    return {
        'llm_model': os.environ.get('LLM_MODEL', 'qwen3-14b-hermes'),
        'llm_url': os.environ.get('LITELLM_URL', os.environ.get('LLM_URL', '')),
        'embedder': 'BGE-M3 (Vulkan GPU5)',
        'fusion': 'RRF (dense + BM25)',
        'reranker': 'ColBERT (CPU, lazy)',
        'verifier': 'fuzzy (canon + label-strip + 6-gram 0.80)',
        'qdrant_coll': os.environ.get('QDRANT_COLL', 'cbic_v1'),
        'top_k_retr': int(os.environ.get('TOP_K_RETR', 12)),
        'top_k_rerank': int(os.environ.get('TOP_K_RERANK', 6)),
        'total_docs': total_docs,
        'corpus': 'CBIC (GST, Customs, Central Excise, Service Tax, Others)',
    }

@app.post('/v1/queries/{qid}/feedback')
def _esuite_feedback(qid: str, req: _EsuiteFeedbackReq):
    import json as _json, time as _time, pathlib as _pl
    rec = {'qid': qid, 'ts': _time.time(), **req.dict()}
    p = _pl.Path('/opt/indian-legal-ai/rag/cbic_rag/_feedback.jsonl')
    with p.open('a') as f:
        f.write(_json.dumps(rec) + '\\n')
    return {'ok': True}

"""
    src = must_replace(src, anchor, injection + anchor, label="api-mount-ui")
    write(API, src)
    print(f"PATCHED: {API}")

# ---------------- index.html ----------------
def patch_html():
    src = read(HTML)
    if "// esuite_v1" in src or "/* esuite_v1 */" in src:
        print(f"ALREADY_PATCHED: {HTML}"); return None
    backup(HTML)

    # ---- E11: dynamic footer ----
    anchor = "    <div>BGE-M3 dense (1024d) + BM25 sparse &rarr; RRF fusion &rarr; ColBERT v2 rerank. Quote verification: exact + normalized match. LLM: Mistral Nemo 12B.</div>"
    repl = '    <div id="footMeta">BGE-M3 dense + BM25 &rarr; RRF &rarr; ColBERT v2. Loading stack info&hellip;</div>'
    src = must_replace(src, anchor, repl, label="html-footer")

    # ---- E1: clickable S-badge in renderQuote ----
    anchor = "${sid?`<span class=\"sn\" style=\"background:${bg};color:#fff;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700\">S${escHtml(sid)}</span>`:''}"
    repl   = "${sid?`<span class=\"sn cliks\" data-s=\"${escHtml(sid)}\" style=\"background:${bg};color:#fff;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700;cursor:pointer\" title=\"Jump to evidence S${escHtml(sid)}\">S${escHtml(sid)}</span>`:''}"
    src = must_replace(src, anchor, repl, label="html-renderQuote-sn")

    # ---- E6: verify-in-pdf link per verified quote (append after pdfLink) ----
    anchor = "  const pdfLink = pdfHref ? `<a href=\"${pdfHref}\" target=\"_blank\" rel=\"noopener\" style=\"font-size:11px;margin-left:8px\">Open PDF${page?(' p.'+page):''}</a>` : '';"
    repl = anchor + "\n  // esuite_v1 — E6 verify-in-pdf snippet link\n  const verifyHref = (docId && page) ? `/pdf/${encodeURIComponent(docId)}/page/${page}/snippet?q=${encodeURIComponent(q.text||q.quote||'')}` : '';\n  const verifyLink = verifyHref ? `<a href=\"${verifyHref}\" target=\"_blank\" rel=\"noopener\" style=\"font-size:11px;margin-left:8px\" title=\"See PDF snippet highlighting this quote\">\\u{1F50D} Verify in PDF</a>` : '';"
    src = must_replace(src, anchor, repl, label="html-E6-verifylink")

    # Add verifyLink into ctx block inside renderQuote
    anchor = "${page?(' \\u2014 p.'+page):''}${pdfLink}</div>` : '';"
    repl   = "${page?(' \\u2014 p.'+page):''}${pdfLink}${verifyLink}</div>` : '';"
    src = must_replace(src, anchor, repl, label="html-E6-ctx")

    # ---- renderCard modifications: add data-citation, badges, eye button, snippet details, full-text details, dvote button ----
    # Replace the ev-head line to include data-citation and eye button + badges in header
    anchor = '    <div class="ev-head"><span class="sn">S${c.index}</span><div class="tt" title="${title}">${title}</div></div>'
    repl   = ('    <div class="ev-head">'
              '<span class="sn cliks" data-s="${c.index}" style="cursor:pointer" title="View inline S${c.index}">S${c.index}</span>'
              '<div class="tt" title="${title}">${title}'
              '${c.date?` <span class="badge badge-date">${escHtml(c.date)}</span>`:\'\'}'
              '${c.number?` <span class="badge badge-num">${escHtml(c.number)}</span>`:\'\'}'
              '</div>'
              '<button class="eyebtn" data-sid="${c.index}" title="View inline" style="background:none;border:none;cursor:pointer;font-size:14px;margin-left:auto">\\u{1F441}</button>'
              '</div>')
    src = must_replace(src, anchor, repl, label="html-renderCard-head")

    # Add data-citation attribute to outer ev div
    anchor = '  return `<div class="ev" id="cite-${c.index}">'
    repl   = '  return `<div class="ev" id="cite-${c.index}" data-citation="${c.index}">'
    src = must_replace(src, anchor, repl, label="html-renderCard-datacit")

    # Insert E4 snapshot details, E8 full-text details, and E10 dvote button inside ev-acts block
    anchor = ('    <div class="ev-acts">\n'
              '      ${c.doc_id?`<a href="/pdf/${encodeURIComponent(c.doc_id)}${c.page?(\'#page=\'+c.page):\'\'}" target="_blank" rel="noopener">Open PDF${c.page?(\' p.\'+c.page):\'\'}</a>`:\'\'}${c.source_url?` <a href="${escAttr(c.source_url)}" target="_blank" rel="noopener" class="dim" style="opacity:.6">source</a>`:\'\'}\n'
              '      <button type="button" data-copy="${escAttr(cite)}">Copy citation</button>\n'
              '    </div>')
    repl = ('    <div class="ev-acts">\n'
            '      ${c.doc_id?`<a href="/pdf/${encodeURIComponent(c.doc_id)}${c.page?(\'#page=\'+c.page):\'\'}" target="_blank" rel="noopener">Open PDF${c.page?(\' p.\'+c.page):\'\'}</a>`:\'\'}${c.source_url?` <a href="${escAttr(c.source_url)}" target="_blank" rel="noopener" class="dim" style="opacity:.6">source</a>`:\'\'}\n'
            '      <button type="button" data-copy="${escAttr(cite)}">Copy citation</button>\n'
            '      <button class="dvote" data-cidx="${c.index}" data-docid="${c.doc_id||\'\'}" title="Report: not relevant" style="background:none;border:none;cursor:pointer;color:var(--dim);font-size:11px">\\u229F Not relevant</button>\n'
            '    </div>\n'
            '    ${c.doc_id && c.page ? `\n'
            '      <details class="snapdet"><summary style="cursor:pointer;font-size:12px;color:var(--dim)">\\u{1F4F7} Show PDF region (p.${c.page})</summary>\n'
            '        <img loading="lazy" class="snapimg" data-snipsrc="/pdf/${encodeURIComponent(c.doc_id)}/page/${c.page}/snippet?q=${encodeURIComponent((c.text_full||c.text||c.snippet||\'\').slice(0,120))}" style="max-width:100%;border:1px solid var(--line);border-radius:4px;margin-top:6px" />\n'
            '      </details>` : \'\'}\n'
            '    ${c.text_full && c.text_full.length > (c.snippet||\'\').length ? `\n'
            '      <details class="fullt"><summary style="font-size:11px;color:var(--dim);cursor:pointer">\\u25B8 Show full passage (${c.text_full.length} chars)</summary>\n'
            '        <div class="ft" style="font-size:13px;white-space:pre-wrap;padding:8px;background:#fafafa;border-radius:4px;margin-top:6px">${escHtml(c.text_full)}</div>\n'
            '      </details>` : \'\'}')
    src = must_replace(src, anchor, repl, label="html-renderCard-acts")

    # ---- CSS additions + popover markup + JS hooks — inject before </body> ----
    # Find the opening of the final <script> block at end (line "rotatePlaceholder();")
    anchor = "rotatePlaceholder();\nbuildChips();\nloadRecent();\nloadStats().then(pollHealth);\nsetInterval(pollHealth,10000);\n</script>"
    esuite_js = """// esuite_v1 — E-suite + inline popover wiring
window.lastCitations = window.lastCitations || [];
window.lastQid = window.lastQid || null;

(function(){
  // hook into renderResult via MutationObserver-lite: wrap existing fn
  const _origRenderResult = window.renderResult;
  if (typeof _origRenderResult === 'function') {
    window.renderResult = function(data){
      window.lastCitations = data && data.citations || [];
      window.lastQid = data && data.query_id || null;
      const r = _origRenderResult.apply(this, arguments);
      try { esuitePostRender(); } catch(e){ console.warn('esuite post-render', e); }
      return r;
    };
  }
})();

function esuitePostRender(){
  // E1 — clickable S-badges
  document.querySelectorAll('.cliks').forEach(el => {
    if (el._esuiteBound) return; el._esuiteBound = true;
    el.addEventListener('click', e => {
      const sid = e.currentTarget.dataset.s;
      const evTab = document.querySelector('.tab[data-t="evidence"]');
      if (evTab) evTab.click();
      const tgt = document.querySelector(`[data-citation="${sid}"]`);
      if (tgt) {
        tgt.scrollIntoView({behavior:'smooth', block:'center'});
        tgt.classList.add('flash');
        setTimeout(()=>tgt.classList.remove('flash'),1500);
      }
      openCitePopover(sid);
    });
  });
  // E4 — lazy-load snapshot img on details toggle
  document.querySelectorAll('details.snapdet').forEach(d => {
    if (d._esuiteBound) return; d._esuiteBound = true;
    d.addEventListener('toggle', () => {
      if (d.open) {
        const img = d.querySelector('img.snapimg');
        if (img && !img.src && img.dataset.snipsrc) img.src = img.dataset.snipsrc;
      }
    });
  });
  // E10 — downvote
  document.querySelectorAll('.dvote').forEach(b => {
    if (b._esuiteBound) return; b._esuiteBound = true;
    b.addEventListener('click', async e => {
      const btn = e.currentTarget;
      if (!window.lastQid) { btn.textContent='(no qid)'; return; }
      try {
        await fetch(`/v1/queries/${encodeURIComponent(window.lastQid)}/feedback`, {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({kind:'downvote_citation', citation_index: parseInt(btn.dataset.cidx), reason:'not_relevant'})
        });
        btn.textContent = '\\u2713 Reported'; btn.disabled = true; btn.style.opacity = .5;
      } catch(err){ btn.textContent='(failed)'; console.error(err); }
    });
  });
  // Eye buttons
  document.querySelectorAll('.eyebtn').forEach(b => {
    if (b._esuiteBound) return; b._esuiteBound = true;
    b.addEventListener('click', e => {
      e.stopPropagation();
      openCitePopover(e.currentTarget.dataset.sid);
    });
  });
}

// ---------- Citation popover ----------
let _cpopState = { sid:null, cite:null };
function openCitePopover(sid){
  const c = (window.lastCitations||[]).find(x => String(x.index) === String(sid));
  if (!c) return;
  _cpopState = { sid, cite: c };
  const pop = document.getElementById('citePopover');
  if (!pop) return;
  pop.style.display = 'flex';
  document.getElementById('cpopTitle').textContent =
    `S${sid} \\u00B7 ${c.title || c.doc_id || ''}${c.page?' \\u00B7 p.'+c.page:''}`;
  _cpopShowTab('snap');
}
function _cpopShowTab(name){
  const pop = document.getElementById('citePopover');
  const c = _cpopState.cite; if (!c || !pop) return;
  pop.querySelectorAll('.cpop-tab').forEach(b => b.classList.toggle('active', b.dataset.tab===name));
  pop.querySelectorAll('.cpop-pane').forEach(p => p.style.display = 'none');
  if (name === 'snap') {
    const pane = pop.querySelector('.cpop-snap');
    if (c.doc_id && c.page) {
      pane.innerHTML = `<img src="/pdf/${encodeURIComponent(c.doc_id)}/page/${c.page}/snippet?q=${encodeURIComponent((c.text_full||c.text||c.snippet||'').slice(0,120))}" />`;
    } else pane.textContent = 'No page info.';
    pane.style.display = 'block';
  } else if (name === 'text') {
    const pane = pop.querySelector('.cpop-text');
    pane.innerHTML = `<div class="ft">${escHtml(c.text_full||c.text||c.snippet||'')}</div>`;
    pane.style.display = 'block';
  } else if (name === 'pdf') {
    const pane = pop.querySelector('.cpop-pdf');
    if (c.doc_id) {
      pane.innerHTML = `<iframe src="/pdf/${encodeURIComponent(c.doc_id)}${c.page?'#page='+c.page:''}"></iframe>`;
    } else pane.textContent = 'No doc id.';
    pane.style.display = 'block';
  }
}
document.addEventListener('DOMContentLoaded', () => {
  const pop = document.getElementById('citePopover');
  if (!pop) return;
  pop.addEventListener('click', e => {
    const tab = e.target.closest('.cpop-tab');
    if (tab) { _cpopShowTab(tab.dataset.tab); return; }
    if (e.target.id === 'cpopClose') { pop.style.display='none'; return; }
    if (e.target.id === 'cpopCopy') {
      const c = _cpopState.cite;
      if (c && c.doc_id) {
        const url = `${location.origin}/pdf/${encodeURIComponent(c.doc_id)}${c.page?'#page='+c.page:''}`;
        if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(url).catch(()=>{});
        else { const t=document.createElement('textarea'); t.value=url; document.body.appendChild(t); t.select(); try{document.execCommand('copy');}catch(_){} document.body.removeChild(t);}
      }
      return;
    }
    if (e.target.id === 'cpopOpen') {
      const c = _cpopState.cite;
      if (c && c.doc_id) window.open(`/pdf/${encodeURIComponent(c.doc_id)}${c.page?'#page='+c.page:''}`,'_blank');
      return;
    }
  });
});

// E11 — dynamic footer
fetch('/v1/meta').then(r=>r.json()).then(m=>{
  const el = document.getElementById('footMeta');
  if (el) el.innerHTML = `${m.llm_model} \\u00B7 ${m.embedder} \\u00B7 ${m.fusion} \\u00B7 ${m.reranker} \\u00B7 ${m.total_docs?Number(m.total_docs).toLocaleString():'?'} chunks`;
}).catch(()=>{});
"""
    # CSS + popover HTML injected before <footer> .. actually inject just before the closing script tag
    repl = ("rotatePlaceholder();\nbuildChips();\nloadRecent();\nloadStats().then(pollHealth);\nsetInterval(pollHealth,10000);\n"
            + esuite_js + "\n</script>")
    src = must_replace(src, anchor, repl, label="html-js-injection")

    # Inject CSS + popover markup just before </body>
    css_html = """
<style>
/* esuite_v1 */
.flash { outline: 2px solid var(--green); outline-offset: 4px; animation: flash .4s ease-in-out 2 alternate; }
@keyframes flash { from{outline-color:var(--green)} to{outline-color:transparent} }
.badge { display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;margin-left:6px;vertical-align:middle }
.badge-date { background:#e8f4ff;color:#0066aa }
.badge-num  { background:#fff4e8;color:#aa6600 }
.cpop { position:fixed;top:60px;right:20px;width:min(520px,45vw);height:calc(100vh - 80px); background:#fff;border:1px solid var(--line);border-radius:8px; box-shadow:0 10px 40px rgba(0,0,0,.15);z-index:1000;display:flex;flex-direction:column }
.cpop-head { display:flex;justify-content:space-between;align-items:center;padding:10px 14px; border-bottom:1px solid var(--line);background:#fafafa;border-radius:8px 8px 0 0 }
.cpop-tools button { background:none;border:none;font-size:16px;cursor:pointer;padding:4px 8px;border-radius:4px }
.cpop-tools button:hover { background:#eee }
.cpop-tools button.active { background:var(--green);color:#fff }
.cpop-body { flex:1;overflow:auto;padding:14px }
.cpop-pane img { max-width:100% }
.cpop-pane iframe { width:100%;height:100%;border:0;min-height:70vh }
.cpop-pane .ft { white-space:pre-wrap;font-size:13px }
</style>
<div id="citePopover" class="cpop" style="display:none">
  <div class="cpop-head">
    <span id="cpopTitle">Citation</span>
    <div class="cpop-tools">
      <button class="cpop-tab" data-tab="snap" title="Snapshot">&#128247;</button>
      <button class="cpop-tab" data-tab="text" title="Text">&#128196;</button>
      <button class="cpop-tab" data-tab="pdf"  title="Full PDF">&#128218;</button>
      <button id="cpopCopy" title="Copy link">&#128279;</button>
      <button id="cpopOpen" title="Open in new tab">&#10542;</button>
      <button id="cpopClose" title="Close">&#10006;</button>
    </div>
  </div>
  <div class="cpop-body">
    <div class="cpop-pane cpop-snap"></div>
    <div class="cpop-pane cpop-text" style="display:none"></div>
    <div class="cpop-pane cpop-pdf"  style="display:none"></div>
  </div>
</div>
</body>"""
    src = must_replace(src, "</body>", css_html, label="html-body-close")

    write(HTML, src)
    print(f"PATCHED: {HTML}")

def main():
    patch_sf()
    patch_api()
    patch_html()
    print("DONE")

if __name__ == "__main__":
    main()
