#!/usr/bin/env python3
# Multi-source recovery v2 - with dedup + relevance check.
import os, sys, json, sqlite3, hashlib, base64, re, time, threading
import urllib.parse as up
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, urllib3
urllib3.disable_warnings()
sys.dont_write_bytecode=True

ROOT='/opt/indian-legal-ai/data/scraped/cbic'
DB=f'{ROOT}/_manifest.sqlite'

LOCK=threading.Lock()
SEEN_SHA=set()

def mk_session():
    s=requests.Session(); s.verify=False
    s.headers.update({'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0'})
    return s

def is_pdf(b): return b and len(b)>1024 and b[:4]==b'%PDF'

def decode(body, ct):
    if ct and ct.startswith('application/json'):
        try:
            j=json.loads(body); d=j.get('data')
            if d: return base64.b64decode(d)
        except: pass
    if body[:4]==b'%PDF': return body
    return None

def _norm(s):
    return re.sub(r'[^a-z0-9]+',' ',(s or '').lower()).strip()

def relevance_ok(doc, pdf_bytes):
    title=_norm(doc.get('title') or '')
    num=_norm(doc.get('number') or '')
    if not title and not num: return True
    import subprocess
    try:
        p=subprocess.Popen(['pdftotext','-q','-l','3','-','-'],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out,_=p.communicate(pdf_bytes, timeout=20)
        txt=_norm(out.decode('utf-8','ignore'))
    except Exception:
        try: txt=_norm(pdf_bytes[:40000].decode('latin-1','ignore'))
        except: txt=''
    words=[w for w in title.split() if len(w)>=5]
    anchors=sorted(set(words), key=len, reverse=True)[:4]
    if num and len(num)>=3: anchors.append(num)
    if not anchors: return True
    hits=sum(1 for a in anchors if a in txt)
    return hits >= max(1, len(anchors)//2)

def try_wayback(s, url, doc):
    if not url: return None,None
    try:
        r=s.get('https://archive.org/wayback/available', params={'url':url}, timeout=15)
        j=r.json()
        snap=j.get('archived_snapshots',{}).get('closest')
        if not snap or not snap.get('available'): return None,None
        surl=snap['url']
        if not surl.startswith('http'): surl='https:'+surl
        surl=re.sub(r'/web/(\d+)/', r'/web/\1id_/', surl)
        r2=s.get(surl, timeout=40)
        p=decode(r2.content, r2.headers.get('content-type',''))
        if is_pdf(p) and relevance_ok(doc, p): return p, surl
    except: pass
    return None,None

def try_indiacode(s, doc):
    if doc['subcategory'] not in ('rules','regulations','allied_acts'): return None,None
    title=(doc['title'] or '').strip()
    if len(title)<6: return None,None
    try:
        r=s.get('https://www.indiacode.nic.in/simple-search',
                params={'query':title[:100],'Submit':'Search'}, timeout=20)
        for m in re.finditer(r'href="([^"]+\.pdf)"', r.text):
            u=m.group(1)
            if not u.startswith('http'): u='https://www.indiacode.nic.in'+u
            r2=s.get(u,timeout=30)
            if is_pdf(r2.content) and relevance_ok(doc, r2.content):
                return r2.content, u
    except: pass
    return None,None

def try_ddg_pdf(s, doc):
    title=(doc['title'] or '').strip()
    if len(title)<10: return None,None
    q=f'"{title[:100]}" filetype:pdf'
    try:
        r=s.get('https://html.duckduckgo.com/html/', params={'q':q}, timeout=20)
        cand=re.findall(r'uddg=([^&"]+)', r.text)
        seen=set()
        for c in cand[:6]:
            try: u=up.unquote(c)
            except: continue
            if u in seen: continue
            seen.add(u)
            if '.pdf' not in u.lower(): continue
            try:
                r2=s.get(u,timeout=25)
                if is_pdf(r2.content) and relevance_ok(doc, r2.content):
                    return r2.content, u
            except: pass
    except: pass
    return None,None

def save(doc, body):
    cat=doc['category']; sub=doc['subcategory']
    d=f'{ROOT}/{cat}/{sub}/_recovered'
    os.makedirs(d, exist_ok=True)
    bn=None
    if doc.get('url_en'):
        bn=up.unquote(up.urlparse(doc['url_en']).path.rsplit('/',1)[-1])
    if not bn or not bn.lower().endswith('.pdf'):
        bn=doc['doc_id'].replace(':','_')+'.pdf'
    path=f'{d}/{bn}'
    with open(path,'wb') as f: f.write(body)
    return path, hashlib.sha256(body).hexdigest()

def process(doc):
    s=mk_session()
    strategies=[
      ('wayback_en', lambda: try_wayback(s, doc.get('url_en'), doc)),
      ('wayback_hi', lambda: try_wayback(s, doc.get('url_hi'), doc)),
      ('indiacode',  lambda: try_indiacode(s, doc)),
      ('ddg_pdf',    lambda: try_ddg_pdf(s, doc)),
    ]
    for name,fn in strategies:
        try:
            body,surl=fn()
            if not (body and is_pdf(body)): continue
            sha=hashlib.sha256(body).hexdigest()
            with LOCK:
                if sha in SEEN_SHA:
                    continue
                SEEN_SHA.add(sha)
            path,sha2=save(doc, body)
            return doc['doc_id'], name, surl, path, sha2, len(body)
        except: pass
    return doc['doc_id'], None, None, None, None, 0

def main():
    c=sqlite3.connect(DB)
    for (h,) in c.execute("SELECT sha256_en FROM docs WHERE sha256_en IS NOT NULL"):
        SEEN_SHA.add(h)
    print(f'pre-seeded SEEN_SHA with {len(SEEN_SHA)} existing hashes', flush=True)
    rows=c.execute("SELECT doc_id,category,subcategory,title,url_en,url_hi,number,date,year FROM docs WHERE path_en IS NULL AND path_hi IS NULL").fetchall()
    docs=[dict(doc_id=r[0],category=r[1],subcategory=r[2],title=r[3],url_en=r[4],url_hi=r[5],number=r[6],date=r[7],year=r[8]) for r in rows]
    print(f'processing {len(docs)} missing docs with 8 threads', flush=True)
    ok=0; fail=0; i=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(process, d): d for d in docs}
        for f in as_completed(futs):
            i+=1
            doc_id, name, surl, path, sha, sz = f.result()
            if path:
                ok+=1
                c.execute('''UPDATE docs SET path_en=?, sha256_en=?, bytes_en=?,
                               downloaded_at=datetime('now'),
                               download_source=?, source_url=?,
                               recovered_at=datetime('now'),
                               last_error='RECOVERED via '||?
                             WHERE doc_id=?''',
                          (path, sha, sz, name, surl, name, doc_id))
                c.commit()
                print(f'[{i:3d}/{len(docs)}] OK {name:12s} {sz:8d}B {path}', flush=True)
            else:
                fail+=1
                if i%10==0: print(f'[{i:3d}/{len(docs)}] progress ok={ok} fail={fail}', flush=True)
    print(f'DONE ok={ok} fail={fail}', flush=True)

if __name__=='__main__':
    main()
