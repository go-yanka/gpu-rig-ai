#!/usr/bin/env python3
"""Restore IGST/UTGST Bill-as-enacted text, hunt real IBC."""
import os, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8"}

# IGST/UTGST: original CBIC bill-e.pdf IS the enacted text (no changes between bill and act)
# IBC: try many mirrors
JOBS = [
    ("igst", "integrated goods and services tax", "t1_gst_circulars/IGST_Act_2017.pdf", [
        "https://cbic-gst.gov.in/aces/Documents/IGST-bill-e.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2263/1/a2017-13.pdf",
    ]),
    ("utgst", "union territory goods", "t1_gst_circulars/UTGST_Act_2017.pdf", [
        "https://cbic-gst.gov.in/aces/Documents/UTGST-bill-e.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2262/1/a2017-14.pdf",
    ]),
    ("ibc", "insolvency and bankruptcy code", "t1_ibc/IBC_Code_2016.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2154/1/AA2016-31.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2154/1/a201631.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2154/3/a2016-31.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2154/2/a2016-31.pdf",
        "https://ibbi.gov.in/webadmin/pdf/legalframwork/2017/Jun/The%20Insolvency%20and%20Bankruptcy%20Code,%202016_2017-06-26%2023:47:58.pdf",
        "https://www.mca.gov.in/Ministry/pdf/TheInsolvencyandBankruptcyofIndia.pdf",
        "https://prsindia.org/files/bills_acts/acts_parliament/2016/The-Insolvency-and-Bankruptcy-Code,-2016.pdf",
    ]),
]

BAD = ["amendment bill","written answer","jstor","research paper","papers laid",
       "digital copy of a book","cornell university"]

for label, kw, rel, urls in JOBS:
    ok = False
    for u in urls:
        try:
            r = requests.get(u, headers=HDRS, timeout=40, verify=False, allow_redirects=True)
        except Exception as e:
            print(f"  [{label}] EXC {type(e).__name__}: {u[:90]}"); continue
        if r.status_code != 200:
            print(f"  [{label}] HTTP {r.status_code}: {u[:90]}"); continue
        if r.content[:4] != b"%PDF":
            print(f"  [{label}] NOT-PDF (starts {r.content[:20]!r}): {u[:90]}"); continue
        try:
            d = fitz.open(stream=r.content, filetype="pdf")
            pg = d.page_count
            tot = sum(len(d.load_page(i).get_text()) for i in range(pg))
            cpp = tot // max(pg,1)
            t = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
            d.close()
        except Exception as e:
            print(f"  [{label}] PARSE-ERR: {e}"); continue
        tl = t.lower()[:8000]
        title = t[:160].replace("\n"," ").strip()
        if kw not in tl:
            print(f"  [{label}] KW MISS '{kw}': {title[:70]} ({pg}p {cpp}c/p)"); continue
        if any(b in tl for b in BAD):
            print(f"  [{label}] BAD-MARKER"); continue
        # need minimum pages and chars/page (reject 1-page stubs)
        if pg < 10:
            print(f"  [{label}] TOO-FEW-PAGES ({pg}p) — skipping {u[:80]}"); continue
        if cpp < 200:
            print(f"  [{label}] SCANNED ({cpp}c/p) — skipping"); continue
        dest = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f: f.write(r.content)
        print(f"  [{label}] SAVED {len(r.content)//1024}KB p={pg} c/p={cpp} -> {rel}")
        print(f"         via {u[:95]}")
        print(f"         title: {title[:100]}")
        ok = True; break
    if not ok:
        print(f"  [{label}] ALL FAILED")

print("\n=== FINAL ===")
tp=0;tk=0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"): continue
    pdfs = sorted([f for f in os.listdir(full) if f.lower().endswith(".pdf")])
    kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
    if pdfs:
        tp+=len(pdfs); tk+=kb
        print(f"  {d:<32} {len(pdfs):>3} pdfs  {kb:>7} KB")
print(f"TOTAL: {tp} pdfs, {tk/1024:.1f} MB")
