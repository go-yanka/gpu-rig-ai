#!/usr/bin/env python3
"""Fetch last 4: SARFAESI, MV Act, POCSO, HSA (full) from indiacode bitstream + mirrors."""
import requests, urllib3, fitz, os
urllib3.disable_warnings()

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
STAGE = "/opt/indian-legal-ai/gst_stage"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8"}

# Multiple candidates per act - try indiacode bitstream first (pattern proven in batch 1)
JOBS = [
    ("sarfaesi_2002", "securitisation", "t1_other_bare_acts/sarfaesi_2002.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2006/1/200254.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2006/1/A2002-54.pdf",
        "https://financialservices.gov.in/beta/sites/default/files/SARFAESI%20Act%202002.pdf",
        "https://home.wb.gov.in/public/assets/frontend/pdf/securitisation_act.pdf",
    ]),
    ("mv_1988", "motor vehicles act", "t1_other_bare_acts/mv_act_1988.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1798/1/198859.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1798/1/A1988-59.pdf",
        "https://morth.nic.in/sites/default/files/MV_Act_English.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1798/3/A1988-59.pdf",
    ]),
    ("pocso_2012", "protection of children from sexual offences", "t1_other_bare_acts/pocso_2012.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2079/1/A2012-32.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2079/1/201232.pdf",
        "https://wcd.nic.in/sites/default/files/POCSO%20Act%2C%202012.pdf",
        "https://ncpcr.gov.in/showfile.php?lang=1&level=1&&sublinkid=1450&lid=1549",
    ]),
    ("hsa_1956", "hindu succession act", "t1_other_bare_acts/hsa_1956.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1645/1/A1956-30.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1645/1/195630.pdf",
        "https://highcourtchd.gov.in/hclscc/subpages/pdf_files/5.pdf",
    ]),
]

BAD = ["amendment bill","written answer","jstor","research paper","papers laid",
       "digital copy of a book","cornell university"]

for label, kw, rel, urls in JOBS:
    ok = False
    for u in urls:
        try:
            r = requests.get(u, headers=HDRS, timeout=25, verify=False,
                             allow_redirects=True)
        except Exception as e:
            print(f"  [{label}] EXC {type(e).__name__}: {u[:80]}")
            continue
        if r.status_code != 200:
            print(f"  [{label}] HTTP {r.status_code}: {u[:80]}")
            continue
        if r.content[:4] != b"%PDF":
            print(f"  [{label}] NOT-PDF (starts {r.content[:20]!r}): {u[:80]}")
            continue
        try:
            d = fitz.open(stream=r.content, filetype="pdf")
            pg = d.page_count
            t = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
            d.close()
        except Exception as e:
            print(f"  [{label}] PARSE-ERR: {e}")
            continue
        tl = t.lower()[:8000]
        title = t[:160].replace("\n"," ").strip()
        if kw not in tl:
            print(f"  [{label}] KW MISS: title={title[:90]}")
            continue
        if any(b in tl for b in BAD):
            print(f"  [{label}] BAD-MARKER")
            continue
        dest = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # overwrite only if new >= existing (HSA stub is 84KB)
        if os.path.exists(dest) and os.path.getsize(dest) > len(r.content):
            print(f"  [{label}] existing larger, keeping")
            ok = True; break
        with open(dest, "wb") as f: f.write(r.content)
        print(f"  [{label}] SAVED {len(r.content)//1024}KB p={pg} -> {rel}")
        print(f"         title: {title[:110]}")
        ok = True; break
    if not ok:
        print(f"  [{label}] ALL FAILED")

print("\n=== Final staging ===")
tp=0;tk=0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"): continue
    pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
    kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
    if pdfs:
        tp+=len(pdfs); tk+=kb
        print(f"  {d:<32} {len(pdfs):>3} pdfs  {kb:>7} KB")
print(f"TOTAL: {tp} pdfs, {tk/1024:.1f} MB")
