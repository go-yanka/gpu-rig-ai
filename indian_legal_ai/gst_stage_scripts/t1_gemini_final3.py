#!/usr/bin/env python3
"""Use Gemini's URLs for the 3 still-missing: MV, POCSO, HSA (full)."""
import requests, urllib3, fitz, os
urllib3.disable_warnings()

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
STAGE = "/opt/indian-legal-ai/gst_stage"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8"}

JOBS = [
    ("mv_1988", "motor vehicles act", "t1_other_bare_acts/mv_act_1988.pdf", [
        "https://web.archive.org/web/2023/https://legislative.gov.in/sites/default/files/A1988-59.pdf",
        "https://web.archive.org/web/2024*/legislative.gov.in/sites/default/files/A1988-59.pdf",
        "https://web.archive.org/web/2023if_/https://legislative.gov.in/sites/default/files/A1988-59.pdf",
        "https://prsindia.org/files/bills_acts/acts_parliament/1988/motor-vehicles-act-1988.pdf",
        "https://morth.nic.in/sites/default/files/Motor%20Vehicles%20Act%2C%201988.pdf",
    ]),
    ("pocso_2012", "protection of children from sexual offences", "t1_other_bare_acts/pocso_2012.pdf", [
        "https://cdnbbsr.s3waas.gov.in/s3ac70f62ecaf8111d5a9fca8847934477/uploads/2025/09/202509101808897426.pdf",
        "https://web.archive.org/web/2023/https://legislative.gov.in/sites/default/files/A2012-32.pdf",
        "https://prsindia.org/files/bills_acts/acts_parliament/2012/pocso-act-2012.pdf",
    ]),
    ("hsa_1956", "hindu succession act", "t1_other_bare_acts/hsa_1956_full.pdf", [
        "https://web.archive.org/web/2023/https://legislative.gov.in/sites/default/files/A1956-30.pdf",
        "https://prsindia.org/files/bills_acts/acts_parliament/1956/hindu-succession-act-1956.pdf",
        "https://cdnbbsr.s3waas.gov.in/s32d579dc29360d8bbfbb4aa541de5afa9/uploads/2025/02/A1956-30.pdf",
    ]),
]

BAD = ["amendment bill","written answer","jstor","research paper","papers laid",
       "digital copy of a book","cornell university"]

for label, kw, rel, urls in JOBS:
    ok = False
    for u in urls:
        try:
            r = requests.get(u, headers=HDRS, timeout=40, verify=False,
                             allow_redirects=True)
        except Exception as e:
            print(f"  [{label}] EXC {type(e).__name__}: {u[:90]}"); continue
        if r.status_code != 200:
            print(f"  [{label}] HTTP {r.status_code}: {u[:90]}"); continue
        if r.content[:4] != b"%PDF":
            print(f"  [{label}] NOT-PDF (starts {r.content[:20]!r}): {u[:90]}"); continue
        try:
            d = fitz.open(stream=r.content, filetype="pdf")
            pg = d.page_count
            t = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
            d.close()
        except Exception as e:
            print(f"  [{label}] PARSE-ERR: {e}"); continue
        tl = t.lower()[:8000]
        title = t[:160].replace("\n"," ").strip()
        if kw not in tl:
            print(f"  [{label}] KW MISS: title={title[:90]}"); continue
        if any(b in tl for b in BAD):
            print(f"  [{label}] BAD-MARKER"); continue
        dest = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f: f.write(r.content)
        print(f"  [{label}] SAVED {len(r.content)//1024}KB p={pg} -> {rel}")
        print(f"         via: {u[:100]}")
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
