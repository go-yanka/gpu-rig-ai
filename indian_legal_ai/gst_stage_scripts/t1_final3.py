#!/usr/bin/env python3
"""Final 3 T1 gaps: Muslim Dissolution 1939, Indian Divorce 1869, BNSS 2023 clean.
Inline QC on every download. No deferred audit."""
import os, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9"}

JOBS = [
    ("muslim_dissolution_1939", "dissolution of muslim marriages", "t1_personal_law/muslim_dissolution_1939.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2404/1/193908.pdf",
        "https://highcourtchd.gov.in/hclscc/subpages/pdf_files/12.pdf",
    ]),
    ("indian_divorce_1869", "divorce act", "t1_personal_law/indian_divorce_1869.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2280/1/A1869-04.pdf",
    ]),
    ("bnss_2023_clean", "bharatiya nagarik suraksha sanhita", "t1_criminal_codes_2023/BNSS_2023.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/21544/1/the_bharatiya_nagarik_suraksha_sanhita%2C_2023.pdf",
    ]),
]

BAD = ["amendment bill", "written answer", "jstor", "research paper", "papers laid",
       "digital copy of a book", "cornell university", "all india christian council",
       "leave of absence", "google books"]


def qc(content, kw):
    if len(content) < 1024: return False, f"too-small ({len(content)}B)", {}
    if content[:4] != b"%PDF": return False, f"not-pdf ({content[:20]!r})", {}
    try:
        d = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        return False, f"parse-err: {e}", {}
    pg = d.page_count
    if pg < 3:
        d.close(); return False, f"stub ({pg}p)", {"pg": pg}
    tot = sum(len(d.load_page(i).get_text()) for i in range(pg))
    cpp = tot // max(pg, 1)
    if cpp < 200:
        d.close(); return False, f"scanned ({cpp}c/p)", {"pg": pg, "cpp": cpp}
    first = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
    d.close()
    tl = first.lower()[:8000]
    title = first[:200].replace("\n", " ").strip()
    if kw.lower() not in tl:
        return False, f"kw-miss '{kw}'", {"pg": pg, "cpp": cpp, "title": title[:100]}
    for b in BAD:
        if b in tl: return False, f"bad '{b}'", {"pg": pg}
    return True, "ok", {"pg": pg, "cpp": cpp, "size_kb": len(content)//1024, "title": title[:140]}


def main():
    sess = requests.Session()
    try: sess.get("https://www.indiacode.nic.in/", headers={"User-Agent": UA}, timeout=15, verify=False)
    except: pass

    saved = 0; failed = []
    for label, kw, rel, urls in JOBS:
        print(f"\n[{label}]")
        got = False
        for u in urls:
            print(f"  try {u[-100:]}")
            try:
                r = sess.get(u, headers=HDRS, timeout=60, verify=False, allow_redirects=True)
            except Exception as e:
                print(f"    EXC {type(e).__name__}: {e}"); continue
            if r.status_code != 200:
                print(f"    http-{r.status_code}"); continue
            ok, reason, meta = qc(r.content, kw)
            if not ok:
                print(f"    REJECT: {reason} {meta}"); continue
            dest = os.path.join(STAGE, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f: f.write(r.content)
            print(f"    SAVED {meta['size_kb']}KB {meta['pg']}p c/p={meta['cpp']}")
            print(f"    title: {meta['title']}")
            saved += 1; got = True; break
        if not got: failed.append(label)

    print("\n" + "=" * 70)
    print(f"FINAL3: {saved}/{len(JOBS)} saved")
    if failed:
        print("FAILED:", ", ".join(failed))

    print("\n=== FINAL T1 STAGING ===")
    tp = 0; tk = 0
    for d in sorted(os.listdir(STAGE)):
        full = os.path.join(STAGE, d)
        if not os.path.isdir(full) or not d.startswith("t1_"): continue
        pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
        kb = sum(os.path.getsize(os.path.join(full, f)) for f in pdfs) // 1024
        if pdfs:
            tp += len(pdfs); tk += kb
            print(f"  {d:<32} {len(pdfs):>3} pdfs  {kb:>7} KB")
    print(f"TOTAL: {tp} pdfs, {tk/1024:.1f} MB")


if __name__ == "__main__":
    main()
