#!/usr/bin/env python3
"""T1 CLOSURE: Mediation Act 2023 + 5 concordance/bridge PDFs.
Inline QC on every download."""
import os, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9"}

# (label, expected_kw, dest_rel, [urls], allow_amendment_markers)
JOBS = [
    ("mediation_2023", "mediation act, 2023", "t1_adr/mediation_2023.pdf", [
        "https://egazette.gov.in/WriteReadData/2023/248775.pdf",
    ], False),
    ("bridge_ipc_bns", "bharatiya nyaya sanhita", "t1_bridges/ipc_to_bns_concordance.pdf", [
        "https://uppolice.gov.in/site/writereaddata/siteContent/Three%20New%20Major%20Acts/202406281710564823BNS_IPC_Comparative.pdf",
    ], True),
    ("bridge_crpc_bnss", "bharatiya nagarik suraksha sanhita", "t1_bridges/crpc_to_bnss_concordance.pdf", [
        "https://uppolice.gov.in/site/writereaddata/siteContent/Three%20New%20Major%20Acts/202407031502194192BNSS2023-20-45.pdf",
    ], True),
    ("bridge_evidence_bsa", "bharatiya sakshya", "t1_bridges/evidence_to_bsa_concordance.pdf", [
        "https://uppolice.gov.in/site/writereaddata/siteContent/Three%20New%20Major%20Acts/202407031507276576BSA2023-9-25.pdf",
    ], True),
    ("bprd_bns_handbook", "bharatiya nyaya sanhita", "t1_bridges/bprd_bns_handbook.pdf", [
        "https://bprd.nic.in/uploads/pdf/BNS%20Book_After%20Correction.pdf",
    ], True),
    ("bprd_bsa_compare", "bharatiya sakshya", "t1_bridges/bprd_bsa_compare.pdf", [
        "https://bprd.nic.in/uploads/pdf/Comparison%20Summary%20BSA%20to%20IEA.pdf",
    ], True),
]

BAD = ["written answer", "jstor", "research paper", "papers laid",
       "digital copy of a book", "cornell university", "all india christian council",
       "leave of absence", "google books"]


def qc(content, kw, allow_amend=False):
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
    first = "".join(d.load_page(i).get_text() for i in range(min(5, pg)))
    d.close()
    tl = first.lower()[:15000]
    title = first[:220].replace("\n", " ").strip()
    if kw.lower() not in tl:
        return False, f"kw-miss '{kw}'", {"pg": pg, "cpp": cpp, "title": title[:140]}
    bad_list = BAD if allow_amend else (BAD + ["amendment bill"])
    for b in bad_list:
        if b in tl: return False, f"bad '{b}'", {"pg": pg}
    return True, "ok", {"pg": pg, "cpp": cpp, "size_kb": len(content)//1024, "title": title[:160]}


def main():
    sess = requests.Session()
    for warm in ["https://egazette.gov.in/", "https://uppolice.gov.in/", "https://bprd.nic.in/"]:
        try: sess.get(warm, headers={"User-Agent": UA}, timeout=15, verify=False)
        except: pass

    saved = 0; failed = []
    for label, kw, rel, urls, allow_amend in JOBS:
        print(f"\n[{label}]")
        got = False
        for u in urls:
            print(f"  try {u[-110:]}")
            try:
                r = sess.get(u, headers=HDRS, timeout=120, verify=False, allow_redirects=True)
            except Exception as e:
                print(f"    EXC {type(e).__name__}: {str(e)[:120]}"); continue
            if r.status_code != 200:
                print(f"    http-{r.status_code}"); continue
            ok, reason, meta = qc(r.content, kw, allow_amend=allow_amend)
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
    print(f"T1 CLOSURE: {saved}/{len(JOBS)} saved")
    if failed:
        print("FAILED:", ", ".join(failed))

    print("\n=== FINAL T1 CORPUS ===")
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
