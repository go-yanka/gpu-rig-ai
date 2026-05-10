#!/usr/bin/env python3
"""T1.5 harvest — 10 URLs from Grok.
Inline QC on every download. Proper categorization.
Finance Acts go in t1_finance_acts (amendment deltas)."""
import os, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9"}

JOBS = [
    # (label, kw, dest_rel, [urls])
    ("income_tax_2025", "income-tax act, 2025", "t1_income_tax/income_tax_act_2025.pdf", [
        "https://www.incometaxindia.gov.in/documents/d/guest/income_tax_act_2025_as_amended_by_fa_act_2026-pdf",
    ]),
    ("posh_2013", "sexual harassment of women at workplace", "t1_workplace/posh_2013.pdf", [
        "https://doe.gov.in/files/inline-documents/DoE_Prevention_sexual_harassment.pdf",
    ]),
    ("rera_2016", "real estate (regulation and development)", "t1_real_estate/rera_2016.pdf", [
        "https://andamannicobar.gov.in/admin-pannel/pressupload/1774592612_THE%20REAL%20ESTATE%20(REGULATION%20AND%20DEVELOPMENT)%20ACT,%202016.pdf",
    ]),
    ("telecom_2023", "telecommunications act, 2023", "t1_telecom/telecom_act_2023.pdf", [
        "https://egazette.gov.in/WriteReadData/2023/250880.pdf",
    ]),
    ("post_office_2023", "post office act, 2023", "t1_other_bare_acts/post_office_2023.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/20064/1/A2023-43.pdf",
    ]),
    ("disaster_mgmt_2005", "disaster management act, 2005", "t1_disaster_mgmt/disaster_mgmt_2005.pdf", [
        "https://ndmindia.mha.gov.in/ndmi/images/The%20Disaster%20Management%20Act,%202005.pdf",
    ]),
    ("public_exams_2024", "public examinations", "t1_other_bare_acts/public_exams_2024.pdf", [
        "https://prsindia.org/files/bills_acts/bills_parliament/2024/The_Public_Examinations_(Prevention_of_Unfair_Means)_Act,_2024.pdf",
    ]),
    ("finance_act_2024", "finance act, 2024", "t1_finance_acts/finance_act_2024.pdf", [
        "https://egazette.gov.in/WriteReadData/2024/256436.pdf",
    ]),
    ("finance_act_2025", "finance act, 2025", "t1_finance_acts/finance_act_2025.pdf", [
        "https://egazette.gov.in/WriteReadData/2025/260786.pdf",
    ]),
    ("finance_act_2026", "finance act, 2026", "t1_finance_acts/finance_act_2026.pdf", [
        "https://www.indiabudget.gov.in/doc/Finance_Bill.pdf",
    ]),
]

BAD = ["written answer", "jstor", "research paper", "papers laid",
       "digital copy of a book", "cornell university", "all india christian council",
       "leave of absence", "google books"]
# Note: "amendment bill" removed from BAD since FA/some Acts reference amendments legitimately


def qc(content, kw, allow_bill=False):
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
    tl = first.lower()[:10000]
    title = first[:220].replace("\n", " ").strip()
    if kw.lower() not in tl:
        return False, f"kw-miss '{kw}'", {"pg": pg, "cpp": cpp, "title": title[:100]}
    bad_list = BAD if allow_bill else (BAD + ["amendment bill"])
    for b in bad_list:
        if b in tl: return False, f"bad '{b}'", {"pg": pg}
    return True, "ok", {"pg": pg, "cpp": cpp, "size_kb": len(content)//1024, "title": title[:160]}


def main():
    sess = requests.Session()
    for warm in ["https://www.indiacode.nic.in/", "https://egazette.gov.in/",
                 "https://www.incometaxindia.gov.in/"]:
        try: sess.get(warm, headers={"User-Agent": UA}, timeout=15, verify=False)
        except: pass

    saved = 0; failed = []
    for label, kw, rel, urls in JOBS:
        print(f"\n[{label}]")
        got = False
        # Finance Acts allow "amendment bill" text (they ARE amendments)
        allow_bill = label.startswith("finance_act")
        for u in urls:
            print(f"  try {u[-110:]}")
            try:
                r = sess.get(u, headers=HDRS, timeout=90, verify=False, allow_redirects=True)
            except Exception as e:
                print(f"    EXC {type(e).__name__}: {str(e)[:100]}"); continue
            if r.status_code != 200:
                print(f"    http-{r.status_code}"); continue
            ok, reason, meta = qc(r.content, kw, allow_bill=allow_bill)
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
    print(f"T1.5 FINAL10: {saved}/{len(JOBS)} saved")
    if failed:
        print("FAILED:", ", ".join(failed))

    print("\n=== FINAL CORPUS (T1 + T1.5) ===")
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
