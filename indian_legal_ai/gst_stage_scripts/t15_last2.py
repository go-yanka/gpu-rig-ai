#!/usr/bin/env python3
"""T1.5 final 2 items + drop Constitution 2020.
Inline QC enforced on every download."""
import os, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9"}

JOBS = [
    ("finance_act_2026", "finance act, 2026", "t1_finance_acts/finance_act_2026.pdf", [
        "https://egazette.gov.in/WriteReadData/2026/271439.pdf",
    ], True),
    ("public_exams_2024", "public examinations", "t1_other_bare_acts/public_exams_2024.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/20100/1/A2024-01.pdf",
    ], False),
]

BAD = ["written answer", "jstor", "research paper", "papers laid",
       "digital copy of a book", "cornell university", "all india christian council",
       "leave of absence", "google books"]


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
    first = "".join(d.load_page(i).get_text() for i in range(min(5, pg)))
    d.close()
    tl = first.lower()[:12000]
    title = first[:220].replace("\n", " ").strip()
    if kw.lower() not in tl:
        return False, f"kw-miss '{kw}'", {"pg": pg, "cpp": cpp, "title": title[:140]}
    bad_list = BAD if allow_bill else (BAD + ["amendment bill"])
    for b in bad_list:
        if b in tl: return False, f"bad '{b}'", {"pg": pg}
    return True, "ok", {"pg": pg, "cpp": cpp, "size_kb": len(content)//1024, "title": title[:160]}


def main():
    sess = requests.Session()
    for warm in ["https://www.indiacode.nic.in/", "https://egazette.gov.in/"]:
        try: sess.get(warm, headers={"User-Agent": UA}, timeout=15, verify=False)
        except: pass

    print("=" * 70)
    print("STAGE 1: Last 2 T1.5 items")
    print("=" * 70)
    saved = 0; failed = []
    for label, kw, rel, urls, allow_bill in JOBS:
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
    print(f"\nSaved: {saved}/{len(JOBS)}" + (f" · failed: {failed}" if failed else ""))

    print("\n" + "=" * 70)
    print("STAGE 2: Drop Constitution 2020 (per Grok review)")
    print("=" * 70)
    const2020 = os.path.join(STAGE, "t1_constitution/constitution_2020.pdf")
    if os.path.exists(const2020):
        sz = os.path.getsize(const2020) // 1024
        os.remove(const2020)
        print(f"  removed constitution_2020.pdf ({sz}KB)")
    else:
        print("  constitution_2020.pdf not found (already removed?)")
    const_dir = os.path.join(STAGE, "t1_constitution")
    for f in sorted(os.listdir(const_dir)):
        print(f"  remaining: {f}")

    print("\n" + "=" * 70)
    print("FINAL T1 + T1.5 CORPUS")
    print("=" * 70)
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
