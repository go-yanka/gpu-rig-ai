#!/usr/bin/env python3
"""Final 14 missing T1 acts via Grok's verified URLs.
Multiple candidates per act, inline QC before save."""
import os, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HDRS = {
    "User-Agent": UA,
    "Accept": "application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

IC = "https://www.indiacode.nic.in/bitstream/123456789"

JOBS = [
    ("special_marriage_1954", "special marriage act", "t1_personal_law/special_marriage_1954.pdf", [
        "https://sclsc.gov.in/theme/front/pdf/ACTS%20FINAL/THE%20SPECIAL%20MARRIAGE%20ACT,%201954.pdf",
        f"{IC}/15317/1/special_marriage_act_1954.pdf",
        f"{IC}/15317/1/the_special_marriage_act%2C_1954.pdf",
    ]),
    ("muslim_dissolution_1939", "dissolution of muslim marriages", "t1_personal_law/muslim_dissolution_1939.pdf", [
        f"{IC}/2404/1/A1939-08.pdf",
        f"{IC}/2404/1/a1939-08.pdf",
        f"{IC}/15334/1/the_dissolution_of_muslim_marriages_act%2C_1939.pdf",
        f"{IC}/15319/1/the_dissolution_of_muslim_marriages_act%2C_1939.pdf",
    ]),
    ("indian_divorce_1869", "divorce act", "t1_personal_law/indian_divorce_1869.pdf", [
        f"{IC}/2253/1/a1869-04.pdf",
        f"{IC}/2253/1/A1869-04.pdf",
        f"{IC}/15322/1/the_divorce_act%2C_1869.pdf",
        f"{IC}/15333/1/the_divorce_act%2C_1869.pdf",
    ]),
    ("indian_trusts_1882", "indian trusts act", "t1_other_bare_acts/indian_trusts_1882.pdf", [
        f"{IC}/2327/3/A1882-02.pdf",
        f"{IC}/2327/1/A1882-02.pdf",
        f"{IC}/2327/2/A1882-02.pdf",
    ]),
    ("benami_1988", "benami", "t1_banking/benami_1988.pdf", [
        f"{IC}/15415/1/the_prohibition_of_benami_property_transactions_act%2C_1988.pdf",
    ]),
    ("trade_marks_1999", "trade marks act", "t1_ip_acts/trade_marks_1999.pdf", [
        f"{IC}/15427/1/the_trade_marks_act%2C_1999.pdf",
    ]),
    ("pc_act_1988", "prevention of corruption", "t1_criminal_special/pc_act_1988.pdf", [
        f"{IC}/1558/1/A1988-49.pdf",
    ]),
    ("ndps_1985", "narcotic drugs", "t1_criminal_special/ndps_1985.pdf", [
        f"{IC}/18974/1/narcotic-drugs-and-psychotropic-substances-act-1985.pdf",
    ]),
    ("environment_1986", "environment", "t1_environment/environment_1986.pdf", [
        f"{IC}/6196/1/the_environment_protection_act%2C1986.pdf",
        f"{IC}/6196/1/the_environment_protection_act,1986.pdf",
    ]),
    ("wildlife_1972", "wild life", "t1_environment/wildlife_1972.pdf", [
        f"{IC}/6198/1/the_wild_life_(protection)_act,_1972.pdf",
        f"{IC}/6198/1/the_wild_life_%28protection%29_act%2C_1972.pdf",
    ]),
    ("industrial_disputes_1947", "industrial disputes act", "t1_pre_codified_labour/industrial_disputes_1947.pdf", [
        f"{IC}/20352/1/the_industrial_disputes_act.pdf",
    ]),
    ("factories_1948", "factories act", "t1_pre_codified_labour/factories_1948.pdf", [
        f"{IC}/15097/1/factory_acta1948-63.pdf",
    ]),
    ("min_wages_1948", "minimum wages act", "t1_pre_codified_labour/minimum_wages_1948.pdf", [
        f"{IC}/20357/1/a1948-011.pdf",
    ]),
    ("gratuity_1972", "payment of gratuity", "t1_pre_codified_labour/gratuity_1972.pdf", [
        f"{IC}/15318/1/payment-of-gratuity-act-1972.pdf",
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
    if pg < 4:
        d.close(); return False, f"stub ({pg}p)", {"pg": pg}
    tot = sum(len(d.load_page(i).get_text()) for i in range(pg))
    cpp = tot // max(pg, 1)
    if cpp < 200:
        d.close(); return False, f"scanned ({cpp}c/p)", {"pg": pg, "cpp": cpp}
    first = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
    d.close()
    tl = first.lower()[:8000]
    title = first[:160].replace("\n", " ").strip()
    if kw.lower() not in tl:
        return False, f"kw-miss '{kw}'", {"pg": pg, "cpp": cpp, "title": title[:80]}
    for b in BAD:
        if b in tl: return False, f"bad '{b}'", {"pg": pg}
    return True, "ok", {"pg": pg, "cpp": cpp, "size_kb": len(content) // 1024, "title": title[:100]}


def main():
    sess = requests.Session()
    try: sess.get("https://www.indiacode.nic.in/", headers={"User-Agent": UA}, timeout=15, verify=False)
    except: pass

    saved = 0; failed = []
    for label, kw, rel, urls in JOBS:
        print(f"\n[{label}]")
        got = False
        for u in urls:
            print(f"  try {u[-95:]}")
            try:
                r = sess.get(u, headers=HDRS, timeout=60, verify=False, allow_redirects=True)
            except Exception as e:
                print(f"    EXC {type(e).__name__}"); continue
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
    print(f"GROK FINAL14: {saved}/{len(JOBS)} saved")
    if failed:
        print("\nFAILED:")
        for f in failed: print(f"  - {f}")

    print("\n=== FINAL STAGING ===")
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
