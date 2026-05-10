#!/usr/bin/env python3
"""Fetch 24 missing acts via Gemini-supplied indiacode handle IDs.
Strategy: construct /bitstream/123456789/<HANDLE>/N/<any>.pdf — we don't know
the filename, so probe HTML of /handle/<HANDLE> page to extract the real
bitstream href, then GET it. Inline QC before save.

indiacode's JS challenge applies to HTML pages but NOT to direct bitstream
GETs once we have the URL — that's been our consistent experience with
IBC (handle 15479), IGST (11909), UTGST (11911).
"""
import os, re, sys, time, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HDRS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# (label, handle, kw, dest_rel, fallback_urls)
JOBS = [
    ("general_clauses_1897",    "2328",  "general clauses act",            "t1_interpretation/general_clauses_1897.pdf", []),
    ("indian_succession_1925",  "2385",  "indian succession act",          "t1_personal_law/indian_succession_1925.pdf", []),
    ("special_marriage_1954",   "2165",  "special marriage act",           "t1_personal_law/special_marriage_1954.pdf", []),
    ("muslim_dissolution_1939", "2404",  "dissolution of muslim marriages","t1_personal_law/muslim_dissolution_1939.pdf", []),
    ("shariat_1937",            "2303",  "shariat",                        "t1_personal_law/shariat_application_1937.pdf", []),
    ("indian_divorce_1869",     "2253",  "divorce act",                    "t1_personal_law/indian_divorce_1869.pdf", []),
    ("indian_trusts_1882",      "2348",  "indian trusts act",              "t1_other_bare_acts/indian_trusts_1882.pdf", []),
    ("indian_easements_1882",   "2349",  "indian easements act",           "t1_other_bare_acts/indian_easements_1882.pdf", []),
    ("insurance_1938",          "2304",  "insurance act",                  "t1_banking/insurance_act_1938.pdf", []),
    ("benami_1988",             "1921",  "benami",                         "t1_banking/benami_1988.pdf", []),
    ("customs_1962",            "2475",  "customs act",                    "t1_customs_excise/customs_act_1962.pdf", []),
    ("central_excise_1944",     "19238", "central excise act",             "t1_customs_excise/central_excise_1944.pdf", []),
    ("trade_marks_1999",        "1992",  "trade marks act",                "t1_ip_acts/trade_marks_1999.pdf",
        ["https://ipindia.gov.in/writereaddata/Portal/IPOAct/1_31_1_trade-marks-act-1999.pdf"]),
    ("pc_act_1988",             "1922",  "prevention of corruption",       "t1_criminal_special/pc_act_1988.pdf", []),
    ("ndps_1985",               "1794",  "narcotic drugs",                 "t1_criminal_special/ndps_1985.pdf", []),
    ("arms_1959",               "1398",  "arms act",                       "t1_criminal_special/arms_1959.pdf", []),
    ("dpdp_2023",               "22037", "digital personal data protection","t1_data_privacy/dpdp_2023.pdf",
        ["https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf"]),
    ("competition_2002",        "2010",  "competition act",                "t1_other_bare_acts/competition_2002.pdf", []),
    ("environment_1986",        "13656", "environment",                    "t1_environment/environment_1986.pdf", []),
    ("wildlife_1972",           "1748",  "wild life",                      "t1_environment/wildlife_1972.pdf",
        ["http://moef.gov.in/wp-content/uploads/2017/06/wildlife1.pdf"]),
    ("industrial_disputes_1947","2436",  "industrial disputes act",        "t1_pre_codified_labour/industrial_disputes_1947.pdf", []),
    ("factories_1948",          "2437",  "factories act",                  "t1_pre_codified_labour/factories_1948.pdf", []),
    ("min_wages_1948",          "2321",  "minimum wages act",              "t1_pre_codified_labour/minimum_wages_1948.pdf", []),
    ("gratuity_1972",           "1394",  "payment of gratuity",            "t1_pre_codified_labour/gratuity_1972.pdf", []),
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


def probe_bitstreams(session, handle):
    """Visit /handle/<handle> and extract bitstream PDF URLs. Returns list."""
    url = f"https://www.indiacode.nic.in/handle/123456789/{handle}"
    try:
        r = session.get(url, headers=HDRS, timeout=30, verify=False, allow_redirects=True)
    except Exception as e:
        return [], f"handle-exc {type(e).__name__}"
    if r.status_code != 200:
        return [], f"handle-{r.status_code}"
    # find PDF bitstreams
    hrefs = re.findall(r'href="(/bitstream/123456789/[^"]+?\.pdf)"', r.text, flags=re.I)
    urls = ["https://www.indiacode.nic.in" + h for h in hrefs]
    # dedupe preserve order
    seen = set(); out = []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out, f"found-{len(out)}"


def try_download(session, url, kw):
    try:
        r = session.get(url, headers={**HDRS, "Accept": "application/pdf,*/*;q=0.8"},
                        timeout=60, verify=False, allow_redirects=True)
    except Exception as e:
        return None, f"exc {type(e).__name__}"
    if r.status_code != 200:
        return None, f"http-{r.status_code}"
    ok, reason, meta = qc(r.content, kw)
    if not ok:
        return None, reason
    return r.content, meta


def main():
    sess = requests.Session()
    # Warm cookies via homepage
    try:
        sess.get("https://www.indiacode.nic.in/", headers=HDRS, timeout=20, verify=False)
    except: pass

    saved = 0
    failed = []

    for label, handle, kw, rel, fallbacks in JOBS:
        print(f"\n[{label}] handle={handle}")
        # Step 1: probe bitstream URLs from handle page
        bitstreams, note = probe_bitstreams(sess, handle)
        print(f"  probe: {note}")
        # Step 2: try candidate URLs (probed + /1/ /2/ /3/ guesses by act-year)
        candidates = list(bitstreams)
        # add guess patterns as a safety net
        for seq in (1, 2, 3):
            candidates.append(f"https://www.indiacode.nic.in/bitstream/123456789/{handle}/{seq}/a.pdf")
        candidates.extend(fallbacks)

        got = False
        for u in candidates:
            print(f"  try {u[-80:]}")
            content, info = try_download(sess, u, kw)
            if content is None:
                print(f"    REJECT: {info}")
                continue
            dest = os.path.join(STAGE, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(content)
            print(f"    SAVED {info['size_kb']}KB {info['pg']}p c/p={info['cpp']}")
            print(f"    title: {info['title']}")
            saved += 1; got = True
            break
        if not got:
            failed.append(label)
            print(f"  FAILED all candidates")
        time.sleep(0.5)

    print("\n" + "=" * 70)
    print(f"GEMINI HANDLES: {saved}/{len(JOBS)} saved")
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
