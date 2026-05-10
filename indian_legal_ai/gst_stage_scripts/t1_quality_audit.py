#!/usr/bin/env python3
"""Sample each staged PDF: first page, middle page, last page. Flag stubs/scans/wrong-content."""
import os, fitz

STAGE = "/opt/indian-legal-ai/gst_stage"

# expected title keyword per filename
EXPECTED = {
    "ipc_1860.pdf": "indian penal code",
    "crpc_1973.pdf": "code of criminal procedure",
    "evidence_act_1872.pdf": "indian evidence act",
    "cpc_1908.pdf": "code of civil procedure",
    "contract_act_1872.pdf": "contract act",
    "ni_act_1881.pdf": "negotiable instruments",
    "partnership_1932.pdf": "partnership",
    "partnership_act_1932.pdf": "partnership",
    "limitation_1963.pdf": "limitation act",
    "limitation_act_1963.pdf": "limitation act",
    "arbitration_1996.pdf": "arbitration",
    "sale_of_goods_1930.pdf": "sale of goods",
    "specific_relief_1963.pdf": "specific relief",
    "sebi_act_1992.pdf": "securities and exchange board",
    "companies_act_2013.pdf": "companies act",
    "income_tax_act_1961.pdf": "income-tax",
    "constitution_2020.pdf": "constitution of india",
    "constitution_2022.pdf": "constitution of india",
    "constitution_of_india.pdf": "constitution",
    "constitution_of_india_full.pdf": "constitution",
    "BNS_2023.pdf": "nyaya sanhita",
    "BSA_2023.pdf": "sakshya",
    "BNSS_2023.pdf": "nagarik suraksha",
    "FEMA_Act_1999.pdf": "foreign exchange management",
    "CGST-Act-2022.pdf": "central goods and services",
    "IGST_Act_2017.pdf": "integrated goods and services",
    "UTGST_Act_2017.pdf": "union territory goods",
    "IBC_Code_2016.pdf": "insolvency and bankruptcy",
    "registration_1908.pdf": "registration act",
    "registration_act_1908.pdf": "registration act",
    "stamp_1899.pdf": "stamp act",
    "stamp_act_1899.pdf": "stamp act",
    "hsa_1956.pdf": "hindu succession",
    "it_act_2000.pdf": "information technology",
    "rti_2005.pdf": "right to information",
    "legal_services_authorities_1987.pdf": "legal services authorities",
    "pmla_2002.pdf": "money-laundering",
    "sarfaesi_2002.pdf": "securitisation",
    "tpa_1882.pdf": "transfer of property",
    "hma_1955.pdf": "hindu marriage",
    "consumer_protection_2019.pdf": "consumer protection",
    "mv_act_1988.pdf": "motor vehicles",
    "dv_act_2005.pdf": "domestic violence",
    "juvenile_justice_2015.pdf": "juvenile justice",
    "pocso_2012.pdf": "sexual offences",
    "wages_2019.pdf": "wages",
    "code_on_wages_2019.pdf": "wages",
    "Code_on_Wages_2019_gaz.pdf": "wages",
    "ir_code_2020.pdf": "industrial relations",
    "industrial_relations_2020.pdf": "industrial relations",
    "ss_code_2020.pdf": "social security",
    "social_security_2020.pdf": "social security",
    "SS_Code_2020_gaz.pdf": "social security",
    "osh_2020.pdf": "occupational safety",
}

# Files to skip from detailed audit (known-circulars/notifications)
SKIP_PATTERNS = ["circular", "notfctn", "schedule", "rate-schedule", "corrigendum",
                 "addendum", "rules", "ruling", "anti-profit", "audit", "e-way",
                 "invoice-", "itc-", "payment-", "refund-", "transition-",
                 "valuation-", "chapter-wise", "gst-compensation",
                 "gst-circular", "cir-cgst", "cgst-amendment", "igst-amendment",
                 "Organization_Chart", "Swachhata"]

def should_skip(fn):
    fl = fn.lower()
    return any(p.lower() in fl for p in SKIP_PATTERNS)

def audit_pdf(path, expected_kw):
    try:
        d = fitz.open(path)
    except Exception as e:
        return {"err": f"open-fail: {e}"}
    pg = d.page_count
    total_chars = 0
    for i in range(pg):
        total_chars += len(d.load_page(i).get_text())
    first = d.load_page(0).get_text()
    mid = d.load_page(pg//2).get_text() if pg > 2 else ""
    last = d.load_page(pg-1).get_text() if pg > 1 else ""
    d.close()
    sz = os.path.getsize(path)
    chars_per_page = total_chars / max(pg,1)
    is_scanned = chars_per_page < 200  # image PDF hallmark
    kw_in_first = expected_kw and expected_kw.lower() in first.lower()[:6000]
    kw_in_body = expected_kw and expected_kw.lower() in (first+mid+last).lower()
    return {
        "pg": pg, "size_kb": sz//1024,
        "total_chars": total_chars, "cpp": int(chars_per_page),
        "scanned": is_scanned,
        "kw_first": kw_in_first, "kw_body": kw_in_body,
        "title_snip": first[:120].replace("\n"," ").strip(),
        "mid_snip": mid[:120].replace("\n"," ").strip(),
        "last_snip": last[:120].replace("\n"," ").strip(),
    }

print("="*90)
print("T1 QUALITY AUDIT — content sampling")
print("="*90)

issues = []
all_files = []

for track in sorted(os.listdir(STAGE)):
    tpath = os.path.join(STAGE, track)
    if not os.path.isdir(tpath) or not track.startswith("t1_"): continue
    pdfs = sorted([f for f in os.listdir(tpath) if f.lower().endswith(".pdf")])
    if not pdfs: continue
    print(f"\n### {track}")
    for f in pdfs:
        if should_skip(f): continue
        full = os.path.join(tpath, f)
        kw = EXPECTED.get(f)
        r = audit_pdf(full, kw)
        all_files.append((track, f, r))
        if "err" in r:
            print(f"  [ERR]   {f}: {r['err']}")
            issues.append((track, f, "open-fail", r.get('err')))
            continue
        flags = []
        if r["scanned"]: flags.append("SCANNED")
        if kw and not r["kw_body"]: flags.append(f"KW-MISS({kw})")
        elif kw and not r["kw_first"]: flags.append("KW-LATE")
        if r["pg"] < 8: flags.append("STUB")
        if r["size_kb"] > 8000: flags.append("LARGE")  # likely Google Books scan
        flag_str = " ["+",".join(flags)+"]" if flags else ""
        print(f"  [{r['pg']:>3}p {r['size_kb']:>5}KB {r['cpp']:>4}c/p]{flag_str} {f}")
        print(f"        1st: {r['title_snip'][:90]}")
        if r["mid_snip"]:
            print(f"        mid: {r['mid_snip'][:90]}")
        if flags:
            issues.append((track, f, ",".join(flags), r['title_snip'][:80]))

print("\n" + "="*90)
print(f"ISSUES FOUND: {len(issues)}")
print("="*90)
for track, f, flag, snip in issues:
    print(f"  [{flag:<30}] {track}/{f}")
    print(f"      {snip}")

# Dedup analysis
print("\n" + "="*90)
print("DUPLICATE CANDIDATES (same keyword, multiple files)")
print("="*90)
by_kw = {}
for track, f, r in all_files:
    if "err" in r: continue
    kw = EXPECTED.get(f)
    if not kw: continue
    by_kw.setdefault(kw, []).append((track, f, r["pg"], r["size_kb"], r["cpp"]))
for kw, entries in by_kw.items():
    if len(entries) > 1:
        print(f"\n  KW: '{kw}' — {len(entries)} files")
        for t,f,pg,sz,cpp in entries:
            print(f"    {t}/{f:<42} {pg:>4}p {sz:>6}KB {cpp:>4}c/p")
