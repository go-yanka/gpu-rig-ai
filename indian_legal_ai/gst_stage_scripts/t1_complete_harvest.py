#!/usr/bin/env python3
"""
T1 complete harvest:
 1. Enumerate full indiacode Central Acts index (paginated short-title browse)
 2. Match to the T1 target list, get handle IDs
 3. Fetch each handle -> extract PDF link -> download PDF
 4. Identify BNS/BNSS/BSA via content inspection of egazette/MHA PDFs
 5. Find A8 Labour Codes via PRS India + other sources
 6. Grab FEMA notifications, RBI master directions, IBC regulations
 7. Stage everything under /opt/indian-legal-ai/gst_stage/t1_<track>/
"""
import urllib.request, urllib.error, re, ssl, http.cookiejar, os, sys, time, json
from urllib.parse import urljoin, quote_plus

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def make_session():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ctx),
    )
    op.addheaders = [("User-Agent", UA), ("Accept", "*/*"),
                     ("Accept-Language", "en-US,en;q=0.9")]
    return op


def fetch(op, url, timeout=30, max_bytes=3_000_000):
    try:
        r = op.open(url, timeout=timeout)
        return r.status, r.read(max_bytes), r.url
    except urllib.error.HTTPError as e:
        try: body = e.read(2000)
        except Exception: body = b""
        return e.code, body, url
    except Exception as e:
        return -1, str(e).encode(), url


def save_pdf(path, data):
    if data[:4] != b"%PDF":
        return False
    if len(data) < 15000:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return True


BASE = "/opt/indian-legal-ai/gst_stage"
DIRS = {
    "A1": f"{BASE}/t1_criminal_codes_2023",
    "A2": f"{BASE}/t1_income_tax",
    "A3": f"{BASE}/t1_companies_sebi",
    "A4": f"{BASE}/t1_ibc",
    "A5": f"{BASE}/t1_fema_rbi",
    "A6": f"{BASE}/t1_commercial_acts",
    "A7": f"{BASE}/t1_civil_procedure",
    "A8": f"{BASE}/t1_labour_codes",
    "A9": f"{BASE}/t1_gst_circulars",
    "A10": f"{BASE}/t1_old_criminal",  # IPC/CrPC/Evidence still needed alongside BNS
    "A11": f"{BASE}/t1_constitution",
    "A12": f"{BASE}/t1_other_bare_acts",  # TP Act, Partnership, SoGA, Specific Relief
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

op = make_session()
fetch(op, "https://www.indiacode.nic.in/", timeout=15)  # warm session


# ============================================================================
# STEP 1: Enumerate the full Central Acts index via pagination
# URL: https://www.indiacode.nic.in/handle/123456789/1362/browse?type=shorttitle&offset=N
# ============================================================================
print("=" * 80)
print("STEP 1 — full Central Acts enumeration")
print("=" * 80)

catalog = {}  # title -> handle URL
offset = 0
total_pages = 0
while total_pages < 40:  # 40 pages * ~20 per page = 800 acts (overkill but safe)
    url = f"https://www.indiacode.nic.in/handle/123456789/1362/browse?type=shorttitle&offset={offset}"
    status, body, _ = fetch(op, url, timeout=40)
    if status != 200:
        print(f"  [{status}] page at offset {offset}, stopping")
        break
    html = body.decode("utf-8", "ignore")
    # The act rows are in table-like structure. Each row has /handle/ URL + title.
    # Pattern based on DSpace XMLUI output
    rows = re.findall(r'<a\s+href="(/handle/123456789/\d+)"[^>]*>\s*([^<]{6,200})\s*</a>', html)
    added = 0
    for href, title in rows:
        title = title.strip()
        if len(title) < 6: continue
        # Skip navigation/generic links
        if title.lower() in ["short title", "act number", "central acts", "next", "previous",
                             "state acts", "unrepealed", "repealed"]: continue
        if href.endswith("/1362"): continue
        full_url = urljoin("https://www.indiacode.nic.in/", href)
        if title not in catalog:
            catalog[title] = full_url
            added += 1
    if added == 0:
        # could be end of listing
        print(f"  offset {offset}: no new entries, stopping")
        break
    print(f"  offset {offset}: +{added} (total {len(catalog)})")
    offset += 20  # DSpace default page size
    total_pages += 1
    if total_pages % 5 == 0:
        time.sleep(0.5)  # be nice

print(f"\n  TOTAL acts enumerated: {len(catalog)}")

# Show sample of what we found (for BNS / labour hunt)
keywords_scan = ["bharatiya", "nyaya", "sanhita", "nagarik", "suraksha", "sakshya",
                 "code on wages", "social security", "industrial relations", "occupational",
                 "evidence", "constitution", "transfer of property", "criminal procedure",
                 "partnership", "sale of goods"]
print(f"\n  BNS/Labour/missing matches in catalog:")
for kw in keywords_scan:
    for title, url in catalog.items():
        if kw in title.lower():
            print(f"    [{kw}]  {title[:80]}  -> {url[-40:]}")


# ============================================================================
# STEP 2: Map catalog to T1 targets
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 2 — map catalog to T1 targets")
print("=" * 80)

TARGETS = {
    # A1 - New Criminal Codes
    "Bharatiya Nyaya Sanhita":           ("A1", "BNS_2023"),
    "Bharatiya Nagarik Suraksha":        ("A1", "BNSS_2023"),
    "Bharatiya Sakshya":                 ("A1", "BSA_2023"),
    # A2 - Income Tax
    "Income-tax Act":                    ("A2", "IT_Act_1961"),
    "Income-Tax Act":                    ("A2", "IT_Act_1961"),
    "Finance Act, 2023":                 ("A2", "Finance_Act_2023"),
    "Finance Act, 2024":                 ("A2", "Finance_Act_2024"),
    "Finance (No. 2) Act, 2024":         ("A2", "Finance_No2_Act_2024"),
    # A3 - Companies + SEBI
    "Companies Act, 2013":               ("A3", "Companies_Act_2013"),
    "Securities and Exchange Board of India Act":  ("A3", "SEBI_Act_1992"),
    "Securities Contracts":              ("A3", "SCRA_1956"),
    "Depositories Act":                  ("A3", "Depositories_Act_1996"),
    # A4 - IBC
    "Insolvency and Bankruptcy Code":    ("A4", "IBC_2016"),
    # A5 - FEMA
    "Foreign Exchange Management":       ("A5", "FEMA_1999"),
    "Prevention of Money Laundering":    ("A5", "PMLA_2002"),
    # A6 - Commercial bare acts
    "Negotiable Instruments":            ("A6", "NI_Act_1881"),
    "Indian Contract Act":               ("A6", "Contract_Act_1872"),
    "Sale of Goods Act":                 ("A6", "SoGA_1930"),
    "Indian Partnership Act":            ("A6", "Partnership_1932"),
    "Specific Relief Act":               ("A6", "Specific_Relief_1963"),
    "Arbitration and Conciliation Act":  ("A6", "Arbitration_1996"),
    "Limitation Act":                    ("A6", "Limitation_1963"),
    # A7 - Civil procedure
    "Code of Civil Procedure":           ("A7", "CPC_1908"),
    "Civil Procedure Code":              ("A7", "CPC_1908"),
    # A8 - Labour codes
    "Code on Wages":                     ("A8", "Code_on_Wages_2019"),
    "Industrial Relations Code":         ("A8", "IR_Code_2020"),
    "Code on Social Security":           ("A8", "SS_Code_2020"),
    "Occupational Safety, Health":       ("A8", "OSH_Code_2020"),
    "Industrial Disputes Act":           ("A8", "ID_Act_1947"),
    "Factories Act":                     ("A8", "Factories_Act_1948"),
    "Payment of Gratuity":               ("A8", "Gratuity_Act_1972"),
    "Employees Provident":               ("A8", "EPF_Act_1952"),
    # A10 - Old criminal (still in force, parallel to BNS)
    "Indian Penal Code":                 ("A10", "IPC_1860"),
    "Code of Criminal Procedure":        ("A10", "CrPC_1973"),
    "Indian Evidence Act":               ("A10", "Evidence_Act_1872"),
    # A11 - Constitution
    "Constitution of India":             ("A11", "Constitution_India"),
    # A12 - Other bare acts
    "Transfer of Property Act":          ("A12", "TPA_1882"),
    "Registration Act":                  ("A12", "Registration_Act_1908"),
    "Indian Stamp Act":                  ("A12", "Stamp_Act_1899"),
    "Consumer Protection Act, 2019":     ("A12", "CP_Act_2019"),
    "Motor Vehicles Act, 1988":          ("A12", "MV_Act_1988"),
    "Hindu Marriage":                    ("A12", "HMA_1955"),
    "Hindu Succession":                  ("A12", "HSA_1956"),
    "Code on Direct Tax":                ("A2", "Direct_Tax_Code"),
    "Benami Transactions":               ("A2", "Benami_1988"),
    "Black Money":                       ("A2", "Black_Money_2015"),
    "Prohibition of Benami":             ("A2", "Prohibition_Benami_2016"),
}

# Match each target against catalog titles (case-insensitive partial match)
matched = {}  # out_name -> (track, handle_url, catalog_title)
for tkey, (track, out_name) in TARGETS.items():
    tkey_low = tkey.lower()
    for title, handle_url in catalog.items():
        if tkey_low in title.lower():
            # Prefer one where the catalog title starts with the tkey
            if out_name not in matched:
                matched[out_name] = (track, handle_url, title)
            else:
                # replace only if current catalog title is shorter / more specific
                _, _, existing_title = matched[out_name]
                if len(title) < len(existing_title) and tkey_low in title.lower():
                    matched[out_name] = (track, handle_url, title)

print(f"\n  Matched {len(matched)} targets to catalog entries:")
for out_name, (track, url, title) in sorted(matched.items()):
    print(f"    [{track}] {out_name:30} -> {title[:70]}")


# ============================================================================
# STEP 3: Fetch each handle -> extract PDF link -> download PDF
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 3 — download PDFs for each matched target")
print("=" * 80)

downloaded = {}
for out_name, (track, handle_url, title) in sorted(matched.items()):
    out_dir = DIRS[track]
    out_path = os.path.join(out_dir, f"{out_name}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 20000:
        downloaded[out_name] = (track, out_path, os.path.getsize(out_path))
        print(f"  [have] {out_name}")
        continue

    status, body, _ = fetch(op, handle_url, timeout=40)
    if status != 200:
        print(f"  [handle {status}] {out_name}")
        continue
    html = body.decode("utf-8", "ignore")
    # Find the main PDF bitstream (prefer newer/amended version = later number in path)
    pdf_links = re.findall(r'href="(/bitstream/123456789/\d+/\d+/[^"]+\.(?:pdf|PDF))"', html)
    if not pdf_links:
        print(f"  [no-pdf-link] {out_name}")
        continue
    # Pick the last (most recent amended) version if multiple
    pdf_url = "https://www.indiacode.nic.in" + pdf_links[-1]
    s2, b2, _ = fetch(op, pdf_url, timeout=60, max_bytes=50_000_000)
    if s2 == 200 and save_pdf(out_path, b2):
        downloaded[out_name] = (track, out_path, len(b2))
        print(f"  [OK {len(b2)//1024}KB] {out_name}")
    else:
        print(f"  [fail {s2} {len(b2)}B] {out_name}  -> {pdf_url[-50:]}")
    time.sleep(0.3)


# ============================================================================
# STEP 4: BNS/BNSS/BSA via egazette+MHA — identify by content
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 4 — BNS/BNSS/BSA via MHA + egazette Wayback")
print("=" * 80)

# All egazette numbers that returned PDFs earlier
EGAZ_CANDIDATES = [
    # Via wayback
    ("https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/250880.pdf", "egaz_250880.pdf"),
    ("https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/250881.pdf", "egaz_250881.pdf"),
    ("https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/250882.pdf", "egaz_250882.pdf"),
    ("https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/250883.pdf", "egaz_250883.pdf"),
    ("https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/250884.pdf", "egaz_250884.pdf"),
    ("https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/250885.pdf", "egaz_250885.pdf"),
    ("https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/250886.pdf", "egaz_250886.pdf"),
    # MHA direct
    ("https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf", "mha_250883.pdf"),
]

# Download all candidates to a tmp identification dir
tmp = "/tmp/bns_id"
os.makedirs(tmp, exist_ok=True)
candidate_files = []
for url, fn in EGAZ_CANDIDATES:
    path = os.path.join(tmp, fn)
    if not os.path.exists(path):
        s, b, _ = fetch(op, url, timeout=60, max_bytes=30_000_000)
        if s == 200 and b[:4] == b"%PDF":
            with open(path, "wb") as f: f.write(b)
            print(f"  [dl {len(b)//1024}KB] {fn}")
        else:
            print(f"  [{s}] {fn}  skipped")
            continue
    else:
        print(f"  [have] {fn}")
    candidate_files.append(path)

# Now content-identify using PyMuPDF
try:
    import fitz
    for path in candidate_files:
        try:
            d = fitz.open(path)
            # Read first 5 pages of text
            head_text = ""
            for i in range(min(5, len(d))):
                head_text += d[i].get_text("text")
            d.close()
        except Exception as e:
            print(f"  [fitz err] {path}: {e}")
            continue
        head_lower = head_text.lower()
        tag = None
        score = 0
        checks = {
            "BNS_2023":  (["bharatiya nyaya sanhita"], ["nagarik", "sakshya adhiniyam"]),
            "BNSS_2023": (["bharatiya nagarik suraksha sanhita"], ["nyaya sanhita as a standalone", "sakshya adhiniyam"]),
            "BSA_2023":  (["bharatiya sakshya adhiniyam"], ["nyaya sanhita standalone", "nagarik suraksha standalone"]),
        }
        best = None
        for name, (pos, _) in checks.items():
            if any(p in head_lower for p in pos):
                best = name
                break
        if best:
            out_path = os.path.join(DIRS["A1"], f"{best}.pdf")
            with open(path, "rb") as fin, open(out_path, "wb") as fout:
                fout.write(fin.read())
            sz = os.path.getsize(out_path)
            downloaded[best] = ("A1", out_path, sz)
            print(f"  [IDENT {best}] from {os.path.basename(path)} ({sz//1024}KB)")
        else:
            # fallback — save as unknown_gazette under A1 for manual inspection
            fn = os.path.basename(path)
            out_path = os.path.join(DIRS["A1"], f"gazette_{fn}")
            if not os.path.exists(out_path):
                with open(path, "rb") as fin, open(out_path, "wb") as fout:
                    fout.write(fin.read())
            print(f"  [?] {os.path.basename(path)}: content unidentified, saved as {os.path.basename(out_path)}")
except ImportError:
    print("  [ERR] PyMuPDF not available — can't content-identify")


# ============================================================================
# STEP 5: A8 Labour Codes via alt sources
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 5 — Labour Codes alt sources")
print("=" * 80)

# If not already matched from catalog, try:
LABOUR_ALT = [
    # Wayback of egazette — Code on Wages was gazetted Aug 8 2019 (#41)
    ("https://web.archive.org/web/2020/https://egazette.gov.in/WriteReadData/2019/210356.pdf", "Code_on_Wages_2019_gaz.pdf"),
    ("https://web.archive.org/web/2021/https://egazette.gov.in/WriteReadData/2020/222114.pdf", "IR_Code_2020_gaz.pdf"),
    ("https://web.archive.org/web/2021/https://egazette.gov.in/WriteReadData/2020/222040.pdf", "SS_Code_2020_gaz.pdf"),
    ("https://web.archive.org/web/2021/https://egazette.gov.in/WriteReadData/2020/221934.pdf", "OSH_Code_2020_gaz.pdf"),
    # Direct labour.gov.in paths that some tools have reported
    ("https://labour.gov.in/sites/default/files/the_code_on_wages_2019_no._29_of_2019.pdf", "Code_on_Wages_2019.pdf"),
    ("https://labour.gov.in/sites/default/files/ir_gazette_of_india.pdf", "IR_Code_2020.pdf"),
    ("https://labour.gov.in/sites/default/files/ss_code_gazette.pdf", "SS_Code_2020.pdf"),
    ("https://labour.gov.in/sites/default/files/osh_gazette.pdf", "OSH_Code_2020.pdf"),
    # Indian Kanoon mirror
    ("https://indiankanoon.org/doc/190076877/", "Code_on_Wages_2019_ik.html"),
]
for url, fn in LABOUR_ALT:
    out_path = os.path.join(DIRS["A8"], fn)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 20000:
        print(f"  [have] {fn}")
        continue
    s, b, _ = fetch(op, url, timeout=60, max_bytes=30_000_000)
    if s == 200 and fn.endswith(".pdf") and b[:4] == b"%PDF" and len(b) > 30000:
        with open(out_path, "wb") as f: f.write(b)
        print(f"  [OK {len(b)//1024}KB] {fn}")
    else:
        print(f"  [{s} {len(b) if isinstance(b,bytes) else '?'}B] {fn}  skipped")


# Also — scrape PRS India acts page for labour codes
print("\n  PRS India enacted acts hunt:")
PRS_SEARCHES = [
    "https://prsindia.org/billtrack?title=wages",
    "https://prsindia.org/billtrack?title=industrial",
    "https://prsindia.org/billtrack?title=social+security",
    "https://prsindia.org/billtrack?title=occupational",
    "https://prsindia.org/search?k=Code+on+Wages",
]
for u in PRS_SEARCHES:
    s, b, _ = fetch(op, u, timeout=30)
    if s == 200 and isinstance(b, bytes):
        html = b.decode("utf-8", "ignore")
        pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
        rel = [p for p in pdfs if any(k in p.lower() for k in
               ["wages", "industrial_relation", "industrial-relation", "ir-code", "social_security",
                "social-security", "ss-code", "osh", "occupational", "code-on"])]
        print(f"  [{s}] {u}: {len(pdfs)} pdfs, {len(rel)} relevant")
        for p in rel[:5]:
            full = urljoin(u, p)
            fn = os.path.basename(full.split("?")[0])[:80]
            out_path = os.path.join(DIRS["A8"], fn)
            if os.path.exists(out_path): continue
            s2, b2, _ = fetch(op, full, timeout=60, max_bytes=30_000_000)
            if s2 == 200 and b2[:4] == b"%PDF":
                with open(out_path, "wb") as f: f.write(b2)
                print(f"    [+{len(b2)//1024}KB] {fn}")


# ============================================================================
# STEP 6: FEMA + RBI Master Directions (A5) — harvest notification PDFs
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 6 — RBI FEMA notifications + Master Directions")
print("=" * 80)
RBI_PAGES = [
    "https://www.rbi.org.in/Scripts/BS_FemaNotifications.aspx",
    "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx",
]
rbi_got = 0
for p in RBI_PAGES:
    s, b, _ = fetch(op, p, timeout=30)
    if s != 200: continue
    html = b.decode("utf-8", "ignore")
    pdfs = re.findall(r'href=["\']([^"\']+\.(?:pdf|PDF))["\']', html)
    pdfs = [urljoin(p, pu) for pu in pdfs]
    # keep FEMA / Master Direction / Foreign
    keep = [u for u in pdfs if any(k in u.lower() for k in ["fema","master","foreign","direction","ndsom"])]
    print(f"  {p}: {len(pdfs)} total, {len(keep)} FEMA/MD")
    for u in keep[:40]:
        fn = os.path.basename(u.split("?")[0])[:80]
        out_path = os.path.join(DIRS["A5"], fn)
        if os.path.exists(out_path): continue
        s2, b2, _ = fetch(op, u, timeout=60, max_bytes=20_000_000)
        if s2 == 200 and b2[:4] == b"%PDF":
            with open(out_path, "wb") as f: f.write(b2)
            rbi_got += 1
    time.sleep(0.3)
print(f"  RBI PDFs saved: {rbi_got}")


# ============================================================================
# STEP 7: IBC regulations from ibbi.gov.in
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 7 — IBBI legal framework")
print("=" * 80)
# seed: already confirmed direct PDF + legal framework page
IBBI_SEEDS = [
    "https://ibbi.gov.in/uploads/legalframwork/48bf32150f5d6b30477b74f652964edc.pdf",
    "https://ibbi.gov.in/legal-framework/act",
    "https://ibbi.gov.in/legal-framework/rules",
    "https://ibbi.gov.in/legal-framework/regulations",
    "https://ibbi.gov.in/legal-framework/circulars-and-notifications",
]
# Save the direct PDF first
ibc_dir = DIRS["A4"]
direct_url = IBBI_SEEDS[0]
out_path = os.path.join(ibc_dir, "IBC_Code_2016.pdf")
if not os.path.exists(out_path):
    s, b, _ = fetch(op, direct_url, timeout=60, max_bytes=30_000_000)
    if s == 200 and b[:4] == b"%PDF":
        with open(out_path, "wb") as f: f.write(b)
        print(f"  [OK {len(b)//1024}KB] IBC_Code_2016.pdf")
# Scrape legal-framework pages for all regulation PDFs
ibbi_got = 0
for p in IBBI_SEEDS[1:]:
    s, b, _ = fetch(op, p, timeout=30)
    if s != 200: continue
    html = b.decode("utf-8","ignore")
    pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
    pdfs = [urljoin(p, pu) for pu in pdfs][:50]  # cap
    for u in pdfs:
        fn = os.path.basename(u.split("?")[0])[:80]
        out_path = os.path.join(ibc_dir, fn)
        if os.path.exists(out_path): continue
        s2, b2, _ = fetch(op, u, timeout=60, max_bytes=20_000_000)
        if s2 == 200 and b2[:4] == b"%PDF" and len(b2) > 20000:
            with open(out_path, "wb") as f: f.write(b2)
            ibbi_got += 1
print(f"  IBBI PDFs saved (in addition to bare code): {ibbi_got}")


# ============================================================================
# STEP 8: A9 GST circulars — harvest from cbic-gst.gov.in
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 8 — cbic-gst.gov.in circulars/notifications harvest")
print("=" * 80)
CBIC_PAGES = [
    "https://cbic-gst.gov.in/",
    "https://cbic-gst.gov.in/gst-goods-services-rates.html",
    "https://cbic-gst.gov.in/vacancy-circulars.html",
]
cbic_got = 0
seen_cbic = set()
for p in CBIC_PAGES:
    s, b, _ = fetch(op, p, timeout=30, max_bytes=3_000_000)
    if s != 200: continue
    html = b.decode("utf-8","ignore")
    pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
    for pu in pdfs:
        full = urljoin(p, pu)
        if full in seen_cbic: continue
        seen_cbic.add(full)
        # Keep circular / notification / rate / rule
        fl = full.lower()
        if not any(k in fl for k in ["circular","notfctn","notif","cgst","igst","rate","rule"]):
            continue
        fn = os.path.basename(full.split("?")[0])[:80]
        out_path = os.path.join(DIRS["A9"], fn)
        if os.path.exists(out_path): continue
        s2, b2, _ = fetch(op, full, timeout=45, max_bytes=20_000_000)
        if s2 == 200 and b2[:4] == b"%PDF" and len(b2) > 15000:
            with open(out_path, "wb") as f: f.write(b2)
            cbic_got += 1
print(f"  CBIC GST PDFs saved: {cbic_got}")


# ============================================================================
# STEP 9: SEBI regulations (LODR, Takeover, Insider Trading, AIF etc.)
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 9 — SEBI regulations")
print("=" * 80)
SEBI_PAGES = [
    "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=3&smid=0",
    "https://www.sebi.gov.in/legal-framework.html",
    "https://www.sebi.gov.in/sebi_data/commondocs/",
    "https://www.sebi.gov.in/legal/regulations.html",
]
sebi_got = 0
for p in SEBI_PAGES:
    s, b, _ = fetch(op, p, timeout=30)
    if s != 200: continue
    html = b.decode("utf-8","ignore")
    pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
    pdfs = [urljoin(p, pu) for pu in pdfs][:50]
    print(f"  {p}: {len(pdfs)} pdfs")
    for u in pdfs[:20]:
        fn = os.path.basename(u.split("?")[0])[:80]
        out_path = os.path.join(DIRS["A3"], f"SEBI_{fn}")
        if os.path.exists(out_path): continue
        s2, b2, _ = fetch(op, u, timeout=45, max_bytes=15_000_000)
        if s2 == 200 and b2[:4] == b"%PDF" and len(b2) > 15000:
            with open(out_path, "wb") as f: f.write(b2)
            sebi_got += 1
print(f"  SEBI PDFs saved: {sebi_got}")


# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n\n" + "=" * 80)
print("FINAL STAGING REPORT")
print("=" * 80)
for track, d in sorted(DIRS.items()):
    files = [f for f in os.listdir(d) if f.endswith(".pdf")]
    total_size = sum(os.path.getsize(os.path.join(d, f)) for f in files)
    print(f"  {track}  {d:50}  {len(files):>3} pdfs, {total_size//1024:>8}KB")

print("\n=== HARVEST COMPLETE ===")
