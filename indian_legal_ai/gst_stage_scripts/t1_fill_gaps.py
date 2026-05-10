#!/usr/bin/env python3
"""
Fill-gap downloader for T1 tracks that produced 0 PDFs in the main harvest.
Uses proven indiacode session pattern + direct bitstream URLs + streaming writes.
Targets: A2, A3, A6, A7, A10, A11, A12.
"""
import urllib.request, urllib.error, re, ssl, http.cookiejar, os, time
from urllib.parse import urljoin, quote_plus

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

STAGE = "/opt/indian-legal-ai/gst_stage"

def make_session():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ctx),
    )
    op.addheaders = [("User-Agent", UA), ("Accept", "*/*"),
                     ("Accept-Language", "en-US,en;q=0.9")]
    return op

def stream_to(op, url, dest, timeout=60):
    """Stream a URL to dest file. Returns (ok, size, reason)."""
    try:
        r = op.open(url, timeout=timeout)
        if r.status != 200:
            return False, 0, f"http-{r.status}"
        # Peek first 4 bytes, then stream rest
        head = r.read(4)
        if head != b"%PDF":
            # still consume rest to keep session clean
            _ = r.read(1024)
            return False, 0, f"not-pdf head={head!r}"
        tmp = dest + ".part"
        n = 4
        with open(tmp, "wb") as f:
            f.write(head)
            while True:
                chunk = r.read(65536)
                if not chunk: break
                f.write(chunk); n += len(chunk)
        os.replace(tmp, dest)
        return True, n, "ok"
    except urllib.error.HTTPError as e:
        return False, 0, f"http-{e.code}"
    except Exception as e:
        return False, 0, str(e)[:80]

def warm(op):
    for u in ["https://www.indiacode.nic.in/",
              "https://www.indiacode.nic.in/handle/123456789/1362"]:
        try: op.open(u, timeout=15).read(1000)
        except Exception: pass


# =============================================================================
# Known-good direct bitstream URLs (proven in prior probe rounds)
# =============================================================================
# Format: (track_dir, filename, url)
DIRECT = [
    # A2 Income Tax
    ("t1_income_tax", "income_tax_act_1961.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/2435/1/A1961-43.pdf"),

    # A3 Companies
    ("t1_companies_sebi", "companies_act_2013.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/2114/5/a2013-18.pdf"),

    # A6 Commercial Acts (confirmed in probe3)
    ("t1_commercial_acts", "ni_act_1881.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/2189/1/AAA1881___26.pdf"),
    ("t1_commercial_acts", "contract_act_1872.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/2187/1/AAA1872___9.pdf"),
    ("t1_commercial_acts", "arbitration_1996.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/1978/1/A1996-26.pdf"),
    ("t1_commercial_acts", "specific_relief_1963.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/1580/1/A1963-47.pdf"),

    # A7 CPC (confirmed in probe3)
    ("t1_civil_procedure", "cpc_1908.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/2191/1/AAA1908___5.pdf"),
]

# =============================================================================
# Fallback: search indiacode for these titles and pick best match
# =============================================================================
# (track_dir, filename_stem, search_query, must_contain_keywords)
SEARCH_TARGETS = [
    # A2 Income Tax rules
    ("t1_income_tax", "income_tax_rules_1962", "Income-tax Rules 1962",
     ["income", "rules"]),

    # A6 commercial
    ("t1_commercial_acts", "sale_of_goods_act_1930", "Sale of Goods Act 1930",
     ["sale", "goods"]),
    ("t1_commercial_acts", "partnership_act_1932", "Indian Partnership Act 1932",
     ["partnership"]),
    ("t1_commercial_acts", "limitation_act_1963", "Limitation Act 1963",
     ["limitation"]),

    # A10 Old Criminal
    ("t1_old_criminal", "ipc_1860", "Indian Penal Code 1860",
     ["penal"]),
    ("t1_old_criminal", "crpc_1973", "Code of Criminal Procedure 1973",
     ["criminal", "procedure"]),
    ("t1_old_criminal", "evidence_act_1872", "Indian Evidence Act 1872",
     ["evidence"]),

    # A12 Other bare acts
    ("t1_other_bare_acts", "tpa_1882", "Transfer of Property Act 1882",
     ["transfer", "property"]),
    ("t1_other_bare_acts", "registration_act_1908", "Registration Act 1908",
     ["registration"]),
    ("t1_other_bare_acts", "stamp_act_1899", "Indian Stamp Act 1899",
     ["stamp"]),
    ("t1_other_bare_acts", "consumer_protection_2019", "Consumer Protection Act 2019",
     ["consumer"]),
    ("t1_other_bare_acts", "mv_act_1988", "Motor Vehicles Act 1988",
     ["motor", "vehicle"]),
    ("t1_other_bare_acts", "hma_1955", "Hindu Marriage Act 1955",
     ["hindu", "marriage"]),
    ("t1_other_bare_acts", "hsa_1956", "Hindu Succession Act 1956",
     ["hindu", "succession"]),
]

# =============================================================================
# A11 Constitution — legislative.gov.in direct URLs (tried in order)
# =============================================================================
CONSTITUTION_TRIES = [
    ("constitution_of_india.pdf",
     "https://legislative.gov.in/sites/default/files/COI.pdf"),
    ("constitution_of_india.pdf",
     "https://legislative.gov.in/sites/default/files/coi-4March2016.pdf"),
    ("constitution_of_india.pdf",
     "https://www.mea.gov.in/Images/pdf1/Part1.pdf"),
]


def search_and_download(op, track_dir, stem, query, must_contain):
    """Search indiacode, find matching handle, fetch its PDF."""
    dest_dir = os.path.join(STAGE, track_dir)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, stem + ".pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 50000:
        return True, os.path.getsize(dest), "already-present"

    # Use central-acts collection search
    url = f"https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query={quote_plus(query)}&submit=Go"
    try:
        r = op.open(url, timeout=30)
        html = r.read(2_000_000).decode("utf-8", "ignore")
    except Exception as e:
        return False, 0, f"search-err: {str(e)[:60]}"

    # Find item links in result table: /handle/123456789/NNNN with descriptive title
    # Results typically appear as table rows containing handle links
    candidates = re.findall(r'<a\s+href="(/handle/123456789/\d+)"[^>]*>([^<]{8,250})</a>', html)
    chosen = None
    for href, title in candidates:
        tl = title.lower().strip()
        if all(k.lower() in tl for k in must_contain):
            chosen = (href, tl)
            break
    if not chosen:
        # Looser: match ANY must_contain keyword
        for href, title in candidates:
            tl = title.lower().strip()
            if any(k.lower() in tl for k in must_contain):
                chosen = (href, tl)
                break
    if not chosen:
        return False, 0, f"no-match ({len(candidates)} cand)"

    handle_url = "https://www.indiacode.nic.in" + chosen[0]
    try:
        r = op.open(handle_url, timeout=30)
        phtml = r.read(1_500_000).decode("utf-8", "ignore")
    except Exception as e:
        return False, 0, f"handle-err: {str(e)[:60]}"

    pdfs = re.findall(r'href="(/bitstream/[^"]+\.(?:pdf|PDF))"', phtml)
    if not pdfs:
        return False, 0, "no-pdf-on-handle"
    pdf_url = "https://www.indiacode.nic.in" + pdfs[0]
    return stream_to(op, pdf_url, dest)


# =============================================================================
# MAIN
# =============================================================================
print("=" * 80)
print("T1 FILL-GAP DOWNLOADER")
print("=" * 80)

op = make_session()
warm(op)

results = {}

print("\n## PHASE 1: Direct known-good bitstream URLs")
for track, fname, url in DIRECT:
    dest_dir = os.path.join(STAGE, track)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 50000:
        ok, size, reason = True, os.path.getsize(dest), "already-present"
    else:
        ok, size, reason = stream_to(op, url, dest)
    tag = f"[OK {size//1024}KB]" if ok else f"[FAIL {reason}]"
    print(f"  {tag} {track}/{fname}")
    results[f"{track}/{fname}"] = (ok, size, reason)
    time.sleep(0.4)

print("\n## PHASE 2: Search-and-download from indiacode catalog")
for track, stem, query, must in SEARCH_TARGETS:
    ok, size, reason = search_and_download(op, track, stem, query, must)
    tag = f"[OK {size//1024}KB]" if ok else f"[FAIL {reason}]"
    print(f"  {tag} {track}/{stem}.pdf  query={query!r}")
    results[f"{track}/{stem}"] = (ok, size, reason)
    time.sleep(0.6)

print("\n## PHASE 3: Constitution of India")
const_dir = os.path.join(STAGE, "t1_constitution")
os.makedirs(const_dir, exist_ok=True)
const_done = False
for fname, url in CONSTITUTION_TRIES:
    dest = os.path.join(const_dir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 50000:
        print(f"  [already-present] {fname}")
        const_done = True
        break
    ok, size, reason = stream_to(op, url, dest)
    tag = f"[OK {size//1024}KB]" if ok else f"[FAIL {reason}]"
    print(f"  {tag} {fname}  <- {url[:80]}")
    if ok:
        const_done = True
        break
if not const_done:
    # Try indiacode as last resort
    ok, size, reason = search_and_download(op, "t1_constitution", "constitution_of_india",
                                           "Constitution of India", ["constitution"])
    tag = f"[OK {size//1024}KB]" if ok else f"[FAIL {reason}]"
    print(f"  {tag} t1_constitution (indiacode fallback)")

print("\n## PHASE 4: Labour Codes — remaining two")
# A8 already has 2 of 4. Try to get the missing two.
LABOUR_TRIES = [
    ("t1_labour_codes", "social_security_code_2020",
     "Code on Social Security 2020", ["social", "security"]),
    ("t1_labour_codes", "osh_code_2020",
     "Occupational Safety Health Working Conditions Code 2020", ["occupational"]),
    ("t1_labour_codes", "code_on_wages_2019",
     "Code on Wages 2019", ["wages"]),
    ("t1_labour_codes", "industrial_relations_code_2020",
     "Industrial Relations Code 2020", ["industrial"]),
]
for track, stem, query, must in LABOUR_TRIES:
    ok, size, reason = search_and_download(op, track, stem, query, must)
    tag = f"[OK {size//1024}KB]" if ok else f"[FAIL {reason}]"
    print(f"  {tag} {track}/{stem}.pdf")
    time.sleep(0.6)

# =============================================================================
# FINAL TALLY
# =============================================================================
print("\n\n" + "=" * 80)
print("FINAL STAGING TALLY")
print("=" * 80)
total_pdfs = 0
total_kb = 0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"):
        continue
    pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
    kb = sum(os.path.getsize(os.path.join(full, f)) for f in pdfs) // 1024
    total_pdfs += len(pdfs); total_kb += kb
    status = "OK " if len(pdfs) > 0 else "!! "
    print(f"  [{status}] {d:<40}  {len(pdfs):>3} pdfs,  {kb:>7} KB")
print(f"\n  TOTAL: {total_pdfs} PDFs, {total_kb} KB ({total_kb/1024:.1f} MB)")
print("\n=== DONE ===")
