#!/usr/bin/env python3
"""
Deep probe round 4 — final targeted:
 1. indiacode brute-force handle scan for: BNS, BNSS, BSA, Labour Codes,
    Evidence Act, CrPC, TP Act, Partnership, SoGA, Constitution
    Strategy: use the indiacode simple-search with POST-like query and parse results properly
 2. egazette known BNS PDF numbers via Wayback
 3. Find the correct handle for Constitution
"""
import urllib.request, urllib.error, re, ssl, http.cookiejar, time
from urllib.parse import urljoin, quote_plus

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def make_session():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ctx),
    )
    opener.addheaders = [("User-Agent", UA), ("Accept", "*/*"),
                         ("Accept-Language", "en-US,en;q=0.9")]
    return opener


def fetch(opener, url, timeout=30):
    try:
        r = opener.open(url, timeout=timeout)
        return r.status, r.read(2_500_000), r.url
    except urllib.error.HTTPError as e:
        try: body = e.read(4000)
        except Exception: body = b""
        return e.code, body, url
    except Exception as e:
        return -1, str(e).encode(), url


# ============================================================================
# PART 1: indiacode real search via discover endpoint
# ============================================================================
print("=" * 80)
print("PART 1 — indiacode /search endpoint (different from simple-search)")
print("=" * 80)

op = make_session()
fetch(op, "https://www.indiacode.nic.in/", timeout=15)

QUERIES = [
    "Bharatiya Nyaya Sanhita 2023",
    "Bharatiya Nagarik Suraksha Sanhita 2023",
    "Bharatiya Sakshya Adhiniyam 2023",
    "Code on Wages 2019",
    "Industrial Relations Code 2020",
    "Code on Social Security 2020",
    "Occupational Safety Health Working Conditions",
    "Indian Evidence Act",
    "Transfer of Property Act",
    "Code of Criminal Procedure 1973",
    "Indian Partnership Act 1932",
    "Sale of Goods Act 1930",
    "Constitution of India",
]

found_handles = {}
for q in QUERIES:
    # Try /handle/123456789/1362/simple-search which lives inside "Central Acts" collection
    url = f"https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query={quote_plus(q)}&submit=Go"
    status, body, _ = fetch(op, url, timeout=30)
    if status != 200:
        print(f"  [{status}] search '{q}'"); continue
    html = body.decode("utf-8", "ignore")
    # Find act-title links (skip navigation; look for links to /handle/<num>/<id> where id is numeric item)
    # Pattern: <a href="/handle/123456789/NNNN">Title</a>
    matches = re.findall(r'<a\s+href="(/handle/123456789/\d+)"[^>]*>([^<]{10,200})</a>', html)
    qlow = q.lower()
    qwords = [w for w in qlow.split() if len(w) > 3]
    relevant = []
    for href, title in matches:
        tl = title.lower()
        # require at least 2 significant words from query to match
        match_count = sum(1 for w in qwords if w in tl)
        if match_count >= min(2, len(qwords)):
            relevant.append((href, title.strip()))
    # Dedup
    seen = set(); uniq = []
    for h, t in relevant:
        if h not in seen: seen.add(h); uniq.append((h,t))
    print(f"\n  '{q}': {len(uniq)} relevant matches")
    for h, t in uniq[:3]:
        full = "https://www.indiacode.nic.in" + h
        found_handles[t] = full
        print(f"    {full}  -> {t[:80]}")


# ============================================================================
# PART 2: visit each found handle and extract the PDF
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 2 — extract PDFs from found handles")
print("=" * 80)

confirmed_pdfs = {}
for title, handle_url in found_handles.items():
    status, body, _ = fetch(op, handle_url, timeout=30)
    if status != 200:
        print(f"  [{status}] {title[:50]}"); continue
    html = body.decode("utf-8", "ignore")
    m = re.search(r'href="(/bitstream/[^"]+\.(?:pdf|PDF))"', html)
    if not m:
        print(f"  [no-pdf-link] {title[:50]}"); continue
    pdf_url = "https://www.indiacode.nic.in" + m.group(1)
    s2, b2, _ = fetch(op, pdf_url, timeout=45)
    if s2 == 200 and b2[:4] == b"%PDF":
        confirmed_pdfs[title] = (pdf_url, len(b2))
        print(f"  [OK {len(b2)//1024}KB] {title[:50]}")
        print(f"    -> {pdf_url[-80:]}")
    else:
        print(f"  [{s2} not-pdf {len(b2)}B] {title[:50]}")


# ============================================================================
# PART 3: egazette BNS via Wayback pattern
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 3 — BNS via egazette Wayback")
print("=" * 80)

# BNS was gazetted Dec 25, 2023. egazette numbers close to that date.
# Known number 250883 worked. Try adjacent numbers.
test_nums = ["250880", "250881", "250882", "250883", "250884", "250885",
             "250886", "250887", "250888", "250889", "250890",
             "2023122500", "2023122501", "250901", "250902"]
for n in test_nums:
    u = f"https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/{n}.pdf"
    s, b, _ = fetch(op, u, timeout=30)
    if s == 200 and isinstance(b, bytes) and b[:4] == b"%PDF":
        # inspect content for BNS-ish keywords
        head = b[:8000].lower()
        is_bns = b"bharatiya nyaya sanhita" in head or b"nyaya sanhita" in head
        is_bnss = b"nagarik suraksha sanhita" in head
        is_bsa = b"sakshya adhiniyam" in head
        tag = ""
        if is_bns: tag = "BNS"
        elif is_bnss: tag = "BNSS"
        elif is_bsa: tag = "BSA"
        else: tag = "other"
        print(f"  [OK {len(b)//1024}KB {tag}] egazette {n}")
    else:
        print(f"  [{s}] egazette {n}")


# ============================================================================
# PART 4: Google-indexed PDF mirrors via common legal sites
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 4 — known legal mirrors for BNS/BNSS/BSA")
print("=" * 80)

# Several law portals mirror the gazette.
# Try exact filenames that commonly appear:
MIRRORS = [
    "https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf",
    "https://www.mha.gov.in/sites/default/files/BNS.pdf",
    "https://bprd.nic.in/WriteReadData/userfiles/file/BNS2023.pdf",
    "https://www.ncrb.gov.in/writereaddata/Linked_Docs/BNS.pdf",
    "https://doj.gov.in/wp-content/uploads/2024/02/BNS.pdf",
    "https://doj.gov.in/wp-content/uploads/2024/02/BNSS.pdf",
    "https://doj.gov.in/wp-content/uploads/2024/02/BSA.pdf",
    "https://main.sci.gov.in/pdf/LU/250883.pdf",
    # ICAI legal publications
    "https://resource.cdn.icai.org/87124bos69822cp1.pdf",
]
for u in MIRRORS:
    s, b, _ = fetch(op, u, timeout=30)
    if s == 200 and isinstance(b, bytes) and b[:4] == b"%PDF":
        print(f"  [OK {len(b)//1024}KB] {u}")
    else:
        hint = b[:60].decode("utf-8","ignore").replace("\n"," ") if isinstance(b,bytes) else ""
        print(f"  [{s}] {u[:80]}  hint={hint[:40]}")


# ============================================================================
# PART 5: MHA site deep scrape for BNS PDFs
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 5 — MHA deep scrape")
print("=" * 80)
MHA_DEEP = [
    "https://www.mha.gov.in/en/documents",
    "https://www.mha.gov.in/en/notifications",
    "https://www.mha.gov.in/en/commoncontent/acts-rules",
    "https://www.mha.gov.in/en/documents/acts",
    "https://www.mha.gov.in/en/bharatiya-nyaya-sanhita",
    "https://www.mha.gov.in/en/documents/rules",
    "https://www.mha.gov.in/sites/default/files/",
]
for u in MHA_DEEP:
    s, b, _ = fetch(op, u, timeout=30)
    if s == 200 and isinstance(b, bytes):
        html = b.decode("utf-8","ignore")
        pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
        bns_like = [p for p in pdfs if any(k in p.lower() for k in
                    ["bharatiya","nyaya","sanhita","nagarik","sakshya","bns","bnss","bsa","250883"])]
        print(f"  [{s}] {u}: {len(pdfs)} total pdfs, {len(bns_like)} BNS-like")
        for p in bns_like[:5]:
            print(f"    {urljoin(u, p)[:130]}")
    else:
        print(f"  [{s}] {u}")


# ============================================================================
# PART 6: Final confirmed PDF summary
# ============================================================================
print("\n\n" + "=" * 80)
print("FINAL SUMMARY — confirmed PDFs this round")
print("=" * 80)
for title, (url, size) in confirmed_pdfs.items():
    print(f"  [{size//1024}KB] {title[:60]}  -> {url[-80:]}")

print("\n=== DONE ===")
