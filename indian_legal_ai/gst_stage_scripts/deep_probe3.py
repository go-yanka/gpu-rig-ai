#!/usr/bin/env python3
"""
Deep probe round 3:
 1. Inspect indiacode search HTML to fix regex
 2. Brute-force known indiacode bitstream patterns for well-known Acts
 3. Try egazette.gov.in search for BNS
 4. Try direct Wikipedia+mirrors for BNS PDFs
 5. Session-based indiacode bitstream with proper redirect chain
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


def fetch_via(opener, url, timeout=30):
    try:
        r = opener.open(url, timeout=timeout)
        data = r.read(2_500_000)
        return r.status, data, r.url
    except urllib.error.HTTPError as e:
        try:
            body = e.read(2000)
        except Exception:
            body = b""
        return e.code, body, url
    except Exception as e:
        return -1, str(e).encode(), url


# ============================================================================
# PART 1: Inspect indiacode search result HTML
# ============================================================================
print("=" * 80)
print("PART 1 — inspect indiacode simple-search HTML structure")
print("=" * 80)

opener = make_session()
# warm up with homepage
fetch_via(opener, "https://www.indiacode.nic.in/", timeout=20)

search_url = "https://www.indiacode.nic.in/simple-search?query=" + quote_plus("Bharatiya Nyaya Sanhita")
status, body, _ = fetch_via(opener, search_url)
print(f"  status={status}  len={len(body)}")
html = body.decode("utf-8", "ignore")
# Show first few kb with handle occurrences context
import textwrap
# find all handle hrefs with surrounding 100 chars
for m in re.finditer(r'.{80}/handle/\d+/\d+.{80}', html):
    print(f"  snippet: {m.group(0)[:250]}")
print()
# Try alternate regexes
alt_re = [
    (r'href="(/handle/[^"]+)"[^>]*>\s*([^<]+?)\s*<', "href(handle)>title<"),
    (r'href=\'(/handle/[^\']+)\'[^>]*>\s*([^<]+?)\s*<', "single-quoted"),
    (r"'(/handle/\d+/\d+)'", "single-quote just href"),
    (r'"(/handle/\d+/\d+)"', "double-quote just href"),
]
for rx, label in alt_re:
    hits = re.findall(rx, html)
    print(f"  regex [{label}]: {len(hits)} hits")
    for h in hits[:3]:
        print(f"    {h}")


# ============================================================================
# PART 2: Brute-force known indiacode bitstream patterns
# Many central acts live at /bitstream/123456789/<N>/1/a<YEAR>-<NUM>.pdf
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 2 — indiacode direct handle + bitstream via session")
print("=" * 80)

# Try visiting /handle/123456789/N for small N range to find catalog
# (too slow for brute force; instead use known-good patterns from prior probe)
KNOWN_HANDLES = {
    "Income Tax 1961":   "/handle/123456789/2435",
    "Companies 2013":    "/handle/123456789/2114",
    "Constitution":      "/handle/123456789/15240",
    "NI Act 1881":       "/handle/123456789/2189",  # verified earlier at bitstream level
    "Contract 1872":     "/handle/123456789/2187",
    "Arbitration 1996":  "/handle/123456789/1978",
    "Specific Relief 1963": "/handle/123456789/1580",
    "Partnership 1932":  "/handle/123456789/2192",
    "CPC 1908":          "/handle/123456789/2191",  # maybe wrong
    "SoGA 1930":         "/handle/123456789/2191",  # maybe wrong
    "Evidence 1872":     None,  # unknown
    "CrPC 1973":         None,
    "TP Act 1882":       None,
}

print("  Visiting each handle, then its bitstream PDF:")
opener2 = make_session()
fetch_via(opener2, "https://www.indiacode.nic.in/", timeout=15)

for name, h in KNOWN_HANDLES.items():
    if not h: continue
    u = "https://www.indiacode.nic.in" + h
    status, body, _ = fetch_via(opener2, u)
    if status != 200:
        print(f"  [{status}] {name}: handle failed")
        continue
    html = body.decode("utf-8", "ignore")
    # Find PDF bitstream
    m = re.search(r'href="(/bitstream/[^"]+\.(?:pdf|PDF))"', html)
    if not m:
        print(f"  [no-pdf] {name}")
        continue
    pdf_url = "https://www.indiacode.nic.in" + m.group(1)
    s2, b2, _ = fetch_via(opener2, pdf_url)
    if s2 == 200 and b2[:4] == b"%PDF":
        print(f"  [OK PDF {len(b2)}B] {name}  -> {pdf_url[-70:]}")
    else:
        print(f"  [{s2} not-pdf] {name}  -> {pdf_url[-70:]}")


# ============================================================================
# PART 3: Try indiacode year-browse for 2023 Central Acts (BNS/BNSS/BSA)
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 3 — indiacode browse Central Acts by year 2023")
print("=" * 80)

BROWSE_URLS = [
    "https://www.indiacode.nic.in/handle/123456789/1362/browse?type=dateissued",
    "https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query=2023",
    "https://www.indiacode.nic.in/handle/123456789/1362",
    "https://www.indiacode.nic.in/handle/123456789/1362/discover?filtertype=dateIssued&filter_relational_operator=equals&filter=2023",
    "https://www.indiacode.nic.in/search?query=bharatiya",
]
opener3 = make_session()
fetch_via(opener3, "https://www.indiacode.nic.in/", timeout=15)
for u in BROWSE_URLS:
    status, body, _ = fetch_via(opener3, u, timeout=40)
    html = body.decode("utf-8", "ignore") if isinstance(body, bytes) else str(body)
    hits = re.findall(r'(/handle/123456789/\d+)[^>]*>\s*([^<]{5,120})<', html, re.I)
    # look specifically for 2023 act titles
    bns_hits = [(h, t.strip()) for h, t in hits if any(k in t.lower() for k in
                ["bharatiya","nyaya","sanhita","nagarik","suraksha","sakshya","adhiniyam"])]
    print(f"  [{status}] {u[:80]}: {len(hits)} total hits, {len(bns_hits)} BNS-like")
    for h, t in bns_hits[:8]:
        print(f"    {h}  -> {t[:80]}")


# ============================================================================
# PART 4: Try alternate BNS sources
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 4 — alternate BNS sources")
print("=" * 80)

ALT_BNS = [
    # Direct gazette CDN patterns (many 2023 Central Acts live here)
    "https://egazette.gov.in/",
    # Bureau of Police Research & Development
    "https://bprd.nic.in/",
    # Criminal law reform committee page
    "https://criminallawreforms.in/",
    # Law Commission of India
    "https://lawcommissionofindia.nic.in/",
    # Supreme Court "e-SCR" or similar act PDFs
    "https://main.sci.gov.in/",
    # National Judicial Academy sometimes has bare texts
    "https://nja.nic.in/",
    # Internet Archive archived PRS PDFs
    "https://web.archive.org/web/2024*/prsindia.org/files/bills_acts/acts_parliament/2023/*",
]
for u in ALT_BNS:
    status, body, _ = fetch_via(opener, u, timeout=25)
    if status == 200:
        html = body.decode("utf-8", "ignore") if isinstance(body, bytes) else ""
        hits = re.findall(r'href=["\']([^"\']+\.pdf)["\'][^>]*>[^<]*(?:bharatiya|nyaya|sanhita|nagarik|sakshya|BNS|BNSS|BSA)[^<]*',
                          html, re.I)
        raw = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
        bns_like = [h for h in raw if any(k in h.lower() for k in
                    ["bharatiya","nyaya","sanhita","nagarik","suraksha","sakshya","bns","bnss","bsa"])]
        print(f"  [{status}] {u}: {len(raw)} total pdfs, {len(bns_like)} BNS-like")
        for h in bns_like[:3]:
            print(f"    {urljoin(u, h)[:130]}")
    else:
        print(f"  [{status}] {u}")

# Also try: Internet Archive wayback for known BNS URLs
print("\n  Wayback Machine search for BNS PDFs:")
for wb in [
    "https://web.archive.org/web/2024/https://prsindia.org/files/bills_acts/acts_parliament/2023/The-Bharatiya-Nyaya-Sanhita,-2023.pdf",
    "https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/250883.pdf",
]:
    status, body, _ = fetch_via(opener, wb, timeout=40)
    is_pdf = isinstance(body, bytes) and body[:4] == b"%PDF"
    print(f"    [{status} {'PDF' if is_pdf else 'not-pdf'}] {wb[:110]}")


# ============================================================================
# PART 5: PRS India actual structure (landing page link hunt)
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 5 — PRS India structure hunt")
print("=" * 80)

PRS = [
    "https://prsindia.org/billtrack",
    "https://prsindia.org/billtrack?title=bharatiya",
    "https://prsindia.org/billtrack/bharatiya-nyaya-sanhita-2023",
    "https://prsindia.org/billtrack/bharatiya-nagarik-suraksha-sanhita-2023",
    "https://prsindia.org/billtrack/bharatiya-sakshya-bill-2023",
    "https://prsindia.org/acts-parliament",
]
for u in PRS:
    status, body, _ = fetch_via(opener, u, timeout=30)
    if status == 200:
        html = body.decode("utf-8", "ignore")
        pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
        bns_like = [h for h in pdfs if any(k in h.lower() for k in
                    ["bharatiya","nyaya","sanhita","nagarik","suraksha","sakshya"])]
        print(f"  [{status}] {u}: {len(pdfs)} pdfs, {len(bns_like)} BNS-like")
        for p in bns_like[:5]:
            print(f"    {urljoin(u, p)[:130]}")
        # Also show any page linking to a 2023 Act page
        page_links = re.findall(r'href=["\']([^"\']*(?:bharatiya|nyaya|sanhita|nagarik|suraksha|sakshya|2023)[^"\']*)["\'][^>]*>[^<]*(?:Bharatiya|BNS|BNSS|BSA|2023)',
                               html, re.I)
        if page_links:
            print(f"    also {len(page_links)} related page links:")
            for pl in page_links[:5]:
                print(f"      {urljoin(u, pl)[:130]}")
    else:
        print(f"  [{status}] {u}")


# ============================================================================
# PART 6: Law Ministry Legislative Department actual structure
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 6 — Legislative Dept structure hunt")
print("=" * 80)

LEG = [
    "https://legislative.gov.in/",
    "https://legislative.gov.in/acts-of-parliament-from-1838-to-2019",
    "https://legislative.gov.in/central-acts-unrepealed-part-i/",
    "https://legislative.gov.in/sites/default/files/",
    # Direct URL guesses based on common patterns
    "https://legislative.gov.in/sites/default/files/A1881-26.pdf",
    "https://legislative.gov.in/sites/default/files/A1872-9.pdf",
]
for u in LEG:
    status, body, _ = fetch_via(opener, u, timeout=30)
    if status == 200 and isinstance(body, bytes) and body[:4] == b"%PDF":
        print(f"  [OK PDF {len(body)}B] {u}")
    elif status == 200:
        html = body.decode("utf-8", "ignore") if isinstance(body, bytes) else ""
        pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
        print(f"  [{status}] {u}: {len(pdfs)} pdfs")
        # sample
        for p in pdfs[:5]:
            print(f"    {urljoin(u, p)[:130]}")
    else:
        print(f"  [{status}] {u}")

print("\n=== DONE ===")
