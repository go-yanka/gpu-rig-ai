#!/usr/bin/env python3
"""
Deep probe for T1 remaining tracks.
For each landing page: fetch HTML, extract all PDF links, report which subset matches the track keywords.
"""
import urllib.request, urllib.error, re, ssl, time, sys
from urllib.parse import urljoin, urlparse

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch(url, timeout=25, binary=False):
    """Fetch with redirect handling; return (status, content, final_url) or None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            data = r.read(2_000_000 if binary else 2_000_000)
            return (r.status, data, r.url)
    except urllib.error.HTTPError as e:
        try:
            body = e.read(2000)
        except Exception:
            body = b""
        return (e.code, body, url)
    except Exception as e:
        return (-1, str(e).encode(), url)


def extract_pdf_links(html_bytes, base_url):
    if isinstance(html_bytes, bytes):
        html = html_bytes.decode("utf-8", "ignore")
    else:
        html = html_bytes
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.I)
    out = []
    for h in hrefs:
        if h.lower().endswith(".pdf") or ".pdf?" in h.lower():
            out.append(urljoin(base_url, h))
    # Dedup while preserving order
    seen = set()
    out2 = []
    for u in out:
        if u not in seen:
            seen.add(u); out2.append(u)
    return out2


def extract_page_links(html_bytes, base_url, same_host=True):
    html = html_bytes.decode("utf-8", "ignore") if isinstance(html_bytes, bytes) else html_bytes
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.I)
    host = urlparse(base_url).netloc
    out = []
    for h in hrefs:
        if h.startswith("#") or h.startswith("mailto:"): continue
        full = urljoin(base_url, h)
        if same_host and urlparse(full).netloc != host: continue
        if full.lower().endswith(".pdf"): continue
        out.append(full)
    seen = set(); out2 = []
    for u in out:
        if u not in seen: seen.add(u); out2.append(u)
    return out2


def match_keywords(url, keywords):
    ul = url.lower()
    return any(k in ul for k in keywords)


def probe_site(name, landing_urls, keywords, max_pages=4, sample_pdfs=8):
    print(f"\n{'='*80}")
    print(f"## {name}")
    print(f"{'='*80}")
    print(f"keywords: {keywords}")
    visited = set()
    all_pdfs = set()
    all_pages = list(landing_urls)
    pages_ok = 0
    while all_pages and len(visited) < max_pages:
        u = all_pages.pop(0)
        if u in visited: continue
        visited.add(u)
        result = fetch(u)
        if result is None or result[0] < 0:
            print(f"  [fail] {u}: {result[1][:100] if result else 'none'}")
            continue
        status, data, final = result
        if status >= 400:
            print(f"  [{status}] {u}")
            continue
        pages_ok += 1
        ctype_marker = "PDF" if data[:4] == b"%PDF" else "HTML"
        print(f"  [{status} {ctype_marker}] {u} ({len(data)}B)")
        if data[:4] == b"%PDF":
            all_pdfs.add(u)
            continue
        pdfs = extract_pdf_links(data, u)
        matched = [p for p in pdfs if match_keywords(p, keywords)]
        print(f"    -> {len(pdfs)} pdfs found, {len(matched)} match keywords")
        for p in matched[:sample_pdfs]:
            all_pdfs.add(p)
            print(f"      + {p[:130]}")
        # enqueue relevant sub-pages (same domain)
        if len(visited) < max_pages - 1:
            sub = extract_page_links(data, u)
            # keep only pages that hint at our track
            sub_match = [s for s in sub if match_keywords(s, keywords)]
            for s in sub_match[:5]:
                if s not in visited and s not in all_pages:
                    all_pages.append(s)
    print(f"\n  SUMMARY: {pages_ok} pages ok, {len(all_pdfs)} candidate PDFs collected")
    return sorted(all_pdfs)


def verify_pdfs(pdfs, limit=5):
    """HEAD-check first few PDFs to confirm they actually serve PDF bytes."""
    print(f"\n  ## verifying first {min(limit, len(pdfs))} candidate PDFs:")
    for u in pdfs[:limit]:
        result = fetch(u)
        if result is None:
            print(f"    [fail] {u}")
            continue
        status, data, _ = result
        if status == 200 and data[:4] == b"%PDF":
            print(f"    [OK PDF {len(data)}B] {u[:120]}")
        elif status in (302, 301, 308):
            print(f"    [REDIRECT {status}] {u[:120]}")
        elif status == 200:
            # Might be HTML error page
            first = data[:80].decode("utf-8", "ignore").replace("\n", " ")
            print(f"    [200 NOT-PDF] {u[:80]}  hint={first[:50]}")
        else:
            print(f"    [{status}] {u[:120]}")


# =============================================================================
# TRACK DEFINITIONS
# =============================================================================

TRACKS = {
    "A1 BNS/BNSS/BSA 2023 (new criminal codes)": {
        "seeds": [
            "https://legislative.gov.in/",
            "https://legislative.gov.in/acts",
            "https://prsindia.org/billtrack",
            "https://prsindia.org/acts-bills",
            "https://www.indiacode.nic.in/handle/123456789/19922",  # BNS on indiacode
            "https://www.mha.gov.in/en",
        ],
        "keywords": ["bharatiya", "nyaya", "sanhita", "nagarik", "suraksha", "sakshya",
                     "bns", "bnss", "bsa", "a2023-45", "a2023-46", "a2023-47"],
    },
    "A2 Income Tax Act 1961": {
        "seeds": [
            "https://incometaxindia.gov.in/",
            "https://incometaxindia.gov.in/pages/acts/income-tax-act.aspx",
            "https://incometaxindia.gov.in/pages/rules/income-tax-rules-1962.aspx",
            "https://incometaxindia.gov.in/Pages/acts.aspx",
            "https://incometaxindia.gov.in/Pages/rules.aspx",
            "https://www.indiacode.nic.in/handle/123456789/2435",
        ],
        "keywords": ["income-tax", "income_tax", "incometax", "it-act",
                     "section-", "chapter-", "itact", "itr1962", "a1961-43", "a2024"],
    },
    "A3 Companies Act 2013 / SEBI / MCA": {
        "seeds": [
            "https://www.mca.gov.in/content/mca/global/en/acts-rules/ebooks/acts.html",
            "https://www.mca.gov.in/MinistryV2/companiesact2013.html",
            "https://www.sebi.gov.in/legal/regulations.html",
            "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes",
            "https://www.icsi.edu/publications/",
            "https://www.indiacode.nic.in/handle/123456789/2114",
        ],
        "keywords": ["companies-act", "companies_act", "companiesact", "a2013-18",
                     "sebi", "lodr", "listing-obligation", "takeover", "insider"],
    },
    "A8 Labour Codes 2019-2020 (Wages/IR/SS/OSH)": {
        "seeds": [
            "https://labour.gov.in/labour-law-reforms",
            "https://labour.gov.in/",
            "https://labour.gov.in/codes-rules",
            "https://labour.gov.in/whatsnew",
            "https://www.indiacode.nic.in/handle/123456789/15225",
        ],
        "keywords": ["wages", "industrial_relation", "industrial-relation",
                     "social_security", "social-security", "osh", "occupational",
                     "code_on", "code-on", "a2019-29", "a2020"],
    },
    "A9 GST circulars (CBIC modern)": {
        "seeds": [
            "https://taxinformation.cbic.gov.in/",
            "https://taxinformation.cbic.gov.in/view-pdf",
            "https://cbic-gst.gov.in/",
            "https://cbic-gst.gov.in/resources.html",
        ],
        "keywords": ["circular", "notification", "notfctn", "cgst", "igst", "rate"],
    },
}

# =============================================================================

print("=" * 80)
print("DEEP PROBE for T1 tracks without confirmed sources")
print("=" * 80)

results = {}
for name, cfg in TRACKS.items():
    pdfs = probe_site(name, cfg["seeds"], cfg["keywords"], max_pages=6, sample_pdfs=12)
    results[name] = pdfs
    if pdfs:
        verify_pdfs(pdfs, limit=4)

print("\n\n")
print("=" * 80)
print("FINAL SUMMARY — confirmed PDFs per track")
print("=" * 80)
for name, pdfs in results.items():
    print(f"\n{name}: {len(pdfs)} candidate PDFs")
    for p in pdfs[:5]:
        print(f"  {p[:140]}")

# Session-based retry for indiacode 302 chain (A6/A7 and others)
print("\n\n" + "=" * 80)
print("A6/A7 indiacode session-handler test")
print("=" * 80)
try:
    import http.cookiejar
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.addheaders = [("User-Agent", UA)]
    urls = [
        ("NI Act 1881", "https://www.indiacode.nic.in/bitstream/123456789/2189/1/AAA1881___26.pdf"),
        ("Contract 1872", "https://www.indiacode.nic.in/bitstream/123456789/2187/1/AAA1872___9.pdf"),
        ("Arbitration 1996", "https://www.indiacode.nic.in/bitstream/123456789/1978/1/A1996-26.pdf"),
        ("CPC 1908", "https://www.indiacode.nic.in/bitstream/123456789/2191/1/AAA1908___5.pdf"),
        ("Specific Relief 1963", "https://www.indiacode.nic.in/bitstream/123456789/1580/1/A1963-47.pdf"),
    ]
    # warm the session with homepage
    try:
        opener.open("https://www.indiacode.nic.in/", timeout=15)
    except Exception as e:
        print(f"  warm-up err: {e}")
    for name, u in urls:
        try:
            r = opener.open(u, timeout=30)
            data = r.read(200_000)
            final = r.url
            if data[:4] == b"%PDF":
                print(f"  [OK PDF {len(data)}B] {name}")
            else:
                first = data[:80].decode("utf-8", "ignore").replace("\n", " ")[:60]
                print(f"  [{r.status} NOT-PDF {len(data)}B] {name} final={final[:80]} hint={first}")
        except Exception as e:
            print(f"  [fail] {name}: {str(e)[:80]}")
except Exception as e:
    print(f"  session test failed: {e}")

# Playwright availability check
print("\n\n" + "=" * 80)
print("Playwright availability on rig")
print("=" * 80)
import subprocess
for cmd in [
    ["python3", "-c", "import playwright; print(playwright.__version__)"],
    ["which", "chromium"],
    ["which", "google-chrome"],
    ["which", "chromium-browser"],
]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print(f"  {' '.join(cmd)} -> stdout={r.stdout.strip()[:80]}  stderr={r.stderr.strip()[:80]}")
    except Exception as e:
        print(f"  {' '.join(cmd)} -> {e}")
