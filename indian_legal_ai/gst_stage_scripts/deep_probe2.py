#!/usr/bin/env python3
"""
Deep probe round 2:
 1. Find indiacode handle IDs for A1 (BNS/BNSS/BSA), A6/A7 bare acts, A8 labour codes
 2. Full PDF extraction from cbic-gst.gov.in (A9 circulars)
 3. MHA + eGazette search for BNS/BNSS/BSA
 4. ICSI publications full listing (for A3 Companies commentary)
"""
import urllib.request, urllib.error, re, ssl, time
from urllib.parse import urljoin, urlparse, quote_plus

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch(url, timeout=30):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.status, r.read(2_500_000), r.url
    except urllib.error.HTTPError as e:
        return e.code, e.read(2000), url
    except Exception as e:
        return -1, str(e).encode(), url


# ============================================================================
# PART 1: indiacode search for specific acts by name (finds handle IDs)
# ============================================================================
print("=" * 80)
print("PART 1 — indiacode search for specific act handle IDs")
print("=" * 80)

ACT_QUERIES = [
    # A1 - New criminal codes 2023
    "Bharatiya Nyaya Sanhita",
    "Bharatiya Nagarik Suraksha Sanhita",
    "Bharatiya Sakshya Adhiniyam",
    # A6/A7 - Commercial bare acts
    "Negotiable Instruments",
    "Indian Contract Act",
    "Sale of Goods Act",
    "Indian Partnership Act",
    "Specific Relief Act",
    "Arbitration and Conciliation",
    "Code of Civil Procedure",
    "Indian Evidence Act",
    "Transfer of Property Act",
    # A8 - Labour codes
    "Code on Wages",
    "Industrial Relations Code",
    "Code on Social Security",
    "Occupational Safety Health",
    # Criminal procedure
    "Code of Criminal Procedure",
    # Constitution
    "Constitution of India",
]

handles_found = {}
for q in ACT_QUERIES:
    # indiacode search URL (returns HTML with handle links)
    search_url = f"https://www.indiacode.nic.in/simple-search?query={quote_plus(q)}"
    code, body, _ = fetch(search_url)
    if code != 200:
        print(f"  [{code}] search FAIL for '{q}'")
        continue
    html = body.decode("utf-8", "ignore")
    # handle pattern: /handle/123456789/NNNNN
    handle_refs = re.findall(r'href="(/handle/123456789/\d+)"[^>]*>([^<]+)', html)
    # filter matches where title contains query keywords (case-insensitive)
    qwords = [w.lower() for w in q.split() if len(w) > 3]
    matched = []
    for href, title in handle_refs:
        tl = title.lower()
        if all(w in tl for w in qwords[:2]):  # at least first 2 query words must match
            matched.append((href, title.strip()[:60]))
    # dedupe
    seen = set()
    uniq = []
    for h, t in matched:
        if h not in seen:
            seen.add(h)
            uniq.append((h, t))
    print(f"\n'{q}': {len(uniq)} handle matches")
    for h, t in uniq[:3]:
        full = urljoin("https://www.indiacode.nic.in/", h)
        handles_found[t] = full
        print(f"  {full}  -> {t}")

# ============================================================================
# PART 2: fetch each handle page and extract the PDF link
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 2 — extract PDF link from each handle page")
print("=" * 80)

pdfs_confirmed = {}
for title, handle_url in handles_found.items():
    code, body, _ = fetch(handle_url)
    if code != 200:
        print(f"  [{code}] {title}: handle failed")
        continue
    html = body.decode("utf-8", "ignore")
    # Look for bitstream PDF links
    m = re.findall(r'href="(/bitstream/[^"]+\.pdf)"', html)
    if not m:
        print(f"  [no-pdf] {title}")
        continue
    pdf_url = urljoin("https://www.indiacode.nic.in/", m[0])
    # Verify the PDF is reachable
    code2, body2, _ = fetch(pdf_url)
    if code2 == 200 and body2[:4] == b"%PDF":
        pdfs_confirmed[title] = pdf_url
        print(f"  [OK {len(body2)}B] {title}  -> {pdf_url[:100]}")
    else:
        print(f"  [{code2} not-pdf] {title}  -> {pdf_url[:100]}")


# ============================================================================
# PART 3: full cbic-gst.gov.in PDF extraction (A9 circulars + notifications)
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 3 — cbic-gst.gov.in full GST PDF harvest")
print("=" * 80)

CBIC_PAGES = [
    "https://cbic-gst.gov.in/",
    "https://cbic-gst.gov.in/gst-goods-services-rates.html",
    "https://cbic-gst.gov.in/vacancy-circulars.html",
    "https://cbic-gst.gov.in/what-new.html",
    "https://cbic-gst.gov.in/hindi/gst-circulars.html",
    "https://cbic-gst.gov.in/Faqs.html",
]

all_pdfs = set()
for p in CBIC_PAGES:
    code, body, _ = fetch(p)
    if code != 200:
        print(f"  [{code}] {p}")
        continue
    html = body.decode("utf-8", "ignore")
    pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
    full_pdfs = [urljoin(p, pu) for pu in pdfs]
    # Keep only GST-relevant
    keep = [u for u in full_pdfs if any(k in u.lower() for k in
            ["circular", "notfctn", "notification", "gst", "cgst", "igst", "rate", "rule"])]
    before = len(all_pdfs)
    for u in keep:
        all_pdfs.add(u)
    print(f"  [{code}] {p}: +{len(all_pdfs) - before} new (total {len(all_pdfs)})")

# Classify by prefix pattern
buckets = {"circular": [], "notification": [], "rate": [], "rule": [], "other": []}
for u in all_pdfs:
    ul = u.lower()
    if "circular" in ul: buckets["circular"].append(u)
    elif "notfctn" in ul or "notification" in ul: buckets["notification"].append(u)
    elif "rate" in ul: buckets["rate"].append(u)
    elif "rule" in ul: buckets["rule"].append(u)
    else: buckets["other"].append(u)

print(f"\n  TOTAL GST PDFs on cbic-gst.gov.in: {len(all_pdfs)}")
for b, us in buckets.items():
    print(f"    {b}: {len(us)}")
    for u in us[:3]:
        print(f"      {u[:120]}")


# ============================================================================
# PART 4: MHA + eGazette for BNS/BNSS/BSA official gazette PDFs
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 4 — MHA + eGazette search for BNS/BNSS/BSA")
print("=" * 80)

MHA_PAGES = [
    "https://www.mha.gov.in/en",
    "https://www.mha.gov.in/en/notifications",
    "https://www.mha.gov.in/en/acts",
    "https://www.mha.gov.in/en/commoncontent/acts-rules",
    "https://www.mha.gov.in/en/divisionofmha/jc-division",
]
for p in MHA_PAGES:
    code, body, _ = fetch(p)
    if code != 200:
        print(f"  [{code}] {p}")
        continue
    html = body.decode("utf-8", "ignore")
    # Look for BNS/BNSS/BSA
    hits = re.findall(r'href=["\']([^"\']+)["\'][^>]*>([^<]{0,80}(?:bharatiya|sanhita|adhiniyam|nyaya|nagarik|sakshya|BNS|BNSS|BSA)[^<]{0,80})', html, re.I)
    print(f"  [{code}] {p}: {len(hits)} BNS-related links")
    for href, label in hits[:10]:
        full = urljoin(p, href)
        print(f"    -> {label.strip()[:80]}")
        print(f"       {full[:120]}")


# ============================================================================
# PART 5: ICSI publications full extraction (A3 commentary)
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 5 — ICSI publications harvest")
print("=" * 80)

ICSI_PAGES = [
    "https://www.icsi.edu/publications/",
    "https://www.icsi.edu/publications/study-material/",
    "https://www.icsi.edu/publications/books/",
]
for p in ICSI_PAGES:
    code, body, _ = fetch(p)
    if code != 200:
        print(f"  [{code}] {p}")
        continue
    html = body.decode("utf-8", "ignore")
    pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
    full_pdfs = [urljoin(p, pu) for pu in pdfs]
    print(f"  [{code}] {p}: {len(full_pdfs)} total PDFs")
    # classify
    company_pdfs = [u for u in full_pdfs if any(k in u.lower() for k in
                    ["compan", "corp", "sebi", "insolven", "fema", "secretar"])]
    print(f"    corporate-law pdfs: {len(company_pdfs)}")
    for u in company_pdfs[:5]:
        print(f"      {u[:120]}")


# ============================================================================
# PART 6: RBI Master Directions page - extract all PDFs
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 6 — RBI Master Directions harvest (A5 FEMA)")
print("=" * 80)

RBI_PAGES = [
    "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx",
    "https://www.rbi.org.in/Scripts/BS_FemaNotifications.aspx",
    "https://rbi.org.in/Scripts/Bs_viewcontent.aspx?Id=3233",  # FEMA rules overview
]
for p in RBI_PAGES:
    code, body, _ = fetch(p)
    if code != 200:
        print(f"  [{code}] {p}")
        continue
    html = body.decode("utf-8", "ignore")
    pdfs = re.findall(r'href=["\']([^"\']+\.(?:pdf|PDF))["\']', html)
    full_pdfs = [urljoin(p, pu) for pu in pdfs]
    # FEMA / master direction filter
    fema_pdfs = [u for u in full_pdfs if any(k in u.lower() for k in
                 ["fema", "master", "direction", "foreign"])]
    print(f"  [{code}] {p}: {len(full_pdfs)} total, {len(fema_pdfs)} FEMA-relevant")
    for u in fema_pdfs[:5]:
        print(f"      {u[:120]}")


# ============================================================================
# PART 7: Taxmann/cleartax/legal repositories for BNS
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 7 — alternate sources for BNS/BNSS/BSA")
print("=" * 80)

ALT_PAGES = [
    "https://lddashboard.legislative.gov.in/actsofparliamentfromtheyear?type=central&year=2023",
    "https://lddashboard.legislative.gov.in/",
    "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/12/2023122595.pdf",  # BNS on CDN
    "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/12/2023122592.pdf",  # BNSS
    "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/12/2023122593.pdf",  # BSA
]
for p in ALT_PAGES:
    code, body, _ = fetch(p)
    if code == 200 and body[:4] == b"%PDF":
        print(f"  [OK PDF {len(body)}B] {p}")
    elif code == 200:
        # HTML - scan for BNS links
        html = body.decode("utf-8", "ignore")
        hits = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.I)
        bns_like = [h for h in hits if any(k in h.lower() for k in ["nyaya","nagarik","sakshya","bharat","2023122","A2023"])]
        print(f"  [{code} HTML] {p}: {len(hits)} pdfs, {len(bns_like)} BNS-like")
        for u in bns_like[:5]:
            full = urljoin(p, u)
            print(f"    {full[:130]}")
    else:
        print(f"  [{code}] {p}: {body[:80] if isinstance(body, bytes) else body[:80]}")


# ============================================================================
# PART 8: egazette.nic.in direct search
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 8 — egazette / CDN searches")
print("=" * 80)

for p in ["https://egazette.gov.in/", "https://egazette.nic.in/",
          "https://www.egazette.gov.in/"]:
    code, body, _ = fetch(p)
    print(f"  [{code}] {p}")

print("\n\n=== DONE ===")
