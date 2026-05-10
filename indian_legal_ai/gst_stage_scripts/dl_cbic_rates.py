#!/usr/bin/env python3
"""Download CBIC rate schedules (Notification 1/2017-CT(R), 11/2017, 12/2017) and amendments."""
import urllib.request, re, os, time
from urllib.parse import urljoin

UA = "Mozilla/5.0 (X11; Linux x86_64)"
OUT = "/opt/indian-legal-ai/gst_stage/rates"
os.makedirs(OUT, exist_ok=True)

TARGETS = [
    "https://cbic-gst.gov.in/pdf/central-tax-rate/notfctn-1-2017-cgst-rate-english.pdf",
    "https://cbic-gst.gov.in/pdf/central-tax-rate/notfctn-2-2017-cgst-rate-english.pdf",
    "https://cbic-gst.gov.in/pdf/central-tax-rate/notfctn-11-2017-cgst-rate-english.pdf",
    "https://cbic-gst.gov.in/pdf/central-tax-rate/notfctn-12-2017-cgst-rate-english.pdf",
    "https://cbic-gst.gov.in/pdf/central-tax-rate/notfctn-13-2017-cgst-rate-english.pdf",
    "https://cbic-gst.gov.in/pdf/integrated-tax-rate/notfctn-1-2017-igst-rate-english.pdf",
    "https://cbic-gst.gov.in/pdf/integrated-tax-rate/notfctn-8-2017-igst-rate-english.pdf",
    "https://cbic-gst.gov.in/pdf/integrated-tax-rate/notfctn-9-2017-igst-rate-english.pdf",
    "https://cbic-gst.gov.in/pdf/integrated-tax-rate/notfctn-10-2017-igst-rate-english.pdf",
    "https://cbic-gst.gov.in/gst-goods-services-rates.html",
    "https://cbic-gst.gov.in/hindi/central-tax-rate-notifications.html",
    "https://cbic-gst.gov.in/central-tax-rate-notifications.html",
    "https://cbic-gst.gov.in/integrated-tax-rate-notifications.html",
]

def fetch(u, out_path=None):
    req = urllib.request.Request(u, headers={"User-Agent": UA})
    try:
        data = urllib.request.urlopen(req, timeout=45).read()
    except Exception as e:
        print(f"  miss {u}: {e}")
        return None
    if out_path and len(data) > 8000 and data[:4] == b"%PDF":
        open(out_path, "wb").write(data)
        return "pdf"
    return data

got = 0
for u in TARGETS:
    fn = os.path.basename(u.split("?")[0])
    out = os.path.join(OUT, fn)
    if u.endswith(".pdf"):
        if os.path.exists(out): continue
        r = fetch(u, out)
        if r == "pdf":
            got += 1
            print(f"  ok {fn}")
    else:
        html = fetch(u)
        if isinstance(html, bytes):
            html = html.decode("utf-8", "ignore")
        if not html: continue
        links = set(re.findall(r'href="([^"]+\.pdf)"', html, re.I))
        print(f"  page {u}: {len(links)} pdfs")
        for href in links:
            pdf = urljoin(u, href)
            fn2 = os.path.basename(pdf.split("?")[0])[:80]
            out2 = os.path.join(OUT, fn2)
            if os.path.exists(out2): continue
            r = fetch(pdf, out2)
            if r == "pdf":
                got += 1
                if got % 10 == 0:
                    print(f"  got {got}")
            time.sleep(0.2)

print(f"TOTAL rate PDFs: {got}")
