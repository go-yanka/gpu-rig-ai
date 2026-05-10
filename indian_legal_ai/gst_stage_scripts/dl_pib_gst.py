#!/usr/bin/env python3
"""Scrape PIB + CBIC for GST-tagged circular/notification PDFs."""
import urllib.request, os, re, time
from urllib.parse import urljoin

UA = "Mozilla/5.0 (X11; Linux x86_64)"
OUT = "/opt/indian-legal-ai/gst_stage/cbic_circulars"
os.makedirs(OUT, exist_ok=True)

SEEDS = [
    "https://cbic-gst.gov.in/cgst-circulars.html",
    "https://cbic-gst.gov.in/central-tax-notfns.html",
    "https://cbic-gst.gov.in/integrated-tax-notfns.html",
    "https://cbic-gst.gov.in/cbic-notfns.html",
    "https://cbic-gst.gov.in/hindi/cgst-circulars.html",
    "https://pib.gov.in/indexd.aspx",
]

visited = set()
q = list(SEEDS)
got = 0

def fetch(u):
    req = urllib.request.Request(u, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=25).read()

while q and len(visited) < 40:
    u = q.pop(0)
    if u in visited: continue
    visited.add(u)
    try:
        html = fetch(u).decode("utf-8", "ignore")
    except Exception as e:
        print(f"  page {u}: {e}")
        continue
    for href in re.findall(r'href="([^"]+\.pdf)"', html, re.I):
        pdf = urljoin(u, href)
        low = pdf.lower()
        if not any(k in low for k in ["gst", "cgst", "igst", "circular", "notification", "notif"]):
            continue
        fn = os.path.basename(pdf.split("?")[0])[:80]
        out = os.path.join(OUT, fn)
        if os.path.exists(out): continue
        try:
            req = urllib.request.Request(pdf, headers={"User-Agent": UA})
            d = urllib.request.urlopen(req, timeout=40).read()
            if d[:4] == b"%PDF" and len(d) > 8000:
                open(out, "wb").write(d)
                got += 1
                if got % 5 == 0:
                    print(f"  got {got}  last={fn}")
        except Exception:
            pass
        time.sleep(0.15)

print(f"TOTAL circular/notification PDFs: {got}")
