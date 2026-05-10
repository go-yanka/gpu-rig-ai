#!/usr/bin/env python3
"""Download State SGST Acts PDFs for top states."""
import urllib.request, os, re
from urllib.parse import urljoin

UA = "Mozilla/5.0 (X11; Linux x86_64)"
OUT = "/opt/indian-legal-ai/gst_stage/state_sgst"
os.makedirs(OUT, exist_ok=True)

TARGETS = {
    "Maharashtra_SGST_Act_2017.pdf": "https://www.mahagst.gov.in/sites/default/files/act/MGST_Act.pdf",
    "Karnataka_SGST_Act_2017.pdf":   "https://gst.kar.nic.in/Documents/KGSTAct2017.pdf",
    "Tamil_Nadu_SGST_Act_2017.pdf":  "https://ctd.tn.gov.in/documents/20143/55552/TN+GST+Act+2017+-+English.pdf",
    "Delhi_SGST_Act_2017.pdf":       "https://dvat.gov.in/website/DVAT_portal/files/pdf/DGST_Act_2017.pdf",
    "Gujarat_SGST_Act_2017.pdf":     "https://commercialtax.gujarat.gov.in/vatwebsite/download/acts/GGST_Act_2017.pdf",
    "UP_SGST_Act_2017.pdf":          "https://comtax.up.nic.in/GST/act/UPGST_Act.pdf",
    "WB_SGST_Act_2017.pdf":          "https://wbcomtax.gov.in/gst/WBGST_Act_2017.pdf",
    "Telangana_SGST_Act_2017.pdf":   "https://tgct.gov.in/tgportal/GST/TGGST_Act_2017.pdf",
}

got = 0
for fn, url in TARGETS.items():
    out = os.path.join(OUT, fn)
    if os.path.exists(out): continue
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        data = urllib.request.urlopen(req, timeout=45).read()
        if data[:4] == b"%PDF" and len(data) > 20000:
            open(out, "wb").write(data)
            got += 1
            print(f"  ok {fn}  {len(data)//1024}KB")
        else:
            print(f"  not-pdf {fn} ({len(data)}B)")
    except Exception as e:
        print(f"  fail {fn}: {e}")

# indiacode fallback search pages for states without direct PDFs
IC = [
    ("Rajasthan", "https://www.indiacode.nic.in/handle/123456789/15352?view_type=browse&sam_handle=123456789/1362"),
    ("MadhyaPradesh", "https://www.indiacode.nic.in/handle/123456789/4300"),
]
for name, page in IC:
    if any(f.startswith(name) for f in os.listdir(OUT)): continue
    try:
        req = urllib.request.Request(page, headers={"User-Agent": UA})
        html = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "ignore")
        m = re.search(r'href="([^"]+\.pdf)"', html)
        if m:
            pdf_url = urljoin(page, m.group(1))
            req2 = urllib.request.Request(pdf_url, headers={"User-Agent": UA})
            data = urllib.request.urlopen(req2, timeout=45).read()
            if data[:4] == b"%PDF":
                out = os.path.join(OUT, f"{name}_SGST_Act_indiacode.pdf")
                open(out, "wb").write(data)
                got += 1
                print(f"  ok fallback {name}")
    except Exception as e:
        print(f"  fallback fail {name}: {e}")

print(f"TOTAL state PDFs: {got}")
