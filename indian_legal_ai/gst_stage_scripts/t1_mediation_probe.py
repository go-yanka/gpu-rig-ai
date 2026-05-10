#!/usr/bin/env python3
"""Probe candidate URLs for Mediation Act 2023 (Act 32 of 2023)."""
import requests, urllib3, fitz
urllib3.disable_warnings()

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
HDRS = {"User-Agent": UA}

# Mediation Act 2023 = Act 32 of 2023, assented 15 Sep 2023
CANDIDATES = [
    # indiacode bitstream handle guesses (2023 acts in range 19800-20100)
    "https://www.indiacode.nic.in/bitstream/123456789/19929/1/A2023-32.pdf",
    "https://www.indiacode.nic.in/bitstream/123456789/19930/1/A2023-32.pdf",
    "https://www.indiacode.nic.in/bitstream/123456789/19950/1/A2023-32.pdf",
    "https://www.indiacode.nic.in/bitstream/123456789/19980/1/A2023-32.pdf",
    "https://www.indiacode.nic.in/bitstream/123456789/20000/1/A2023-32.pdf",
    "https://www.indiacode.nic.in/bitstream/123456789/20050/1/A2023-32.pdf",
    "https://www.indiacode.nic.in/bitstream/123456789/19929/1/the_mediation_act%2C_2023.pdf",
    # PRS
    "https://prsindia.org/files/bills_acts/acts_parliament/2023/The_Mediation_Act_2023.pdf",
    "https://prsindia.org/files/bills_acts/acts_parliament/2023/Mediation_Act_2023.pdf",
    # MHA / Law Ministry gazette
    "https://egazette.gov.in/WriteReadData/2023/248676.pdf",
    "https://egazette.gov.in/WriteReadData/2023/249094.pdf",
    "https://egazette.gov.in/WriteReadData/2023/249200.pdf",
    # THC mirror
    "https://thc.nic.in/Central%20Governmental%20Acts/Mediation%20Act,%202023.pdf",
    "https://thc.nic.in/Central%20Governmental%20Acts/The%20Mediation%20Act,%202023.pdf",
    # cdnbbsr CDN pattern (unlikely but cheap)
    "https://cdnbbsr.s3waas.gov.in/s3ec037a4bf9ba2bd774068ad50351fb89/uploads/2023/09/2023091500.pdf",
    # Law Ministry
    "https://legislative.gov.in/sites/default/files/A2023-32.pdf",
    "https://legislative.gov.in/sites/default/files/The%20Mediation%20Act%2C%202023.pdf",
    # incometaxindia (sometimes hosts non-tax acts)
    "https://www.incometaxindia.gov.in/Documents/Acts/Mediation-Act-2023.pdf",
]

for u in CANDIDATES:
    try:
        r = requests.get(u, headers=HDRS, timeout=30, verify=False, allow_redirects=True)
        head = r.content[:4] if r.content else b""
        if r.status_code == 200 and head == b"%PDF":
            try:
                d = fitz.open(stream=r.content)
                pg = d.page_count
                tot = sum(len(d.load_page(i).get_text()) for i in range(pg))
                cpp = tot // max(pg, 1)
                first = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
                d.close()
                has_kw = "mediation" in first.lower()[:5000]
                title = first[:150].replace("\n", " ").strip()
                print(f"OK    {pg:>4}p c/p={cpp:>5} kw={has_kw} | {u}")
                print(f"      title: {title[:100]}")
            except Exception as e:
                print(f"PARSE {u}: {e}")
        else:
            print(f"{r.status_code:<4} {head!r:<10} {u[-70:]}")
    except Exception as e:
        print(f"EXC   {type(e).__name__} {u[-70:]}")
