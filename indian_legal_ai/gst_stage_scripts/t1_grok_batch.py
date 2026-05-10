#!/usr/bin/env python3
"""Batch-fetch Grok's 18 URLs, verify with fitz, place in staging."""
import requests, urllib3, fitz, os
urllib3.disable_warnings()

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
STAGE = "/opt/indian-legal-ai/gst_stage"
HDRS = {"User-Agent": UA,
        "Accept": "application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"}

# (label, primary_url, title_keyword, dest_rel, fallback_urls)
JOBS = [
    ("income_tax_1961",
     "https://www.indiacode.nic.in/bitstream/123456789/2435/1/a1961-43.pdf",
     "income-tax act, 1961",
     "t1_income_tax/income_tax_act_1961.pdf", []),
    ("companies_2013",
     "https://www.indiacode.nic.in/bitstream/123456789/2114/5/A2013-18.pdf",
     "companies act, 2013",
     "t1_companies_sebi/companies_act_2013.pdf",
     ["https://nclt.gov.in/sites/default/files/Act%26rules/the_companies_act_2013_0.pdf"]),
    ("igst_2017",
     "https://cbic-gst.gov.in/aces/Documents/IGST-bill-e.pdf",
     "integrated goods and services tax",
     "t1_gst_circulars/IGST_Act_2017.pdf", []),
    ("utgst_2017",
     "https://cbic-gst.gov.in/aces/Documents/UTGST-bill-e.pdf",
     "union territory goods and services tax",
     "t1_gst_circulars/UTGST_Act_2017.pdf", []),
    ("fema_1999",
     "https://ifsca.gov.in/Document/Legal/63-the-foreign-exchange-management-act-199917092020075653.pdf",
     "foreign exchange management act, 1999",
     "t1_fema_rbi/FEMA_Act_1999.pdf", []),
    ("pmla_2002",
     "https://dor.gov.in/files/acts_files/PMLA_2002.pdf",
     "prevention of money",
     "t1_other_bare_acts/pmla_2002.pdf", []),
    ("sarfaesi_2002",
     "https://home.wb.gov.in/public/assets/frontend/pdf/securitisation_act.pdf",
     "securitisation",
     "t1_other_bare_acts/sarfaesi_2002.pdf", []),
    ("sale_of_goods_1930",
     "https://www.indiacode.nic.in/bitstream/123456789/2390/1/193003.pdf",
     "sale of goods act",
     "t1_commercial_acts/sale_of_goods_1930.pdf", []),
    ("specific_relief_1963",
     "https://www.indiacode.nic.in/bitstream/123456789/1583/7/A1963-47.pdf",
     "specific relief act",
     "t1_commercial_acts/specific_relief_1963.pdf", []),
    ("tpa_1882",
     "https://www.indiacode.nic.in/bitstream/123456789/2338/1/A1882-04.pdf",
     "transfer of property act",
     "t1_other_bare_acts/tpa_1882.pdf", []),
    ("hma_1955",
     "https://highcourtchd.gov.in/hclscc/subpages/pdf_files/4.pdf",
     "hindu marriage act",
     "t1_other_bare_acts/hma_1955.pdf", []),
    ("consumer_2019",
     "https://www.indiacode.nic.in/bitstream/123456789/16939/1/a2019-35.pdf",
     "consumer protection act, 2019",
     "t1_other_bare_acts/consumer_protection_2019.pdf",
     ["https://static.investindia.gov.in/s3fs-public/2019-08/Consumer%20protection%20act%202019.pdf"]),
    ("mv_1988",
     "https://mvd.kerala.gov.in/sites/default/files/Downloads/Motor%20Vehicles%20Act%2C%201988.pdf",
     "motor vehicles act",
     "t1_other_bare_acts/mv_act_1988.pdf", []),
    ("bnss_2023",
     "https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf",
     "nagarik suraksha sanhita",
     "t1_criminal_codes_2023/BNSS_2023.pdf", []),
    ("dv_2005",
     "https://www.indiacode.nic.in/bitstream/123456789/15436/1/protection_of_women_from_domestic_violence_act%2C_2005.pdf",
     "domestic violence",
     "t1_other_bare_acts/dv_act_2005.pdf", []),
    ("juvenile_2015",
     "https://cara.wcd.gov.in/pdf/jj%20act%202015.pdf",
     "juvenile justice",
     "t1_other_bare_acts/juvenile_justice_2015.pdf", []),
    ("arbitration_1996",
     "https://www.indiacode.nic.in/bitstream/123456789/21922/1/the_arbitration_and_conciliation_act%2C_1996_act_no._26_of_1996.pdf",
     "arbitration and conciliation",
     "t1_commercial_acts/arbitration_1996.pdf",
     ["https://sclsc.gov.in/theme/front/pdf/ACTS%20FINAL/THE%20ARBITRATION%20AND%20CONCILIATION%20ACT,%201996.pdf"]),
]

BAD = ["amendment bill","written answer","jstor","research paper","papers laid",
       "digital copy of a book","cornell university"]

results = []
os.makedirs("/tmp/grok_hits", exist_ok=True)

for label, url, kw, rel, fallbacks in JOBS:
    urls_to_try = [url] + fallbacks
    ok = False
    for u in urls_to_try:
        try:
            r = requests.get(u, headers=HDRS, timeout=30, verify=False,
                             allow_redirects=True)
        except Exception as e:
            print(f"  [{label}] EXC {type(e).__name__}: {u[:70]}")
            continue
        if r.status_code != 200:
            print(f"  [{label}] HTTP {r.status_code}: {u[:70]}")
            continue
        if r.content[:4] != b"%PDF":
            print(f"  [{label}] NOT-PDF (starts {r.content[:20]!r}): {u[:70]}")
            continue
        # parse
        try:
            d = fitz.open(stream=r.content, filetype="pdf")
            pg = d.page_count
            t = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
            d.close()
        except Exception as e:
            print(f"  [{label}] PARSE-ERR: {e}")
            continue
        tl = t.lower()[:8000]
        title = t[:160].replace("\n"," ").strip()
        if kw not in tl:
            print(f"  [{label}] KW MISS ('{kw}' not in text): title={title[:90]}")
            continue
        if any(b in tl for b in BAD):
            print(f"  [{label}] BAD-MARKER in text")
            continue
        # save both raw and to dest
        raw = f"/tmp/grok_hits/{label}.pdf"
        with open(raw, "wb") as f: f.write(r.content)
        dest = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f: f.write(r.content)
        sz = len(r.content)//1024
        print(f"  [{label}] SAVED {sz}KB p={pg} -> {rel}")
        print(f"         title: {title[:110]}")
        results.append((label, "OK", sz, pg, rel, title[:110]))
        ok = True
        break
    if not ok:
        results.append((label, "FAIL", 0, 0, rel, ""))

print("\n" + "="*70)
print("=== SUMMARY ===")
ok_count = sum(1 for r in results if r[1]=="OK")
print(f"Success: {ok_count}/{len(JOBS)}")
print("\nFailed:")
for r in results:
    if r[1] != "OK": print(f"  - {r[0]}")

print("\n=== Final staging ===")
tp=0;tk=0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"): continue
    pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
    kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
    if pdfs:
        tp+=len(pdfs); tk+=kb
        print(f"  {d:<32} {len(pdfs):>3} pdfs  {kb:>7} KB")
print(f"TOTAL: {tp} pdfs, {tk/1024:.1f} MB")
