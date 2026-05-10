#!/usr/bin/env python3
"""Mass-scan cdnbbsr.s3waas.gov.in for bare acts."""
import requests, urllib3, fitz, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
urllib3.disable_warnings()
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"

def make_session():
    s = requests.Session(); s.headers.update({"User-Agent": UA}); s.verify = False
    return s

STAGE = "/opt/indian-legal-ai/gst_stage"
HASH = "s3ec037a4bf9ba2bd774068ad50351fb89"
# We know 060260-060288 range has acts. Expand massively.

# Target acts to find
TARGETS = {
    "ipc": (["indian penal code"],                       "t1_old_criminal/ipc_1860.pdf"),
    "crpc": (["criminal procedure, 1973","criminal procedure,1973"], "t1_old_criminal/crpc_1973.pdf"),
    "evidence": (["indian evidence act"],                 "t1_old_criminal/evidence_act_1872.pdf"),
    "companies2013": (["companies act, 2013"],            "t1_companies_sebi/companies_act_2013.pdf"),
    "income_tax": (["income-tax act, 1961","income tax act, 1961"], "t1_income_tax/income_tax_act_1961.pdf"),
    "sale_of_goods": (["sale of goods act"],              "t1_commercial_acts/sale_of_goods_1930.pdf"),
    "specific_relief": (["specific relief act"],          "t1_commercial_acts/specific_relief_1963.pdf"),
    "arbitration": (["arbitration and conciliation"],     "t1_commercial_acts/arbitration_1996.pdf"),
    "tpa": (["transfer of property act"],                 "t1_other_bare_acts/tpa_1882.pdf"),
    "consumer2019": (["consumer protection act, 2019"],   "t1_other_bare_acts/consumer_protection_2019.pdf"),
    "mv1988": (["motor vehicles act"],                    "t1_other_bare_acts/mv_act_1988.pdf"),
    "hma": (["hindu marriage act"],                       "t1_other_bare_acts/hma_1955.pdf"),
    "bnss": (["nagarik suraksha sanhita"],                "t1_criminal_codes_2023/BNSS_2023.pdf"),
    "igst": (["integrated goods and services tax act, 2017"], "t1_gst_circulars/IGST_Act.pdf"),
    "utgst": (["union territory goods and services"],    "t1_gst_circulars/UTGST_Act.pdf"),
    "fema1999": (["foreign exchange management act"],     "t1_fema_rbi/FEMA_1999.pdf"),
    "pmla": (["prevention of money laundering"],          "t1_other_bare_acts/pmla_2002.pdf"),
    "cgst": (["central goods and services tax act, 2017"],None),  # already have
    "contract": (["indian contract act"], None),
    "ni": (["negotiable instruments"], None),
    "cpc1908": (["code of civil procedure"], None),
}

def probe_one(url):
    s = make_session()
    try:
        r = s.head(url, timeout=8)
        if r.status_code != 200: return None
        r2 = s.get(url, timeout=30)
        if r2.status_code != 200 or r2.content[:4] != b"%PDF": return None
        d = fitz.open(stream=r2.content, filetype="pdf")
        t = ""
        for i in range(min(2, d.page_count)):
            t += d.load_page(i).get_text()
        d.close()
        tl = t.lower()[:3000]
        return (url, len(r2.content), d.page_count, t[:300].replace("\n"," "), tl, r2.content)
    except Exception:
        return None


# Build list of URLs to probe
URLS = []
# Known good zone: 2023/06 range 060200-060400
for mid in range(2023060200, 2023060400):
    URLS.append(f"https://cdnbbsr.s3waas.gov.in/{HASH}/uploads/2023/06/{mid}.pdf")
# Also: 2023/05, 2023/07, 2023/12 adjacent ranges (often ministries upload in bursts)
for yrmo, base in [("2023/05", 2023050100), ("2023/07", 2023070100),
                    ("2023/12", 2023120000)]:
    for i in range(300):
        URLS.append(f"https://cdnbbsr.s3waas.gov.in/{HASH}/uploads/{yrmo}/{base+i}.pdf")

print(f"Probing {len(URLS)} URLs with 20 threads...")

found = []
with ThreadPoolExecutor(max_workers=20) as ex:
    futs = {ex.submit(probe_one, u): u for u in URLS}
    for i, fu in enumerate(as_completed(futs)):
        r = fu.result()
        if r:
            url, sz, pg, snippet, lowtext, content = r
            found.append(r)
            if i % 50 == 0 or len(found) <= 20:
                print(f"  [hit #{len(found)}] {url[-40:]} {sz//1024}KB p={pg}: {snippet[:100]}")

print(f"\n{len(found)} PDFs found. Matching to targets...")
print()

saved = {}
for url, sz, pg, snippet, tl, content in found:
    for tkey, (kws, dest_rel) in TARGETS.items():
        if any(k in tl for k in kws):
            if tkey in saved: continue
            # Filter out garbage
            if "amendment bill" in tl[:1000] and "principal" not in tl[:1000]:
                continue
            if "research paper" in tl or "jstor" in tl:
                continue
            saved[tkey] = (url, sz, pg, snippet)
            if dest_rel:
                dest = os.path.join(STAGE, dest_rel)
                # Only overwrite if current file is missing or smaller
                existing = os.path.exists(dest)
                if not existing:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as f: f.write(content)
                    print(f"  [SAVED] {tkey} -> {dest_rel} ({sz//1024}KB, {pg}pp)")
                    print(f"         from {url[-50:]}")
                else:
                    print(f"  [skip] {tkey} already on disk -> {dest_rel}")
            else:
                print(f"  [already-have] {tkey} at {url[-50:]}")
            break

print(f"\n{len(saved)} targets matched")
unfound = [k for k in TARGETS if k not in saved and TARGETS[k][1] is not None]
print(f"still missing: {unfound}")

# Final staging
print("\n=== FINAL STAGING ===")
total_pdfs = 0; total_kb = 0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"): continue
    pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
    kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
    if pdfs:
        total_pdfs += len(pdfs); total_kb += kb
        print(f"  [{d:<32}] {len(pdfs):>3} pdfs  {kb:>7} KB")
print(f"TOTAL: {total_pdfs} pdfs  {total_kb/1024:.1f} MB")
