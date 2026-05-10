#!/usr/bin/env python3
"""Fast parallel probe: 30 threads, short timeouts, save hits to /tmp/hits, then classify."""
import requests, urllib3, fitz, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
urllib3.disable_warnings()

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
STAGE = "/opt/indian-legal-ai/gst_stage"
HITS_DIR = "/tmp/hits"
os.makedirs(HITS_DIR, exist_ok=True)

HASH = "s3ec037a4bf9ba2bd774068ad50351fb89"

# Broader range, multiple months under same hash
URLS = []
for yrmo in ["2023/06", "2023/05", "2023/07", "2023/12"]:
    base = int(yrmo.replace("/","")) * 10000  # e.g. 20230600000
    for i in range(200, 700):
        mid = base + i
        URLS.append((mid, f"https://cdnbbsr.s3waas.gov.in/{HASH}/uploads/{yrmo}/{mid}.pdf"))

def probe(item):
    mid, url = item
    try:
        r = requests.head(url, timeout=4, verify=False,
                          headers={"User-Agent": UA}, allow_redirects=True)
        if r.status_code != 200: return None
        r2 = requests.get(url, timeout=12, verify=False,
                          headers={"User-Agent": UA}, stream=False)
        if r2.status_code != 200 or r2.content[:4] != b"%PDF":
            return None
        path = os.path.join(HITS_DIR, f"{mid}.pdf")
        with open(path, "wb") as f: f.write(r2.content)
        return (mid, len(r2.content), path)
    except Exception:
        return None

print(f"Probing {len(URLS)} URLs with 30 threads...")
hits = []
done = 0
with ThreadPoolExecutor(max_workers=30) as ex:
    futs = {ex.submit(probe, u): u for u in URLS}
    for f in as_completed(futs):
        done += 1
        r = f.result()
        if r:
            hits.append(r)
        if done % 200 == 0:
            print(f"  checked {done}/{len(URLS)}, hits={len(hits)}")
print(f"DONE: {len(hits)} PDFs saved to {HITS_DIR}")

# Classify
print("\n=== Classifying hits ===")
TARGETS = {
 "ipc":            (["indian penal code"],              "t1_old_criminal/ipc_1860.pdf"),
 "crpc":           (["code of criminal procedure, 1973","code of criminal procedure,1973"], "t1_old_criminal/crpc_1973.pdf"),
 "evidence":       (["indian evidence act"],            "t1_old_criminal/evidence_act_1872.pdf"),
 "contract":       (["indian contract act"],            "t1_commercial_acts/contract_act_1872.pdf"),
 "companies2013":  (["companies act, 2013"],            "t1_companies_sebi/companies_act_2013.pdf"),
 "income_tax":     (["income-tax act, 1961","income tax act, 1961"], "t1_income_tax/income_tax_act_1961.pdf"),
 "sale_of_goods":  (["sale of goods act"],              "t1_commercial_acts/sale_of_goods_1930.pdf"),
 "specific_relief":(["specific relief act"],            "t1_commercial_acts/specific_relief_1963.pdf"),
 "arbitration":    (["arbitration and conciliation"],   "t1_commercial_acts/arbitration_1996.pdf"),
 "ni":             (["negotiable instruments act"],     "t1_commercial_acts/ni_act_1881.pdf"),
 "cpc":            (["code of civil procedure"],        "t1_civil_procedure/cpc_1908.pdf"),
 "tpa":            (["transfer of property act"],       "t1_other_bare_acts/tpa_1882.pdf"),
 "consumer2019":   (["consumer protection act, 2019"],  "t1_other_bare_acts/consumer_protection_2019.pdf"),
 "mv1988":         (["motor vehicles act"],             "t1_other_bare_acts/mv_act_1988.pdf"),
 "hma":            (["hindu marriage act"],             "t1_other_bare_acts/hma_1955.pdf"),
 "bnss":           (["nagarik suraksha sanhita"],       "t1_criminal_codes_2023/BNSS_2023.pdf"),
 "igst":           (["integrated goods and services"],  "t1_gst_circulars/IGST_Act.pdf"),
 "utgst":          (["union territory goods"],          "t1_gst_circulars/UTGST_Act.pdf"),
 "fema1999":       (["foreign exchange management act"],"t1_fema_rbi/FEMA_1999.pdf"),
 "pmla":           (["prevention of money laundering"], "t1_other_bare_acts/pmla_2002.pdf"),
 "hsa1956":        (["hindu succession act"],           "t1_other_bare_acts/hsa_1956.pdf"),
 "sarfaesi":       (["securitisation and reconstruction"], "t1_other_bare_acts/sarfaesi_2002.pdf"),
 "rti":            (["right to information act"],       "t1_other_bare_acts/rti_2005.pdf"),
 "domestic_viol":  (["protection of women from domestic violence"], "t1_other_bare_acts/dv_2005.pdf"),
 "juvenile":       (["juvenile justice"],               "t1_other_bare_acts/juvenile_2015.pdf"),
 "pocso":          (["protection of children from sexual offences"], "t1_other_bare_acts/pocso_2012.pdf"),
 "it_act":         (["information technology act"],     "t1_other_bare_acts/it_act_2000.pdf"),
}
BAD = ["amendment bill","written answer","jstor","research paper","papers laid",
       "digital copy of a book","cornell university"]

classified = {}
catalog = []
for mid, sz, path in hits:
    try:
        d = fitz.open(path)
        pg = d.page_count
        t = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
        d.close()
    except Exception:
        continue
    tl = t.lower()[:6000]
    title = t[:160].replace("\n"," ").strip()
    # match best target
    best = None
    for tkey, (kws, dest_rel) in TARGETS.items():
        if any(k in tl for k in kws):
            # amendment-bill filter
            if "amendment bill" in tl[:1000] and "principal" not in tl[:1000]: continue
            if any(b in tl for b in BAD): continue
            best = tkey; break
    catalog.append((mid, sz, pg, title, best))
    if best and best not in classified:
        dest = os.path.join(STAGE, TARGETS[best][1])
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                with open(path, "rb") as src: f.write(src.read())
            classified[best] = (mid, sz, pg, title)
            print(f"  [SAVED] {best:<16} <- {mid}  {sz//1024}KB p={pg}")
        else:
            classified[best] = (mid, sz, pg, title)
            print(f"  [have] {best:<16} <- {mid}  (dest exists)")

print(f"\n=== Catalog of {len(catalog)} hits ===")
for mid, sz, pg, title, best in sorted(catalog):
    tag = best or "-"
    print(f"  {mid}  {sz//1024:>5}KB p={pg:>4}  [{tag:<16}]  {title[:100]}")

print(f"\n=== Still missing ===")
for k, (kws, dest) in TARGETS.items():
    if k not in classified:
        print(f"  {k:<16} -> {dest}")

print(f"\n=== Final staging ===")
tp=0;tk=0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"): continue
    pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
    kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
    if pdfs:
        tp+=len(pdfs);tk+=kb
        print(f"  {d:<32} {len(pdfs):>3} pdfs  {kb:>7} KB")
print(f"TOTAL: {tp} pdfs, {tk/1024:.1f} MB")
