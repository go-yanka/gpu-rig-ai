#!/usr/bin/env python3
"""Sequential cdnbbsr scan — smaller, focused."""
import requests, urllib3, fitz, os, time
urllib3.disable_warnings()
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
s = requests.Session(); s.headers.update({"User-Agent": UA}); s.verify = False
STAGE = "/opt/indian-legal-ai/gst_stage"

HASH = "s3ec037a4bf9ba2bd774068ad50351fb89"

TARGETS = {
    "ipc": (["indian penal code"],                       "t1_old_criminal/ipc_1860.pdf"),
    "crpc": (["code of criminal procedure, 1973"],       "t1_old_criminal/crpc_1973.pdf"),
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
    "igst": (["integrated goods and services"],           "t1_gst_circulars/IGST_Act.pdf"),
    "utgst": (["union territory goods"],                  "t1_gst_circulars/UTGST_Act.pdf"),
    "fema1999": (["foreign exchange management act"],     "t1_fema_rbi/FEMA_1999.pdf"),
    "pmla": (["prevention of money laundering"],          "t1_other_bare_acts/pmla_2002.pdf"),
    "sebi_ICDR": (["issue of capital and disclosure"],    "t1_companies_sebi/sebi_ICDR.pdf"),
    "sebi_LODR": (["listing obligations"],                "t1_companies_sebi/sebi_LODR.pdf"),
}

saved = {}
found_index = []

def process(url):
    try:
        r = s.get(url, timeout=20)
        if r.status_code != 200 or r.content[:4] != b"%PDF": return
        try:
            d = fitz.open(stream=r.content, filetype="pdf")
            pages = d.page_count
            t = ""
            for i in range(min(2, pages)):
                t += d.load_page(i).get_text()
            d.close()
        except Exception:
            return
        tl = t.lower()[:4000]
        title = t[:200].replace("\n"," ").strip()
        found_index.append((url, len(r.content), pages, title))
        # match to targets
        for tkey, (kws, dest_rel) in TARGETS.items():
            if tkey in saved: continue
            if not any(k in tl for k in kws): continue
            if "amendment bill" in tl[:600] and "principal" not in tl[:600]: continue
            if "research paper" in tl: continue
            if "page 1" == tl.strip()[:7] and pages < 5: continue
            # Save
            saved[tkey] = (url, len(r.content), pages, title)
            if dest_rel:
                dest = os.path.join(STAGE, dest_rel)
                if not os.path.exists(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as f: f.write(r.content)
                    print(f"  [SAVED {len(r.content)//1024}KB pg={pages}] {tkey} <- {url[-40:]}")
                    print(f"         {title[:140]}")
            break
    except Exception:
        pass


# Known productive range — 2023/06/ IDs
print("=== Scanning 2023/06/2023060200..060600 ===")
for mid in range(2023060200, 2023060600):
    url = f"https://cdnbbsr.s3waas.gov.in/{HASH}/uploads/2023/06/{mid}.pdf"
    process(url)
    # print progress every 50
    if (mid - 2023060200) % 50 == 49:
        print(f"  ...checked {mid-2023060199}, hits={len(found_index)}, saved={len(saved)}")

print(f"\nTotal PDFs found in 2023/06: {len(found_index)}")
print(f"Matched to targets: {len(saved)}")
for u, sz, pg, t in found_index:
    print(f"  {u[-18:]} {sz//1024:>5}KB p={pg:>4}: {t[:90]}")

print("\n=== STILL NEEDED ===")
for k,(kws,dest) in TARGETS.items():
    if dest and k not in saved:
        print(f"  missing: {k} -> {dest}")

print("\n=== STAGING ===")
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
