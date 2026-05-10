#!/usr/bin/env python3
"""Round 4: indiacode search -> handle -> real bitstream (with referer chain).
Also probe cdnbbsr adjacent IDs."""
import requests, urllib3, fitz, os, re, time
urllib3.disable_warnings()
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
s = requests.Session()
s.headers.update({"User-Agent": UA, "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                  "Accept-Language":"en-US,en;q=0.9"})
s.verify = False
STAGE = "/opt/indian-legal-ai/gst_stage"

def verify_bytes(b, kws):
    if not b or b[:4] != b"%PDF": return False, 0, ""
    try:
        d = fitz.open(stream=b, filetype="pdf")
        pg = d.page_count
        t = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
        d.close()
        tl = t.lower()[:5000]
        hit = any(k in tl for k in kws)
        bad = ["amendment bill","written answer","jstor","research paper",
               "papers laid","digital copy of a book","cornell university",
               "early	  journal	  content"]
        if any(x in tl for x in bad): hit = False
        return hit, pg, t[:150].replace("\n"," ")
    except Exception:
        return False, 0, "parse-err"

def indiacode_search_download(query, kws, dest):
    """Search indiacode -> land on handle -> find bitstream -> download with referer."""
    # Step 1: warm session with homepage
    try: s.get("https://www.indiacode.nic.in/", timeout=20)
    except: pass
    # Step 2: search
    url = "https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query=" + requests.utils.quote(query) + "&submit=Go"
    try:
        r = s.get(url, timeout=30)
        html = r.text
    except Exception as e:
        return False, f"search-err: {e}"
    # Find all /handle/123456789/NNNN/ links (item pages, not collection browse)
    handles = re.findall(r'/handle/123456789/(\d+)', html)
    handles = list(dict.fromkeys(handles))  # dedup preserving order
    # Skip collection pages (1362 etc)
    handles = [h for h in handles if int(h) > 10 and h != "1362"]
    tried = []
    for h in handles[:12]:
        hurl = f"https://www.indiacode.nic.in/handle/123456789/{h}"
        try:
            rr = s.get(hurl, timeout=25, headers={"Referer": url})
            phtml = rr.text
        except Exception:
            continue
        # Look for title match to filter
        title_match = re.search(r'<title>([^<]+)</title>', phtml)
        title = title_match.group(1).lower() if title_match else ""
        # Check page text for kws
        ptext = re.sub(r'<[^>]+>', ' ', phtml).lower()[:3000]
        if not any(k in ptext for k in kws):
            continue
        # Find bitstream download link
        bs = re.findall(r'/bitstream/123456789/\d+/\d+/[^"\'\s<>]+\.(?:pdf|PDF)', phtml)
        for b in bs:
            pdf_url = "https://www.indiacode.nic.in" + b
            try:
                rr2 = s.get(pdf_url, timeout=45, headers={"Referer": hurl}, allow_redirects=True)
                if rr2.status_code == 200:
                    ok, pg, snip = verify_bytes(rr2.content, kws)
                    if ok:
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with open(dest, "wb") as f: f.write(rr2.content)
                        return True, f"OK pg={pg} handle={h} sz={len(rr2.content)//1024}KB"
                    else:
                        tried.append((h, b, "WRONG-CONTENT", snip[:60]))
                else:
                    tried.append((h, b, f"HTTP-{rr2.status_code}", ""))
            except Exception as e:
                tried.append((h, b, f"ERR:{str(e)[:30]}", ""))
        time.sleep(0.3)
    return False, f"no-match-in-{len(handles)}-handles, tried={len(tried)}"


# ==========================================================================
print("=" * 80)
print("PART 1: IndiaCode search-handle-bitstream chain")
print("=" * 80)

TARGETS = [
 ("t1_old_criminal/ipc_1860.pdf",              ["indian penal code"],    "indian penal code 1860"),
 ("t1_old_criminal/crpc_1973.pdf",             ["criminal procedure"],   "code of criminal procedure 1973"),
 ("t1_old_criminal/evidence_act_1872.pdf",     ["indian evidence"],      "indian evidence act 1872"),
 ("t1_income_tax/income_tax_act_1961.pdf",     ["income-tax act","income tax act"], "income-tax act 1961"),
 ("t1_companies_sebi/companies_act_2013.pdf",  ["companies act"],        "companies act 2013"),
 ("t1_commercial_acts/sale_of_goods_1930.pdf", ["sale of goods"],        "sale of goods act 1930"),
 ("t1_commercial_acts/specific_relief_1963.pdf",["specific relief"],     "specific relief act 1963"),
 ("t1_commercial_acts/arbitration_1996.pdf",   ["arbitration and conciliation"], "arbitration and conciliation act 1996"),
 ("t1_other_bare_acts/tpa_1882.pdf",           ["transfer of property"], "transfer of property act 1882"),
 ("t1_other_bare_acts/consumer_protection_2019.pdf",["consumer protection"], "consumer protection act 2019"),
 ("t1_other_bare_acts/mv_act_1988.pdf",        ["motor vehicles"],       "motor vehicles act 1988"),
 ("t1_other_bare_acts/hma_1955.pdf",           ["hindu marriage"],       "hindu marriage act 1955"),
]

for rel, kws, q in TARGETS:
    dest = os.path.join(STAGE, rel)
    ok, msg = indiacode_search_download(q, kws, dest)
    flag = "OK" if ok else "FAIL"
    print(f"  [{flag}] {rel:<50}  {msg}")
    time.sleep(0.6)


# ==========================================================================
print("\n" + "=" * 80)
print("PART 2: Probe cdnbbsr adjacent IDs (NIC S3WAAS CDN)")
print("=" * 80)
# The hash s3ec037a4bf9ba2bd774068ad50351fb89 hosted NI Act at 2023060272.
# Probe nearby IDs + other hashes
HASHES = [
    "s3ec037a4bf9ba2bd774068ad50351fb89",
    "s3ec045421e013565f7f1afa0cfe8ad87a",
    "s32d579dc29360d8bbfbb4aa541de5afa9",
    "s380537a945c7aaa788ccfcdf1b99b5d8f",
]
# Try IDs near the known-good 2023060272
hits = []
for h in HASHES:
    for yr_mo in ["2023/06","2023/12","2023/05","2024/01","2024/06","2024/12","2025/03"]:
        for bid in range(2023060260, 2023060290):
            u = f"https://cdnbbsr.s3waas.gov.in/{h}/uploads/{yr_mo}/{bid}.pdf"
            try:
                r = s.head(u, timeout=8)
                if r.status_code == 200:
                    # fetch content
                    r2 = s.get(u, timeout=30)
                    if r2.status_code == 200 and r2.content[:4] == b"%PDF":
                        try:
                            d = fitz.open(stream=r2.content, filetype="pdf")
                            t = d.load_page(0).get_text()[:250].replace("\n"," ")
                            d.close()
                            print(f"  [FOUND] {u[-40:]}  {t[:100]}")
                            hits.append((u, t))
                        except Exception: pass
            except Exception: pass
        if hits and len(hits) >= 5: break
    if hits and len(hits) >= 5: break
print(f"  probed, {len(hits)} hits found")


# ==========================================================================
print("\n" + "=" * 80)
print("PART 3: MHA legal framework search + BNSS dedicated hunt")
print("=" * 80)
# MHA /en/commoncontent/acts-rules, /en/documents/acts etc.
for u in ["https://www.mha.gov.in/en/commoncontent/acts-rules",
          "https://www.mha.gov.in/en/notifications/gazette-notifications",
          "https://www.mha.gov.in/en/divisionofmha/cs-i-division",
          "https://www.mha.gov.in/en/divisionofmha/cs-ii-division",
          "https://www.mha.gov.in/"]:
    try:
        r = s.get(u, timeout=25)
        if r.status_code == 200:
            pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', r.text, re.I)
            bnss_like = [p for p in pdfs if any(k in p.lower() for k in ["nagarik","suraksha","bnss","250884","250881"])]
            print(f"  [{r.status_code}] {u}: {len(pdfs)} pdfs, {len(bnss_like)} bnss-like")
            for p in bnss_like[:8]: print(f"    {p}")
    except Exception as e:
        print(f"  err: {u}: {str(e)[:40]}")


# ==========================================================================
print("\n" + "=" * 80)
print("PART 4: BNSS via egazette.gov.in Wayback")
print("=" * 80)
# BNSS was gazetted 25 Dec 2023. Try egazette WriteReadData/2023 numbers via Wayback
for n in ["250881","250882","250883","250884","250885","250886","250887",
         "2023-12-25-145","250880","250889","250890","250892","250893"]:
    u = f"https://web.archive.org/web/2024/https://egazette.gov.in/WriteReadData/2023/{n}.pdf"
    try:
        r = s.get(u, timeout=25, allow_redirects=True)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            d = fitz.open(stream=r.content, filetype="pdf")
            t = d.load_page(0).get_text()[:250].replace("\n"," ")
            d.close()
            tl = t.lower()
            tag = "BNS" if "nyaya sanhita" in tl else "BNSS" if "nagarik suraksha" in tl else "BSA" if "sakshya adhiniyam" in tl else "?"
            print(f"  [{tag:<4}] egaz/{n} {len(r.content)//1024}KB: {t[:100]}")
    except Exception:
        pass
    time.sleep(0.3)


print("\n=== FINAL STAGING ===")
total_pdfs = 0; total_kb = 0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"): continue
    pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
    kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
    if pdfs:
        total_pdfs += len(pdfs); total_kb += kb
        print(f"  [{d:<32}]  {len(pdfs):>3} pdfs  {kb:>7} KB")
print(f"\nTOTAL: {total_pdfs} pdfs  {total_kb/1024:.1f} MB")
