#!/usr/bin/env python3
"""Download the verified URLs + wider scan for still-missing acts."""
import requests, urllib3, fitz, os, time, re
urllib3.disable_warnings()
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
s = requests.Session(); s.headers.update({"User-Agent": UA}); s.verify = False
STAGE = "/opt/indian-legal-ai/gst_stage"

def verify_bytes(content, kw_list):
    if content[:4] != b"%PDF": return False, 0, ""
    try:
        d = fitz.open(stream=content, filetype="pdf")
        pages = d.page_count
        txt = "".join(d.load_page(i).get_text() for i in range(min(3, pages)))
        d.close()
        tl = txt.lower()[:5000]
        hit = any(k in tl for k in kw_list)
        bad = ["amendment bill","written answer","jstor","research paper",
               "papers laid","digital copy of a book","cornell university"]
        if any(b in tl for b in bad): hit = False
        return hit, pages, txt[:150].replace("\n"," ")
    except Exception as e:
        return False, 0, f"parse-err: {e}"

def dl(url, dest, kw_list, retries=2):
    for _ in range(retries):
        try:
            r = s.get(url, timeout=90, allow_redirects=True)
            if r.status_code != 200: return False, f"HTTP-{r.status_code}", 0
            ok, pg, snip = verify_bytes(r.content, kw_list)
            if not ok: return False, f"WRONG ({snip[:80]})", 0
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f: f.write(r.content)
            return True, f"OK pg={pg}", len(r.content)
        except Exception as e:
            last = str(e)[:50]
        time.sleep(1)
    return False, f"ERR {last}", 0


print("=" * 80)
print("PART 1: Download VERIFIED URLs")
print("=" * 80)

VERIFIED = [
 ("t1_commercial_acts/contract_act_1872.pdf", ["indian contract"],
  "https://aphc.gov.in/docs/6_20241130050320.pdf"),
 ("t1_civil_procedure/cpc_1908.pdf", ["code of civil procedure"],
  "https://www.indiacode.nic.in/bitstream/123456789/13813/1/the_code_of_civil_procedure%2C_1908.pdf"),
 ("t1_commercial_acts/ni_act_1881.pdf", ["negotiable instruments"],
  "https://cdnbbsr.s3waas.gov.in/s3ec037a4bf9ba2bd774068ad50351fb89/uploads/2023/06/2023060272.pdf"),
 ("t1_gst_circulars/CGST-Act-2022.pdf", ["central goods and services"],
  "https://cbic-gst.gov.in/pdf/CGST-Act-2017-amended-01012022.pdf"),
 ("t1_criminal_codes_2023/BSA_2023.pdf", ["sakshya adhiniyam"],
  "https://www.mha.gov.in/sites/default/files/250882_english_01042024.pdf"),
]

for relpath, kws, url in VERIFIED:
    dest = os.path.join(STAGE, relpath)
    ok, msg, sz = dl(url, dest, kws)
    print(f"  [{msg:<30}] {sz//1024:>6}KB -> {relpath}")

# Delete the junk files that are wrong content
print("\n== Removing confirmed-junk files ==")
JUNK = [
 "t1_criminal_codes_2023/BNSS_2023.pdf",  # contained BSA, now replaced by BSA_2023
 "t1_criminal_codes_2023/gazette_egaz_250880.pdf",
 "t1_criminal_codes_2023/gazette_egaz_250881.pdf",
 "t1_criminal_codes_2023/gazette_egaz_250885.pdf",
 "t1_criminal_codes_2023/gazette_egaz_250886.pdf",
 "t1_income_tax/income_tax_act_1961.pdf",
 "t1_companies_sebi/companies_act_2013.pdf",
 "t1_commercial_acts/ni_act_1881.pdf",  # will be rewritten above
 "t1_commercial_acts/sale_of_goods_1930.pdf",
 "t1_commercial_acts/specific_relief_1963.pdf",
 "t1_commercial_acts/arbitration_1996.pdf",
 "t1_old_criminal/ipc_1860.pdf",
 "t1_old_criminal/crpc_1973.pdf",
 "t1_old_criminal/evidence_act_1872.pdf",
 "t1_other_bare_acts/tpa_1882.pdf",
 "t1_other_bare_acts/consumer_protection_2019.pdf",
 "t1_other_bare_acts/mv_act_1988.pdf",
 "t1_other_bare_acts/hma_1955.pdf",
]
for j in JUNK:
    p = os.path.join(STAGE, j)
    if os.path.exists(p):
        os.remove(p); print(f"  removed {j}")


print("\n" + "=" * 80)
print("PART 2: Wider MHA scan 250800-250950 for BNSS and others")
print("=" * 80)
for n in range(250800, 250950):
    u = f"https://www.mha.gov.in/sites/default/files/{n}_english_01042024.pdf"
    try:
        r = s.get(u, timeout=15, allow_redirects=True)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            d = fitz.open(stream=r.content, filetype="pdf")
            t = d.load_page(0).get_text()[:250].replace("\n", " ")
            d.close()
            tl = t.lower()
            tag = ("BNS" if "nyaya sanhita" in tl else
                   "BNSS" if "nagarik suraksha" in tl else
                   "BSA" if "sakshya adhiniyam" in tl else "?")
            print(f"  [{tag:<4}] {n} {len(r.content)//1024}KB: {t[:120]}")
    except Exception:
        pass
    time.sleep(0.08)


print("\n" + "=" * 80)
print("PART 3: AP HC /docs scan — they had Contract Act; what else?")
print("=" * 80)
# aphc.gov.in/docs/N_DATE.pdf — try probing sequential IDs
# Their Contract Act was at ID=6. Try 1-30.
for aid in range(1, 50):
    # We don't know the date suffix but each act has a unique one; try known pattern
    # Actually their URL was "/docs/6_20241130050320.pdf" — the number after underscore is upload datetime.
    # Can't enumerate without a directory listing. Try /docs/ listing.
    pass
# Instead, try fetching the AP HC docs page that lists these
try:
    r = s.get("https://aphc.gov.in/bareacts.html", timeout=20)
    if r.status_code == 200:
        hrefs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', r.text, re.I)
        print(f"  aphc.gov.in/bareacts.html: {len(hrefs)} pdf links")
        for h in hrefs[:40]:
            print(f"    {h}")
    else:
        print(f"  [{r.status_code}] aphc.gov.in/bareacts.html")
except Exception as e:
    print(f"  err: {e}")

# Try alternate AP HC paths
for p in ["/acts.html","/BareActs.html","/bareact.html","/Acts","/docs/","/library.html"]:
    try:
        r = s.get("https://aphc.gov.in" + p, timeout=15, allow_redirects=True)
        print(f"  [{r.status_code}] aphc.gov.in{p}  size={len(r.content)//1024}KB")
        if r.status_code == 200 and len(r.content) > 5000:
            hrefs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', r.text, re.I)
            print(f"      -> {len(hrefs)} pdf links")
            for h in hrefs[:10]: print(f"         {h}")
    except Exception as e:
        pass


print("\n" + "=" * 80)
print("PART 4: cdnbbsr.s3waas search — these are NIC-hosted PDFs")
print("=" * 80)
# The cdnbbsr S3 URLs follow pattern: /s<hash>/uploads/YYYY/MM/<timestamp><seq>.pdf
# Can't enumerate, but legislative.gov.in now redirects many assets there.
# Try searching for known "publications" page
for url in [
 "https://legislative.gov.in/acts",
 "https://legislative.gov.in/constitution-of-india",
 "https://legislative.gov.in/principal-act",
 "https://legislative.gov.in/rpa1951",
 "https://legislative.gov.in/acts-central-acts",
 "https://legislative.gov.in/en/principal-acts",
]:
    try:
        r = s.get(url, timeout=20, allow_redirects=True)
        if r.status_code == 200:
            hrefs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', r.text, re.I)
            # filter cdnbbsr or direct PDFs
            pdf_links = [h for h in hrefs if "cdnbbsr" in h or h.endswith(".pdf")]
            print(f"  [{r.status_code}] {url}: {len(pdf_links)} pdf links")
            for h in pdf_links[:15]: print(f"    {h}")
        else:
            print(f"  [{r.status_code}] {url}")
    except Exception as e:
        print(f"  err: {url[:60]}: {str(e)[:40]}")


print("\n" + "=" * 80)
print("PART 5: bareactslive.com / Bombay HC / other mirrors")
print("=" * 80)
TESTS = [
 ("bombayhc", ["indian penal code"], "https://bombayhighcourt.nic.in/libweb/acts/2000.04.pdf"),
 ("bareactslive_ipc", ["indian penal code"], "https://bareactslive.com/ACA/ACT034.HTM"),
 ("advocatekhoj_ipc", ["indian penal code"], "https://www.advocatekhoj.com/library/bareacts/indianpenalcode/index.php"),
 ("ap_hc_ipc",  ["indian penal code"], "https://aphc.gov.in/docs/ipc.pdf"),
 ("ap_hc_evidence", ["evidence"],      "https://aphc.gov.in/docs/evidence.pdf"),
 ("ap_hc_crpc",["criminal procedure"], "https://aphc.gov.in/docs/crpc.pdf"),
 ("dojingov_ipc",["indian penal code"],"https://doj.gov.in/sites/default/files/IPC.pdf"),
 ("dojingov_crpc",["criminal procedure"],"https://doj.gov.in/sites/default/files/CrPC.pdf"),
 ("nlu_bareacts_ipc", ["indian penal code"], "https://www.nludelhi.ac.in/download/publication/IPC.pdf"),
]
for label, kws, url in TESTS:
    try:
        r = s.get(url, timeout=20, allow_redirects=True)
        ct = r.headers.get("Content-Type","")[:40]
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            ok, pg, snip = verify_bytes(r.content, kws)
            tag = "OK" if ok else "WRONG"
            print(f"  [{tag}] {len(r.content)//1024}KB  {label}  {url[:70]}")
            if snip: print(f"       {snip[:140]}")
        else:
            print(f"  [{r.status_code} ct={ct[:30]}] {label}  {url[:70]}")
    except Exception as e:
        print(f"  [err] {label}: {str(e)[:40]}")


print("\n=== DONE ===")
