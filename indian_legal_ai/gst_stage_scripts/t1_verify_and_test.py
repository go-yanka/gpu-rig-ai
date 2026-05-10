#!/usr/bin/env python3
"""Test Grok/Gemini candidate URLs + MHA gazette scan. Pure content verification."""
import requests, urllib3, fitz, time
urllib3.disable_warnings()
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
s = requests.Session(); s.headers.update({"User-Agent": UA}); s.verify = False

def verify(url, kw_list):
    try:
        r = s.get(url, timeout=60, allow_redirects=True)
        if r.status_code != 200:
            return f"HTTP-{r.status_code}", 0, 0, ""
        if r.content[:4] != b"%PDF":
            return "NOT-PDF", 0, 0, r.content[:40].decode("utf-8","ignore")
        d = fitz.open(stream=r.content, filetype="pdf")
        pages = d.page_count
        txt = "".join(d.load_page(i).get_text() for i in range(min(3, pages)))
        d.close()
        tl = txt.lower()[:5000]
        hit = any(k in tl for k in kw_list)
        bad_markers = ["amendment bill", "written answer", "jstor",
                       "research paper", "papers laid",
                       "digital copy of a book", "cornell university"]
        if any(b in tl for b in bad_markers):
            hit = False
        snip = txt[:220].replace("\n", " ").strip()
        return ("OK" if hit else "WRONG-CONTENT"), len(r.content), pages, snip
    except Exception as e:
        return f"ERR:{str(e)[:50]}", 0, 0, ""

print("=" * 80)
print("PART A: Grok candidate URLs")
print("=" * 80)
GROK = [
 ("evidence_hydpol",    ["indian evidence act"],       "https://www.hyderabadpolice.gov.in/PDF/acts/THEINDIANEVIDENCEACT.pdf"),
 ("contract_aphc",      ["indian contract act"],       "https://aphc.gov.in/docs/6_20241130050320.pdf"),
 ("cpc_13813",          ["code of civil procedure"],   "https://www.indiacode.nic.in/bitstream/123456789/13813/1/the_code_of_civil_procedure%2C_1908.pdf"),
 ("constitution_cdn",   ["constitution of india"],     "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf"),
 ("cgst_2020_cbic",     ["central goods and services"],"https://cbic-gst.gov.in/pdf/CGST-Act-Updated-30092020.pdf"),
 ("igst_2020_cbic",     ["integrated goods"],          "https://cbic-gst.gov.in/pdf/IGST-Act-Updated-30092020.pdf"),
 ("utgst_2020_cbic",    ["union territory goods"],     "https://cbic-gst.gov.in/pdf/UTGST-Act-Updated-30092020.pdf"),
 ("cpc_gemini_cdn",     ["code of civil procedure"],   "https://cdnbbsr.s3waas.gov.in/s3ec045421e013565f7f1afa0cfe8ad87a/uploads/2023/12/2023123197.pdf"),
 ("ni_gemini_cdn",      ["negotiable instruments"],    "https://cdnbbsr.s3waas.gov.in/s3ec037a4bf9ba2bd774068ad50351fb89/uploads/2023/06/2023060272.pdf"),
 ("stamp_gemini_cdn",   ["indian stamp"],              "https://cdnbbsr.s3waas.gov.in/s32d579dc29360d8bbfbb4aa541de5afa9/uploads/2025/03/20250306577835334.pdf"),
 ("cgst_2022_cbic",     ["central goods and services"],"https://cbic-gst.gov.in/pdf/CGST-Act-2017-amended-01012022.pdf"),
]
for label, kws, url in GROK:
    status, sz, pg, snip = verify(url, kws)
    print(f"[{status:<18}] {sz//1024:>6}KB p={pg:>4}  {label}")
    if snip: print(f"       {snip[:180]}")
    print(f"       {url}")
    time.sleep(0.4)

print()
print("=" * 80)
print("PART B: MHA gazette scan 250870-250900 (find BNS, BNSS, BSA)")
print("=" * 80)
for n in range(250870, 250901):
    u = f"https://www.mha.gov.in/sites/default/files/{n}_english_01042024.pdf"
    try:
        r = s.get(u, timeout=20, allow_redirects=True)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            d = fitz.open(stream=r.content, filetype="pdf")
            t = d.load_page(0).get_text()[:300].replace("\n", " ")
            d.close()
            tl = t.lower()
            tag = "BNS" if "nyaya sanhita" in tl else ("BNSS" if "nagarik suraksha" in tl else ("BSA" if "sakshya adhiniyam" in tl else "?"))
            print(f"  [{tag:<4}] {n} {len(r.content)//1024}KB: {t[:140]}")
    except Exception:
        pass
    time.sleep(0.2)

print()
print("=" * 80)
print("PART C: Income Tax alternate URLs")
print("=" * 80)
IT = [
 "https://incometaxindia.gov.in/Acts/Income-tax Act, 1961/2024/102120000000084114.pdf",
 "https://incometaxindia.gov.in/Acts/Income-tax%20Act%2C%201961/2024/102120000000084114.pdf",
 "https://incometaxindia.gov.in/Acts/Income-tax%20Act,%201961/2023/102120000000084115.pdf",
 "https://incometaxindia.gov.in/Acts/Income-tax%20Act,%201961/2022/102120000000084110.pdf",
]
for u in IT:
    try:
        r = s.get(u, timeout=30, allow_redirects=True)
        ct = r.headers.get("Content-Type","")
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            d = fitz.open(stream=r.content, filetype="pdf")
            t = d.load_page(0).get_text()[:200].replace("\n"," ")
            d.close()
            tl = t.lower()
            tag = "OK-IT" if "income-tax act" in tl or "income tax act" in tl else "?"
            print(f"  [{tag}] {len(r.content)//1024}KB  {u[:80]}")
            print(f"       {t[:150]}")
        else:
            print(f"  [{r.status_code}] ct={ct[:30]}  {u[:80]}")
    except Exception as e:
        print(f"  [err] {str(e)[:40]}  {u[:60]}")

print()
print("=" * 80)
print("PART D: Common legal-mirror sites for missing bare acts")
print("=" * 80)
MIRRORS = [
 ("ipc_hyd",     ["indian penal code"], "https://www.hyderabadpolice.gov.in/PDF/acts/THEINDIANPENALCODE1860.pdf"),
 ("crpc_hyd",    ["criminal procedure"],"https://www.hyderabadpolice.gov.in/PDF/acts/THECODEOFCRIMINALPROCEDURE.pdf"),
 ("mva_hyd",     ["motor vehicles"],    "https://www.hyderabadpolice.gov.in/PDF/acts/THEMOTORVEHICLESACT.pdf"),
 ("contract_bombay",["indian contract"],"https://bombayhighcourt.nic.in/libweb/acts/1872.09.pdf"),
 ("contract_jknhrc",["indian contract"],"https://jkhrc.nic.in/acts/indian%20contract%20act.pdf"),
 ("ipc_jknhrc", ["indian penal code"], "https://jkhrc.nic.in/acts/IPC.pdf"),
 ("ipc_mha",    ["indian penal code"], "https://www.mha.gov.in/sites/default/files/2022-08/A1860-45.pdf"),
 ("ipc_mea",    ["indian penal code"], "https://www.mea.gov.in/Images/CPV/legaltreaties/Indian-Penal-Code.pdf"),
 ("ipc_ncw",    ["indian penal code"], "https://ncw.nic.in/sites/default/files/IPC.pdf"),
 ("consumer_cdn",["consumer protection"],"https://consumeraffairs.nic.in/sites/default/files/CP%20Act%202019.pdf"),
 ("hma_legislative",["hindu marriage"],"https://cdnbbsr.s3waas.gov.in/s3e2ad76f2326fbc6b56a45a56c59fafdb/uploads/2023/05/2023050559.pdf"),
]
for label, kws, url in MIRRORS:
    status, sz, pg, snip = verify(url, kws)
    print(f"[{status:<18}] {sz//1024:>6}KB p={pg:>4}  {label}")
    if snip: print(f"       {snip[:180]}")
    time.sleep(0.4)
