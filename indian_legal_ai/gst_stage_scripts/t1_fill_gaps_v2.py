#!/usr/bin/env python3
"""
T1 fill-gap v2 — use `requests` + multi-candidate URL lists for each missing act.
Also tries Internet Archive identifier-guessing.
"""
import requests, urllib3, os, re, time
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"

def S():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*",
                      "Accept-Language": "en-US,en;q=0.9"})
    s.verify = False
    return s

def stream_to(s, url, dest, timeout=60, referer=None):
    try:
        h = {}
        if referer: h["Referer"] = referer
        r = s.get(url, timeout=timeout, stream=True, allow_redirects=True, headers=h)
        if r.status_code != 200:
            r.close(); return False, 0, f"http-{r.status_code}"
        it = r.iter_content(8192)
        try:
            first = next(it)
        except StopIteration:
            r.close(); return False, 0, "empty"
        if not first.startswith(b"%PDF"):
            r.close(); return False, 0, f"not-pdf {first[:24]!r}"
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            f.write(first); n = len(first)
            for ch in it:
                f.write(ch); n += len(ch)
        r.close()
        if n < 20000:
            os.remove(tmp); return False, 0, f"too-small {n}B"
        os.replace(tmp, dest)
        return True, n, "ok"
    except Exception as e:
        return False, 0, str(e)[:80]

def try_candidates(s, track, fname, candidates):
    dest_dir = os.path.join(STAGE, track)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 50000:
        return True, os.path.getsize(dest), "already"
    for url in candidates:
        ok, size, reason = stream_to(s, url, dest)
        if ok:
            return True, size, url
        time.sleep(0.3)
    return False, 0, "all-candidates-failed"


# =============================================================================
# Candidate URL lists — tried in order, first valid PDF wins
# =============================================================================

TARGETS = [
    # ---------- A2 Income Tax ----------
    ("t1_income_tax", "income_tax_act_1961.pdf", [
        "https://incometaxindia.gov.in/Acts/Income-tax%20Act,%201961/Printing%20version%20ITA.pdf",
        "https://incometaxindia.gov.in/Pages/acts/income-tax-act.aspx",
        "https://archive.org/download/in.gazette.central.2020.incometaxact/incometaxact1961.pdf",
        "https://dor.gov.in/sites/default/files/IT-Act.pdf",
        "https://www.mca.gov.in/Ministry/pdf/Income-Tax-Act-1961.pdf",
    ]),

    # ---------- A3 Companies / SEBI ----------
    ("t1_companies_sebi", "companies_act_2013.pdf", [
        "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
        "https://www.mca.gov.in/content/dam/mca/pdf/CompaniesAct2013.pdf",
        "https://ca2013.com/wp-content/uploads/2013/08/Companies-Act-2013.pdf",
        "https://www.icsi.edu/media/webmodules/CSJ/June/CompaniesAct2013.pdf",
    ]),
    ("t1_companies_sebi", "sebi_act_1992.pdf", [
        "https://www.sebi.gov.in/acts/act15ac.pdf",
        "https://www.sebi.gov.in/sebi_data/attachdocs/1456380272553.pdf",
    ]),

    # ---------- A6 Commercial Acts ----------
    ("t1_commercial_acts", "contract_act_1872.pdf", [
        "https://legislative.gov.in/sites/default/files/A1872-09.pdf",
        "https://www.mca.gov.in/Ministry/pdf/IndianContractAct1872.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1872-9.pdf",
    ]),
    ("t1_commercial_acts", "ni_act_1881.pdf", [
        "https://legislative.gov.in/sites/default/files/A1881-26.pdf",
        "https://rbidocs.rbi.org.in/rdocs/content/pdfs/NIAAct.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1881-26.pdf",
    ]),
    ("t1_commercial_acts", "sale_of_goods_1930.pdf", [
        "https://legislative.gov.in/sites/default/files/A1930-03.pdf",
        "https://legislative.gov.in/sites/default/files/A1930-3.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1930-3.pdf",
    ]),
    ("t1_commercial_acts", "partnership_act_1932.pdf", [
        "https://legislative.gov.in/sites/default/files/A1932-09.pdf",
        "https://legislative.gov.in/sites/default/files/A1932-9.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1932-9.pdf",
    ]),
    ("t1_commercial_acts", "specific_relief_1963.pdf", [
        "https://legislative.gov.in/sites/default/files/A1963-47.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1963-47.pdf",
    ]),
    ("t1_commercial_acts", "arbitration_1996.pdf", [
        "https://legislative.gov.in/sites/default/files/A1996-26.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1996-26.pdf",
        "https://doj.gov.in/sites/default/files/Arbitration-and-Conciliation-Act-1996.pdf",
    ]),
    ("t1_commercial_acts", "limitation_act_1963.pdf", [
        "https://legislative.gov.in/sites/default/files/A1963-36.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1963-36.pdf",
    ]),

    # ---------- A7 Civil Procedure ----------
    ("t1_civil_procedure", "cpc_1908.pdf", [
        "https://legislative.gov.in/sites/default/files/A1908-05.pdf",
        "https://legislative.gov.in/sites/default/files/A1908-5.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1908-5.pdf",
        "https://doj.gov.in/sites/default/files/CPC.pdf",
    ]),

    # ---------- A10 Old Criminal Codes ----------
    ("t1_old_criminal", "ipc_1860.pdf", [
        "https://legislative.gov.in/sites/default/files/A1860-45.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1860-45.pdf",
        "https://www.ncwapps.nic.in/acts/TheIndianPenalCode1860.pdf",
        "https://www.mea.gov.in/Images/CPV/lcld1_IPC.pdf",
    ]),
    ("t1_old_criminal", "crpc_1973.pdf", [
        "https://legislative.gov.in/sites/default/files/A1974-02.pdf",
        "https://legislative.gov.in/sites/default/files/A1974-2.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1974-2.pdf",
        "https://www.ncwapps.nic.in/acts/TheCodeofCriminalProcedure1973.pdf",
    ]),
    ("t1_old_criminal", "evidence_act_1872.pdf", [
        "https://legislative.gov.in/sites/default/files/A1872-01.pdf",
        "https://legislative.gov.in/sites/default/files/A1872-1.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1872-1.pdf",
        "https://www.ncwapps.nic.in/acts/TheIndianEvidenceAct1872.pdf",
    ]),

    # ---------- A11 Constitution ----------
    ("t1_constitution", "constitution_of_india_full.pdf", [
        "https://legislative.gov.in/sites/default/files/COI.pdf",
        "https://legislative.gov.in/sites/default/files/coi-4March2016.pdf",
        "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/COI.pdf",
    ]),

    # ---------- A12 Other bare acts ----------
    ("t1_other_bare_acts", "tpa_1882.pdf", [
        "https://legislative.gov.in/sites/default/files/A1882-04.pdf",
        "https://legislative.gov.in/sites/default/files/A1882-4.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1882-4.pdf",
    ]),
    ("t1_other_bare_acts", "registration_act_1908.pdf", [
        "https://legislative.gov.in/sites/default/files/A1908-16.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1908-16.pdf",
    ]),
    ("t1_other_bare_acts", "stamp_act_1899.pdf", [
        "https://legislative.gov.in/sites/default/files/A1899-02.pdf",
        "https://legislative.gov.in/sites/default/files/A1899-2.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1899-2.pdf",
    ]),
    ("t1_other_bare_acts", "consumer_protection_2019.pdf", [
        "https://legislative.gov.in/sites/default/files/A2019-35.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A2019-35.pdf",
        "https://consumeraffairs.nic.in/sites/default/files/CPAct2019.pdf",
    ]),
    ("t1_other_bare_acts", "mv_act_1988.pdf", [
        "https://legislative.gov.in/sites/default/files/A1988-59.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1988-59.pdf",
        "https://morth.nic.in/sites/default/files/MV_Act.pdf",
    ]),
    ("t1_other_bare_acts", "hma_1955.pdf", [
        "https://legislative.gov.in/sites/default/files/A1955-25.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1955-25.pdf",
    ]),
    ("t1_other_bare_acts", "hsa_1956.pdf", [
        "https://legislative.gov.in/sites/default/files/A1956-30.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A1956-30.pdf",
    ]),

    # ---------- A8 Labour Codes (fill the remaining two) ----------
    ("t1_labour_codes", "code_on_wages_2019.pdf", [
        "https://labour.gov.in/sites/default/files/THE_CODE_ON_WAGES_2019_No_29_cy.pdf",
        "https://labour.gov.in/sites/default/files/CodeonWages2019.pdf",
        "https://legislative.gov.in/sites/default/files/A2019-29.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A2019-29.pdf",
    ]),
    ("t1_labour_codes", "industrial_relations_2020.pdf", [
        "https://labour.gov.in/sites/default/files/IR_Gazette_of_India.pdf",
        "https://legislative.gov.in/sites/default/files/A2020-35.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A2020-35.pdf",
    ]),
    ("t1_labour_codes", "social_security_2020.pdf", [
        "https://labour.gov.in/sites/default/files/SS_Code_Gazette.pdf",
        "https://legislative.gov.in/sites/default/files/A2020-36.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A2020-36.pdf",
    ]),
    ("t1_labour_codes", "osh_2020.pdf", [
        "https://labour.gov.in/sites/default/files/OSH_Gazette.pdf",
        "https://legislative.gov.in/sites/default/files/A2020-37.pdf",
        "https://lddashboard.legislative.gov.in/sites/default/files/A2020-37.pdf",
    ]),
]

# ==========================================================================
print("=" * 80)
print("T1 FILL-GAPS v2 — multi-candidate streaming downloader")
print("=" * 80)

s = S()
# warm
for u in ["https://legislative.gov.in/", "https://lddashboard.legislative.gov.in/"]:
    try: s.get(u, timeout=20)
    except Exception: pass

ok_count = 0
fail = []
for track, fname, cands in TARGETS:
    success, size, info = try_candidates(s, track, fname, cands)
    if success:
        ok_count += 1
        src = info if info == "already" else info.split("/")[2]
        print(f"  [OK {size//1024}KB] {track}/{fname}  src={src}")
    else:
        fail.append((track, fname))
        print(f"  [FAIL] {track}/{fname}")

print(f"\n{ok_count}/{len(TARGETS)} succeeded, {len(fail)} failed")

# =============================================================================
# Round 2: For failures, try Internet Archive with EXACT-title search
# =============================================================================
print("\n" + "=" * 80)
print("ROUND 2: Internet Archive fallback for failures")
print("=" * 80)

IA_QUERIES = {
    "income_tax_act_1961.pdf": 'title:"Income Tax Act" AND title:1961',
    "companies_act_2013.pdf": 'title:"Companies Act" AND title:2013',
    "contract_act_1872.pdf": 'title:"Indian Contract Act"',
    "ni_act_1881.pdf": 'title:"Negotiable Instruments Act"',
    "sale_of_goods_1930.pdf": 'title:"Sale of Goods Act"',
    "partnership_act_1932.pdf": 'title:"Indian Partnership Act"',
    "cpc_1908.pdf": 'title:"Code of Civil Procedure"',
    "ipc_1860.pdf": 'title:"Indian Penal Code"',
    "crpc_1973.pdf": 'title:"Code of Criminal Procedure"',
    "evidence_act_1872.pdf": 'title:"Indian Evidence Act"',
    "tpa_1882.pdf": 'title:"Transfer of Property Act"',
    "registration_act_1908.pdf": 'title:"Registration Act" AND title:1908',
    "stamp_act_1899.pdf": 'title:"Indian Stamp Act"',
    "mv_act_1988.pdf": 'title:"Motor Vehicles Act"',
    "hma_1955.pdf": 'title:"Hindu Marriage Act"',
    "hsa_1956.pdf": 'title:"Hindu Succession Act"',
    "constitution_of_india_full.pdf": 'title:"Constitution of India"',
    "specific_relief_1963.pdf": 'title:"Specific Relief Act"',
    "limitation_act_1963.pdf": 'title:"Limitation Act" AND title:1963',
    "consumer_protection_2019.pdf": 'title:"Consumer Protection Act" AND title:2019',
    "arbitration_1996.pdf": 'title:"Arbitration" AND title:1996',
    "sebi_act_1992.pdf": 'title:"SEBI Act"',
    "code_on_wages_2019.pdf": 'title:"Code on Wages"',
    "industrial_relations_2020.pdf": 'title:"Industrial Relations Code"',
    "social_security_2020.pdf": 'title:"Social Security Code"',
    "osh_2020.pdf": 'title:"Occupational Safety"',
}

ia_ok = 0
for track, fname in fail:
    q = IA_QUERIES.get(fname)
    if not q:
        print(f"  [skip no-query] {fname}"); continue
    try:
        url = ("https://archive.org/advancedsearch.php?q=" + requests.utils.quote(q) +
               "+AND+mediatype%3Atexts&fl%5B%5D=identifier&fl%5B%5D=title&rows=15&output=json")
        r = s.get(url, timeout=30)
        docs = r.json().get("response",{}).get("docs",[])
        if not docs:
            print(f"  [no-results] {fname} q={q[:50]}"); continue
        # Pick first doc that has a reasonable-size PDF
        got = False
        for doc in docs[:5]:
            ident = doc["identifier"]
            meta = s.get("https://archive.org/metadata/" + ident, timeout=30).json()
            pdfs = [f for f in meta.get("files",[])
                    if f.get("name","").lower().endswith(".pdf")
                    and int(f.get("size","0") or 0) > 50000]
            if not pdfs: continue
            # pick largest PDF
            pdfs.sort(key=lambda f: int(f.get("size","0") or 0), reverse=True)
            fn = pdfs[0]["name"]
            dl = "https://archive.org/download/" + ident + "/" + fn
            dest_dir = os.path.join(STAGE, track)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, fname)
            ok, size, reason = stream_to(s, dl, dest)
            if ok:
                print(f"  [OK {size//1024}KB] {track}/{fname}  ident={ident}")
                ia_ok += 1; got = True; break
        if not got:
            print(f"  [no-valid-pdf] {fname}")
        time.sleep(0.5)
    except Exception as e:
        print(f"  [err] {fname}: {str(e)[:60]}")

print(f"\nIA round: {ia_ok} additional successes")

# =============================================================================
# FINAL TALLY
# =============================================================================
print("\n\n" + "=" * 80)
print("FINAL STAGING TALLY")
print("=" * 80)
total_pdfs = 0; total_kb = 0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"): continue
    pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
    kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
    total_pdfs += len(pdfs); total_kb += kb
    flag = "OK" if pdfs else "!!"
    print(f"  [{flag}] {d:<36} {len(pdfs):>4} pdfs  {kb:>8} KB")
print(f"\n  TOTAL: {total_pdfs} PDFs, {total_kb} KB ({total_kb/1024:.1f} MB)")
print("\n=== DONE ===")
