#!/usr/bin/env python3
"""T1 expansion: 30 additional foundational acts + fix IBC. Multi-candidate URLs per act."""
import os, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8"}

# Create new staging tracks
for t in ["t1_interpretation","t1_personal_law","t1_ip_acts","t1_banking","t1_customs_excise",
          "t1_criminal_special","t1_environment","t1_pre_codified_labour","t1_data_privacy"]:
    os.makedirs(os.path.join(STAGE, t), exist_ok=True)

# (label, title_kw, dest_rel, [urls_to_try_in_order])
JOBS = [
    # === Interpretation framework ===
    ("general_clauses_1897", "general clauses act", "t1_interpretation/general_clauses_1897.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2309/1/A1897-10.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2309/1/a1897-10.pdf",
        "https://legislative.gov.in/sites/default/files/A1897-10.pdf",
    ]),
    # === Personal law ===
    ("indian_succession_1925", "indian succession act", "t1_personal_law/indian_succession_1925.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2362/1/A1925-39.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2362/1/a1925-39.pdf",
    ]),
    ("special_marriage_1954", "special marriage act", "t1_personal_law/special_marriage_1954.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1650/1/A1954-43.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1650/1/a1954-43.pdf",
    ]),
    ("muslim_dissolution_1939", "dissolution of muslim marriages", "t1_personal_law/muslim_dissolution_1939.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2401/1/A1939-08.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2401/1/a1939-08.pdf",
    ]),
    ("shariat_1937", "shariat", "t1_personal_law/shariat_application_1937.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2402/1/A1937-26.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2402/1/a1937-26.pdf",
    ]),
    ("indian_divorce_1869", "indian divorce act", "t1_personal_law/indian_divorce_1869.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2357/1/A1869-04.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2357/1/a1869-04.pdf",
    ]),
    # === Property-adjacent ===
    ("indian_trusts_1882", "indian trusts act", "t1_other_bare_acts/indian_trusts_1882.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2340/1/A1882-02.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2340/1/a1882-02.pdf",
    ]),
    ("indian_easements_1882", "indian easements act", "t1_other_bare_acts/indian_easements_1882.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2341/1/A1882-05.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2341/1/a1882-05.pdf",
    ]),
    # === Banking / finance ===
    ("rbi_1934", "reserve bank of india act", "t1_banking/rbi_act_1934.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2386/1/A1934-02.pdf",
        "https://rbidocs.rbi.org.in/rdocs/Publications/PDFs/RBIA1934170510.pdf",
        "https://www.rbi.org.in/Scripts/BS_ViewRbiAct.aspx",
    ]),
    ("banking_regulation_1949", "banking regulation act", "t1_banking/banking_regulation_1949.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1631/1/A1949-10.pdf",
        "https://rbidocs.rbi.org.in/rdocs/Publications/PDFs/BANKI15122014.pdf",
    ]),
    ("insurance_1938", "insurance act, 1938", "t1_banking/insurance_act_1938.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2397/1/A1938-04.pdf",
        "https://irdai.gov.in/documents/37343/365858/Insurance+Act+1938.pdf",
    ]),
    ("benami_1988", "benami transactions", "t1_banking/benami_1988.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1782/1/A1988-45.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1782/1/a1988-45.pdf",
    ]),
    # === Customs / Excise ===
    ("customs_1962", "customs act, 1962", "t1_customs_excise/customs_act_1962.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1587/1/A1962-52.pdf",
        "https://cbic.gov.in/resources//htdocs-cbec/customs/cs-act/customs-act-ch1-ch17.pdf",
    ]),
    ("central_excise_1944", "central excise act", "t1_customs_excise/central_excise_1944.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1367/1/A1944-01.pdf",
        "https://cbic.gov.in/resources//htdocs-cbec/excise/cx-act/cx-act-2015.pdf",
    ]),
    # === IP ===
    ("copyright_1957", "copyright act", "t1_ip_acts/copyright_1957.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1367/1/A1957-14.pdf",
        "https://copyright.gov.in/documents/CopyrightRules1957.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1367/1/a1957-14.pdf",
    ]),
    ("patents_1970", "patents act", "t1_ip_acts/patents_1970.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1392/1/A1970-39.pdf",
        "https://ipindia.gov.in/writereaddata/Portal/IPOAct/1_31_1_patent-act-1970-11march2015.pdf",
    ]),
    ("trade_marks_1999", "trade marks act", "t1_ip_acts/trade_marks_1999.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1993/1/A1999-47.pdf",
        "https://ipindia.gov.in/writereaddata/Portal/IPOAct/1_43_1_tmr-act.pdf",
    ]),
    # === Criminal special ===
    ("pc_act_1988", "prevention of corruption act", "t1_criminal_special/pc_act_1988.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1793/1/A1988-49.pdf",
        "https://cvc.gov.in/sites/default/files/pcact2018_0.pdf",
    ]),
    ("ndps_1985", "narcotic drugs", "t1_criminal_special/ndps_1985.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1791/1/A1985-61.pdf",
        "https://narcoticsindia.nic.in/writereaddata/Portal/uploads/Acts/15_NDPS_Act.pdf",
    ]),
    ("uapa_1967", "unlawful activities", "t1_criminal_special/uapa_1967.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1470/1/A1967-37.pdf",
        "https://www.mha.gov.in/sites/default/files/A1967-37.pdf",
    ]),
    ("arms_1959", "arms act", "t1_criminal_special/arms_1959.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1435/1/A1959-54.pdf",
    ]),
    # === Modern ===
    ("dpdp_2023", "digital personal data protection", "t1_data_privacy/dpdp_2023.pdf", [
        "https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/19865/1/dpdp_act_2023.pdf",
    ]),
    ("aadhaar_2016", "aadhaar", "t1_data_privacy/aadhaar_2016.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2155/1/a2016-18.pdf",
        "https://uidai.gov.in/images/targeted_delivery_of_financial_and_other_subsidies_benefits_and_services_13072016.pdf",
    ]),
    ("competition_2002", "competition act", "t1_other_bare_acts/competition_2002.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2004/1/A2002-12.pdf",
        "https://www.cci.gov.in/images/legalframework/en/the-competition-act-20021652267631.pdf",
    ]),
    # === Environment ===
    ("environment_1986", "environment (protection) act", "t1_environment/environment_1986.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1767/1/A1986-29.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1767/1/a1986-29.pdf",
    ]),
    ("wildlife_1972", "wild life", "t1_environment/wildlife_1972.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1726/1/A1972-53.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1726/1/a1972-53.pdf",
    ]),
    # === Pre-codified labour ===
    ("industrial_disputes_1947", "industrial disputes act", "t1_pre_codified_labour/industrial_disputes_1947.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1404/1/A1947-14.pdf",
        "https://labour.gov.in/sites/default/files/THEINDUSTRIALDISPUTES_ACT1947_0.pdf",
    ]),
    ("factories_1948", "factories act", "t1_pre_codified_labour/factories_1948.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1552/1/A1948-63.pdf",
        "https://labour.gov.in/sites/default/files/factories_act_1948.pdf",
    ]),
    ("min_wages_1948", "minimum wages act", "t1_pre_codified_labour/minimum_wages_1948.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1551/1/A1948-11.pdf",
        "https://labour.gov.in/sites/default/files/TheMinimumWagesAct1948_0.pdf",
    ]),
    ("gratuity_1972", "payment of gratuity", "t1_pre_codified_labour/gratuity_1972.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1622/1/A1972-39.pdf",
        "https://labour.gov.in/sites/default/files/Payment%20of%20Gratuity%20Act%20%281%29.pdf",
    ]),
    # === IBC fix (Grok: new handle 15479) ===
    ("ibc_2016", "insolvency and bankruptcy", "t1_ibc/IBC_Code_2016.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/15479/1/the_insolvency_and_bankruptcy_code%2C_2016.pdf",
        "https://prsindia.org/files/bills_acts/acts_parliament/2016/the-insolvency-and-bankruptcy-code-act,-2016.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2154/1/AA2016-31.pdf",
    ]),
    # === IGST / UTGST enacted (Grok: PRS URLs) ===
    ("igst_enacted", "integrated goods and services tax act, 2017", "t1_gst_circulars/IGST_Act_2017.pdf", [
        "https://prsindia.org/files/bills_acts/acts_parliament/2017/the-integrated-goods-and-services-tax-act,-2017.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/11909/1/A2017-13.pdf",
    ]),
    ("utgst_enacted", "union territory goods and services tax act, 2017", "t1_gst_circulars/UTGST_Act_2017.pdf", [
        "https://prsindia.org/files/bills_acts/acts_parliament/2017/the-union-territory-goods-and-services-tax-act,-2017.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/11911/1/A2017-14.pdf",
    ]),
]

BAD = ["amendment bill","written answer","jstor","research paper","papers laid",
       "digital copy of a book","cornell university","all india christian council"]

results = []
for label, kw, rel, urls in JOBS:
    ok = False
    for u in urls:
        try:
            r = requests.get(u, headers=HDRS, timeout=30, verify=False, allow_redirects=True)
        except Exception as e:
            print(f"  [{label:<26}] EXC {type(e).__name__}: {u[:80]}"); continue
        if r.status_code != 200:
            print(f"  [{label:<26}] HTTP {r.status_code}: {u[:80]}"); continue
        if r.content[:4] != b"%PDF":
            print(f"  [{label:<26}] NOT-PDF ({r.content[:20]!r}): {u[:80]}"); continue
        try:
            d = fitz.open(stream=r.content, filetype="pdf")
            pg = d.page_count
            tot = sum(len(d.load_page(i).get_text()) for i in range(pg))
            cpp = tot // max(pg,1)
            t = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
            d.close()
        except Exception as e:
            print(f"  [{label:<26}] PARSE-ERR: {e}"); continue
        tl = t.lower()[:8000]
        title = t[:160].replace("\n"," ").strip()
        if kw not in tl:
            print(f"  [{label:<26}] KW MISS '{kw}' ({pg}p {cpp}c/p): {title[:60]}"); continue
        if any(b in tl for b in BAD):
            print(f"  [{label:<26}] BAD-MARKER"); continue
        if cpp < 200:
            print(f"  [{label:<26}] SCANNED ({cpp}c/p)"); continue
        if pg < 4:
            print(f"  [{label:<26}] STUB ({pg}p)"); continue
        dest = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f: f.write(r.content)
        print(f"  [{label:<26}] SAVED {len(r.content)//1024:>5}KB {pg:>3}p c/p={cpp}")
        results.append((label, "OK", pg, len(r.content)//1024, u))
        ok = True; break
    if not ok:
        results.append((label, "FAIL", 0, 0, ""))

print("\n" + "="*80)
ok_count = sum(1 for r in results if r[1]=="OK")
print(f"SUCCESS: {ok_count}/{len(JOBS)}")
print("\nFAILED:")
for r in results:
    if r[1] != "OK": print(f"  - {r[0]}")

print("\n=== FINAL STAGING ===")
tp=0;tk=0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"): continue
    pdfs = sorted([f for f in os.listdir(full) if f.lower().endswith(".pdf")])
    kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
    if pdfs:
        tp+=len(pdfs); tk+=kb
        print(f"  {d:<32} {len(pdfs):>3} pdfs  {kb:>7} KB")
print(f"TOTAL: {tp} pdfs, {tk/1024:.1f} MB")
