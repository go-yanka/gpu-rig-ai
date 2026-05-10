#!/usr/bin/env python3
"""Delete confirmed-bad files, refetch from Gemini's new URLs + hunt real IBC Code."""
import os, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119 Safari/537.36"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8"}

# --- STEP 1: DELETE confirmed-bad files ---
TO_DELETE = [
    # wrong content / junk / scanned
    "t1_commercial_acts/limitation_act_1963.pdf",
    "t1_commercial_acts/partnership_act_1932.pdf",
    "t1_labour_codes/SS_Code_2020_gaz.pdf",
    "t1_labour_codes/code_on_wages_2019.pdf",
    "t1_labour_codes/industrial_relations_2020.pdf",
    "t1_other_bare_acts/registration_act_1908.pdf",
    "t1_other_bare_acts/stamp_act_1899.pdf",
    "t1_constitution/constitution_of_india.pdf",
    "t1_constitution/constitution_of_india_full.pdf",
    # IBC stubs - will replace
    "t1_ibc/2023-10-01-153733-32y6d-5979e652304e4a01e5499dcb740df1c3.pdf",
    "t1_ibc/72872f4aa5992b9a391826d2e734f78e.pdf",
    "t1_ibc/IBC_Code_2016.pdf",
    "t1_ibc/Organization_Chart.pdf",
    "t1_ibc/Swachhata-hi-Seva.pdf",
    # HSA stub - will replace
    "t1_other_bare_acts/hsa_1956.pdf",
    # POCSO scan - will replace
    "t1_other_bare_acts/pocso_2012.pdf",
    # IGST/UTGST bill - will replace with gazette
    "t1_gst_circulars/IGST_Act_2017.pdf",
    "t1_gst_circulars/UTGST_Act_2017.pdf",
]

# FEMA stubs - keep only the Act itself
FEMA_DIR = os.path.join(STAGE, "t1_fema_rbi")
FEMA_KEEP = {"FEMA_Act_1999.pdf"}

print("=== STEP 1: Delete confirmed-bad ===")
deleted = 0
for rel in TO_DELETE:
    p = os.path.join(STAGE, rel)
    if os.path.exists(p):
        os.remove(p); deleted += 1
        print(f"  del  {rel}")

# FEMA cleanup
print("\n=== STEP 1b: FEMA notifications -> move to t1_fema_notifications ===")
FEMA_NOTIF = os.path.join(STAGE, "t1_fema_notifications")
os.makedirs(FEMA_NOTIF, exist_ok=True)
moved = 0
for f in os.listdir(FEMA_DIR):
    if f not in FEMA_KEEP:
        src = os.path.join(FEMA_DIR, f)
        dst = os.path.join(FEMA_NOTIF, f)
        if os.path.isfile(src):
            os.rename(src, dst); moved += 1
print(f"  moved {moved} FEMA notifications out of principal-Act folder")

print(f"\nDeleted {deleted}, moved {moved}")

# --- STEP 2: REFETCH from Gemini's URLs + fallbacks ---
JOBS = [
    ("mv_1988", "motor vehicles act", "t1_other_bare_acts/mv_act_1988.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/9460/1/a1988-59.pdf",
        "https://parivahan.gov.in/parivahan/sites/default/files/MOTOR_VEHICLES_ACT_1988_0.pdf",
        "https://egazette.gov.in/WriteReadData/2019/210413.pdf",
    ]),
    ("hsa_1956", "hindu succession act", "t1_other_bare_acts/hsa_1956.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/1713/1/AAA1956suc___30.pdf",
    ]),
    ("pocso_2012_text", "sexual offences", "t1_other_bare_acts/pocso_2012.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2079/1/AA2012-32.pdf",
    ]),
    ("igst_enacted", "integrated goods and services tax", "t1_gst_circulars/IGST_Act_2017.pdf", [
        "https://egazette.gov.in/WriteReadData/2017/175283.pdf",
        "https://cbic-gst.gov.in/pdf/IGST-Act-2017.pdf",
    ]),
    ("utgst_enacted", "union territory goods and services tax", "t1_gst_circulars/UTGST_Act_2017.pdf", [
        "https://egazette.gov.in/WriteReadData/2017/175285.pdf",
        "https://cbic-gst.gov.in/pdf/UTGST-Act-2017.pdf",
    ]),
    ("ibc_2016", "insolvency and bankruptcy", "t1_ibc/IBC_Code_2016.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2154/1/a2016-31.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2154/5/a2016-31.pdf",
        "https://ibbi.gov.in/uploads/legalframwork/48bf32150f5d6b30477b74f652964edc.pdf",
    ]),
]

BAD = ["amendment bill","written answer","jstor","research paper","papers laid",
       "digital copy of a book","cornell university","all india christian council"]

print("\n=== STEP 2: Refetch from new URLs ===")
for label, kw, rel, urls in JOBS:
    ok = False
    for u in urls:
        try:
            r = requests.get(u, headers=HDRS, timeout=40, verify=False, allow_redirects=True)
        except Exception as e:
            print(f"  [{label}] EXC {type(e).__name__}: {u[:90]}"); continue
        if r.status_code != 200:
            print(f"  [{label}] HTTP {r.status_code}: {u[:90]}"); continue
        if r.content[:4] != b"%PDF":
            print(f"  [{label}] NOT-PDF (starts {r.content[:20]!r}): {u[:90]}"); continue
        try:
            d = fitz.open(stream=r.content, filetype="pdf")
            pg = d.page_count
            t = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
            tot = sum(len(d.load_page(i).get_text()) for i in range(pg))
            cpp = tot // max(pg,1)
            d.close()
        except Exception as e:
            print(f"  [{label}] PARSE-ERR: {e}"); continue
        tl = t.lower()[:8000]
        title = t[:160].replace("\n"," ").strip()
        if kw not in tl:
            print(f"  [{label}] KW MISS '{kw}': {title[:90]}"); continue
        if any(b in tl for b in BAD):
            print(f"  [{label}] BAD-MARKER"); continue
        if cpp < 200:
            print(f"  [{label}] SCANNED ({cpp}c/p): {title[:70]}"); continue
        dest = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f: f.write(r.content)
        print(f"  [{label}] SAVED {len(r.content)//1024}KB p={pg} c/p={cpp} -> {rel}")
        print(f"         via {u[:95]}")
        print(f"         title: {title[:100]}")
        ok = True; break
    if not ok:
        print(f"  [{label}] ALL FAILED")

# --- STEP 3: Final summary ---
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
