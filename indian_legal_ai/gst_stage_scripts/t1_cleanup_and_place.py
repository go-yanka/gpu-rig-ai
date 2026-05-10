#!/usr/bin/env python3
"""Clean up mislabeled files and place PDFs from /tmp/hits by KNOWN mid -> destination."""
import os, shutil, fitz

STAGE = "/opt/indian-legal-ai/gst_stage"
HITS = "/tmp/hits"

# Confirmed mappings from PDF title inspection
PLACE = {
    2023060211: ("t1_old_criminal/crpc_1973.pdf",       "code of criminal procedure"),
    2023060228: ("t1_old_criminal/ipc_1860.pdf",        "indian penal code"),
    2023060229: ("t1_commercial_acts/limitation_1963.pdf", "limitation act"),
    2023060247: ("t1_other_bare_acts/it_act_2000.pdf",  "information technology act"),
    2023060272: ("t1_commercial_acts/ni_act_1881.pdf",  "negotiable instruments"),
    2023060273: ("t1_civil_procedure/cpc_1908.pdf",     "civil procedure"),
    2023060283: ("t1_commercial_acts/contract_act_1872.pdf", "contract act"),
    2023060288: ("t1_other_bare_acts/legal_services_authorities_1987.pdf", "legal services authorities"),
    2023060293: ("t1_other_bare_acts/rti_2005.pdf",     "right to information"),
    2023060298: ("t1_old_criminal/evidence_act_1872.pdf", "indian evidence"),
}

# Remove known-mislabeled files first (current ipc_1860.pdf = CrPC, tpa_1882.pdf = Contract)
REMOVE = [
    "t1_old_criminal/ipc_1860.pdf",       # will be rewritten with correct
    "t1_old_criminal/evidence_act_1872.pdf",  # refresh
    "t1_other_bare_acts/tpa_1882.pdf",    # was Contract Act - wrong destination
]
print("=== Removing mislabeled stubs ===")
for r in REMOVE:
    p = os.path.join(STAGE, r)
    if os.path.exists(p):
        os.remove(p); print(f"  removed {r}")

print("\n=== Placing files by content verification ===")
for mid, (rel, kw) in PLACE.items():
    src = os.path.join(HITS, f"{mid}.pdf")
    dest = os.path.join(STAGE, rel)
    if not os.path.exists(src):
        print(f"  [MISS] {mid}: no source"); continue
    # verify
    try:
        d = fitz.open(src)
        pg = d.page_count
        t = "".join(d.load_page(i).get_text() for i in range(min(2, pg))).lower()
        d.close()
    except Exception as e:
        print(f"  [ERR]  {mid}: {e}"); continue
    if kw not in t[:6000]:
        print(f"  [WRONG] {mid}: kw '{kw}' not in first pages -> SKIP {rel}")
        continue
    # ok, copy
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # existing-larger check: only overwrite if new is >= existing or existing missing
    if os.path.exists(dest):
        if os.path.getsize(dest) >= os.path.getsize(src):
            print(f"  [keep-existing] {rel} ({os.path.getsize(dest)//1024}KB >= {os.path.getsize(src)//1024}KB)")
            continue
    shutil.copy(src, dest)
    print(f"  [SAVED] {rel:<50} {os.path.getsize(dest)//1024:>5}KB p={pg:>3}")

print("\n=== Final staging ===")
tp=0;tk=0
for d in sorted(os.listdir(STAGE)):
    full = os.path.join(STAGE, d)
    if not os.path.isdir(full) or not d.startswith("t1_"): continue
    pdfs = sorted([f for f in os.listdir(full) if f.lower().endswith(".pdf")])
    kb = sum(os.path.getsize(os.path.join(full,f)) for f in pdfs) // 1024
    if pdfs:
        tp+=len(pdfs);tk+=kb
        print(f"\n  {d} ({len(pdfs)} pdfs, {kb} KB):")
        for f in pdfs:
            fkb = os.path.getsize(os.path.join(full,f)) // 1024
            print(f"    {f:<50} {fkb:>6} KB")
print(f"\nTOTAL: {tp} pdfs, {tk/1024:.1f} MB")

print("\n=== Verifying all t1_old_criminal + new placements by title ===")
VERIFY = [
    "t1_old_criminal/ipc_1860.pdf",
    "t1_old_criminal/crpc_1973.pdf",
    "t1_old_criminal/evidence_act_1872.pdf",
    "t1_civil_procedure/cpc_1908.pdf",
    "t1_commercial_acts/contract_act_1872.pdf",
    "t1_commercial_acts/ni_act_1881.pdf",
    "t1_other_bare_acts/it_act_2000.pdf",
    "t1_other_bare_acts/rti_2005.pdf",
    "t1_other_bare_acts/legal_services_authorities_1987.pdf",
]
for rel in VERIFY:
    p = os.path.join(STAGE, rel)
    if not os.path.exists(p):
        print(f"  [missing] {rel}"); continue
    try:
        d = fitz.open(p)
        pg = d.page_count
        t = d.load_page(0).get_text()[:200].replace("\n"," ")
        d.close()
        print(f"  [{pg:>3}p {os.path.getsize(p)//1024:>4}KB] {rel}")
        print(f"         {t[:130]}")
    except Exception as e:
        print(f"  [parse-err] {rel}: {e}")
