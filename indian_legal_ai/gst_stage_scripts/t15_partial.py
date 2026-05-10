#!/usr/bin/env python3
"""T1.5 partial harvest: POSH + RERA (Grok-confirmed URLs).
Plus cleanup: empty dirs + duplicate labour wages file + old constitution.
Inline QC on every download."""
import os, shutil, hashlib, requests, urllib3, fitz
urllib3.disable_warnings()

STAGE = "/opt/indian-legal-ai/gst_stage"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HDRS = {"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9"}

# make new categories
NEW_DIRS = ["t1_workplace", "t1_real_estate"]
for d in NEW_DIRS:
    os.makedirs(os.path.join(STAGE, d), exist_ok=True)

JOBS = [
    ("posh_2013", "sexual harassment of women at workplace", "t1_workplace/posh_2013.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2119/1/A2013-14.pdf",
    ]),
    ("rera_2016", "real estate (regulation and development)", "t1_real_estate/rera_2016.pdf", [
        "https://www.indiacode.nic.in/bitstream/123456789/2155/1/A2016-16.pdf",
    ]),
]

BAD = ["amendment bill", "written answer", "jstor", "research paper", "papers laid",
       "digital copy of a book", "cornell university", "all india christian council",
       "leave of absence", "google books"]


def qc(content, kw):
    if len(content) < 1024: return False, f"too-small ({len(content)}B)", {}
    if content[:4] != b"%PDF": return False, f"not-pdf ({content[:20]!r})", {}
    try:
        d = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        return False, f"parse-err: {e}", {}
    pg = d.page_count
    if pg < 3:
        d.close(); return False, f"stub ({pg}p)", {"pg": pg}
    tot = sum(len(d.load_page(i).get_text()) for i in range(pg))
    cpp = tot // max(pg, 1)
    if cpp < 200:
        d.close(); return False, f"scanned ({cpp}c/p)", {"pg": pg, "cpp": cpp}
    first = "".join(d.load_page(i).get_text() for i in range(min(3, pg)))
    d.close()
    tl = first.lower()[:8000]
    title = first[:200].replace("\n", " ").strip()
    if kw.lower() not in tl:
        return False, f"kw-miss '{kw}'", {"pg": pg, "cpp": cpp, "title": title[:100]}
    for b in BAD:
        if b in tl: return False, f"bad '{b}'", {"pg": pg}
    return True, "ok", {"pg": pg, "cpp": cpp, "size_kb": len(content)//1024, "title": title[:140]}


def main():
    print("=" * 70)
    print("STAGE 1: Harvest POSH + RERA (Grok URLs)")
    print("=" * 70)
    sess = requests.Session()
    try: sess.get("https://www.indiacode.nic.in/", headers={"User-Agent": UA}, timeout=15, verify=False)
    except: pass

    saved = 0; failed = []
    for label, kw, rel, urls in JOBS:
        print(f"\n[{label}]")
        got = False
        for u in urls:
            print(f"  try {u[-95:]}")
            try:
                r = sess.get(u, headers=HDRS, timeout=60, verify=False, allow_redirects=True)
            except Exception as e:
                print(f"    EXC {type(e).__name__}"); continue
            if r.status_code != 200:
                print(f"    http-{r.status_code}"); continue
            ok, reason, meta = qc(r.content, kw)
            if not ok:
                print(f"    REJECT: {reason} {meta}"); continue
            dest = os.path.join(STAGE, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f: f.write(r.content)
            print(f"    SAVED {meta['size_kb']}KB {meta['pg']}p c/p={meta['cpp']}")
            print(f"    title: {meta['title']}")
            saved += 1; got = True; break
        if not got: failed.append(label)

    print(f"\nSTAGE 1 result: {saved}/{len(JOBS)} saved")
    if failed: print("FAILED:", failed)

    print("\n" + "=" * 70)
    print("STAGE 2: Cleanup")
    print("=" * 70)

    # 2a: delete empty placeholder dirs
    empty_deleted = 0
    for d in sorted(os.listdir(STAGE)):
        full = os.path.join(STAGE, d)
        if not os.path.isdir(full): continue
        if not d.startswith("t1_"): continue
        contents = os.listdir(full)
        if not contents:
            os.rmdir(full); empty_deleted += 1
            print(f"  deleted empty dir: {d}")
    print(f"  empty dirs deleted: {empty_deleted}")

    # 2b: dedup by sha256 across tree
    print("\nScanning for byte-identical duplicates...")
    seen = {}  # sha -> path
    dups = []
    for root, _, files in os.walk(STAGE):
        for f in files:
            if not f.lower().endswith(".pdf"): continue
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                h = hashlib.sha256(fh.read()).hexdigest()
            if h in seen:
                dups.append((p, seen[h]))
            else:
                seen[h] = p
    if dups:
        print(f"  found {len(dups)} byte-identical duplicates:")
        for dup, orig in dups:
            print(f"    DUP: {dup}")
            print(f"    ORIG: {orig}")
    else:
        print("  no byte-identical duplicates")

    # 2c: label known semantic dupes (labour wages, constitution) - don't auto-delete, just flag
    print("\nSemantic dup candidates (NOT auto-deleted — manual review):")
    labour = os.path.join(STAGE, "t1_labour_codes")
    if os.path.isdir(labour):
        for f in sorted(os.listdir(labour)):
            if "wages" in f.lower():
                sz = os.path.getsize(os.path.join(labour, f)) // 1024
                print(f"  labour: {f}  {sz}KB")
    const = os.path.join(STAGE, "t1_constitution")
    if os.path.isdir(const):
        for f in sorted(os.listdir(const)):
            sz = os.path.getsize(os.path.join(const, f)) // 1024
            print(f"  const:  {f}  {sz}KB")

    print("\n" + "=" * 70)
    print("FINAL T1 STAGING")
    print("=" * 70)
    tp = 0; tk = 0
    for d in sorted(os.listdir(STAGE)):
        full = os.path.join(STAGE, d)
        if not os.path.isdir(full) or not d.startswith("t1_"): continue
        pdfs = [f for f in os.listdir(full) if f.lower().endswith(".pdf")]
        kb = sum(os.path.getsize(os.path.join(full, f)) for f in pdfs) // 1024
        if pdfs:
            tp += len(pdfs); tk += kb
            print(f"  {d:<32} {len(pdfs):>3} pdfs  {kb:>7} KB")
    print(f"TOTAL: {tp} pdfs, {tk/1024:.1f} MB")


if __name__ == "__main__":
    main()
