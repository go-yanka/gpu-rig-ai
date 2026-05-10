#!/usr/bin/env python3
"""Build cross-reference mapping chunks from Repeal & Savings schedules
of BNS 2023, BNSS 2023, BSA 2023 (vs IPC, CrPC, Evidence).

Strategy:
- Extract the "Repeal and Savings" section + schedules from each new code
- Detect old_section → new_section correspondences
- Build mapping bridge chunks as JSONL for later ingestion

Output: /opt/indian-legal-ai/gst_stage/t1_bridges/bridges.jsonl
Each line: {"type": "mapping_bridge", "old_act": "IPC 1860", "old_section": "378",
            "new_act": "BNS 2023", "new_section": "303", "topic": "Theft",
            "context_prepend": "[Current: BNS Sec 303 Theft | Legacy: IPC Sec 378]"}
"""
import os, re, json, fitz

STAGE = "/opt/indian-legal-ai/gst_stage"
OUT_DIR = os.path.join(STAGE, "t1_bridges")
os.makedirs(OUT_DIR, exist_ok=True)

PAIRS = [
    ("BNS 2023", "IPC 1860", os.path.join(STAGE, "t1_criminal_codes_2023/BNS_2023.pdf"),
     os.path.join(STAGE, "t1_old_criminal/ipc_1860.pdf")),
    ("BNSS 2023", "CrPC 1973", os.path.join(STAGE, "t1_criminal_codes_2023/BNSS_2023.pdf"),
     os.path.join(STAGE, "t1_old_criminal/crpc_1973.pdf")),
    ("BSA 2023", "Evidence Act 1872", os.path.join(STAGE, "t1_criminal_codes_2023/BSA_2023.pdf"),
     os.path.join(STAGE, "t1_old_criminal/evidence_act_1872.pdf")),
]


def get_full_text(path):
    d = fitz.open(path)
    pages = []
    for i in range(d.page_count):
        pages.append(d.load_page(i).get_text())
    d.close()
    return pages


def extract_toc_sections(pages, act_label):
    """Extract 'Section N. Title' entries from arrangement-of-sections (usually first 5-15 pages)."""
    # Arrangement of Sections lives in first ~15% of pages, or we scan first 30 pages
    scan = "\n".join(pages[:min(30, len(pages))])
    # Patterns: "  303.  Theft." or "SECTIONS 1. Short title" etc.
    # Match: number + dot + title (up to newline or next number)
    pat = re.compile(r"\n\s*(\d{1,3})\.\s+([A-Z][^\n]{3,120})", re.MULTILINE)
    out = {}
    for m in pat.finditer(scan):
        num = m.group(1)
        title = m.group(2).strip().rstrip(".")
        # drop obviously bogus matches (prices, years)
        if len(title) < 4: continue
        if title[0].islower(): continue
        # keep first occurrence
        if num not in out:
            out[num] = title
    return out


def extract_repeal_schedule(pages, old_act_label):
    """Find 'Repeal and Savings' block + any concordance table at end of new-code PDFs.
    Not all PDFs contain explicit mapping tables, so we also fall back on positional heuristic."""
    text = "\n".join(pages)
    # Find "Repeal and Savings" or "Repeal and savings" section text
    m = re.search(r"repeal\s+and\s+sav(?:ings?|ing)", text, re.IGNORECASE)
    rs_text = ""
    if m:
        start = m.start()
        rs_text = text[start:start + 3000]
    return rs_text


def build_bridges():
    all_bridges = []
    summary = []
    for new_label, old_label, new_path, old_path in PAIRS:
        if not os.path.exists(new_path) or not os.path.exists(old_path):
            summary.append(f"{new_label}/{old_label}: MISSING FILE (new={os.path.exists(new_path)} old={os.path.exists(old_path)})")
            continue

        new_pages = get_full_text(new_path)
        old_pages = get_full_text(old_path)

        new_secs = extract_toc_sections(new_pages, new_label)
        old_secs = extract_toc_sections(old_pages, old_label)

        summary.append(f"\n{new_label} <-> {old_label}")
        summary.append(f"  {new_label} sections parsed: {len(new_secs)}")
        summary.append(f"  {old_label} sections parsed: {len(old_secs)}")

        # Build bridges by matching normalized topic keywords between old and new titles
        # Strategy: for each old section with non-trivial title, find best-matching new section
        def norm(s):
            return re.sub(r"[^a-z ]+", "", s.lower()).strip()

        old_items = [(num, title, norm(title)) for num, title in old_secs.items()]
        new_items = [(num, title, norm(title)) for num, title in new_secs.items()]

        pairs_built = 0
        for o_num, o_title, o_norm in old_items:
            if len(o_norm.split()) < 2: continue
            # find exact-title match first
            exact = [n for n in new_items if n[2] == o_norm]
            candidate = exact[0] if exact else None
            if not candidate:
                # fuzzy: shared significant words
                o_words = set(w for w in o_norm.split() if len(w) > 3)
                if not o_words: continue
                best = None; best_overlap = 0
                for n_num, n_title, n_norm in new_items:
                    n_words = set(w for w in n_norm.split() if len(w) > 3)
                    ov = len(o_words & n_words)
                    if ov > best_overlap and ov >= max(2, len(o_words) // 2):
                        best_overlap = ov; best = (n_num, n_title, n_norm)
                candidate = best
            if not candidate: continue
            n_num, n_title, _ = candidate
            topic = o_title if len(o_title) < len(n_title) else n_title
            bridge = {
                "type": "mapping_bridge",
                "old_act": old_label,
                "old_section": o_num,
                "old_title": o_title,
                "new_act": new_label,
                "new_section": n_num,
                "new_title": n_title,
                "topic": topic,
                "context_prepend": f"[Current: {new_label} Sec {n_num} {n_title} | Legacy: {old_label} Sec {o_num} {o_title}]",
                "text": f"The concept of '{topic}' is now covered under {new_label} Section {n_num} ('{n_title}') "
                        f"which replaces {old_label} Section {o_num} ('{o_title}'). "
                        f"When a query references '{old_label} Section {o_num}', the current law is "
                        f"'{new_label} Section {n_num}'.",
            }
            all_bridges.append(bridge)
            pairs_built += 1
        summary.append(f"  bridges built: {pairs_built}")

    # write out
    out_path = os.path.join(OUT_DIR, "bridges.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for b in all_bridges:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    summary.append(f"\nTotal bridges: {len(all_bridges)} -> {out_path}")

    # sample output
    summary.append("\n=== SAMPLE BRIDGES ===")
    for b in all_bridges[:10]:
        summary.append(f"  {b['context_prepend']}")
    if len(all_bridges) > 10:
        summary.append(f"  ... ({len(all_bridges) - 10} more)")

    return "\n".join(summary)


if __name__ == "__main__":
    print(build_bridges())
