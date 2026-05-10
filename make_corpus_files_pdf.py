"""Generate a PDF listing every ingested source file per category, with chunk counts."""
import json, urllib.request, os, re
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether,
)

RIG = "http://192.168.1.107:8090"
FILES_JSON = "D:/_gpu_rig_ai/_corpus_files.json"
OUT = "D:/_gpu_rig_ai/corpus_files_by_category.pdf"

# ---- fetch live meta for labels + health ----
def fetch(path):
    with urllib.request.urlopen(f"{RIG}{path}", timeout=20) as r:
        return json.loads(r.read())

print("Fetching live meta...")
health = fetch("/health")
meta = fetch("/meta")
with open(FILES_JSON, "r") as f:
    files_by_ds = json.load(f)

label_of = {c["value"]: c["label"] for c in meta["categories"]}
count_of = {c["value"]: c["count"] for c in meta["categories"]}

def prettify_filename(fn):
    """Make a filename more readable: strip .pdf, replace separators, title-case if needed."""
    if not fn:
        return "(unknown)"
    name = re.sub(r"\.pdf$", "", fn, flags=re.I)
    name = name.replace("%20", " ")
    # long hash-named ones
    if re.search(r"[A-F0-9]{20,}", name):
        # extract the prefix before the hash
        m = re.match(r"^([A-Z]+[\w\-()]*?)(?=[A-F0-9]{15,})", name)
        prefix = m.group(1) if m else name[:25]
        return f"{prefix} (hash-named, see full row)"
    return name

def filename_note(fn):
    """Extra-small text under the name for hash-named files."""
    if not fn: return ""
    if re.search(r"[A-F0-9]{20,}", fn):
        return f"<font size=6 color='#999'>{fn}</font>"
    return ""

# ---- styles ----
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"],
                          fontSize=22, textColor=colors.HexColor("#1a4480"),
                          spaceAfter=4, alignment=0))
styles.add(ParagraphStyle(name="SubHead", parent=styles["Normal"],
                          fontSize=10, textColor=colors.HexColor("#555"),
                          spaceAfter=12))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"],
                          fontSize=13, textColor=colors.HexColor("#1a4480"),
                          spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="CatHead", parent=styles["Heading3"],
                          fontSize=11, textColor=colors.HexColor("#1a4480"),
                          spaceBefore=10, spaceAfter=2, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="CatSub", parent=styles["Normal"],
                          fontSize=8.5, textColor=colors.HexColor("#777"),
                          spaceAfter=3))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"],
                          fontSize=9, leading=12))
styles.add(ParagraphStyle(name="FileCell", parent=styles["Normal"],
                          fontSize=8.5, leading=11))

# ---- document ----
doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=1.6*cm, rightMargin=1.6*cm,
                        topMargin=1.5*cm, bottomMargin=1.5*cm,
                        title="Indian Legal AI — Source Files per Category")
story = []

# Title
story.append(Paragraph("Indian Legal AI — Source Files per Category", styles["TitleBig"]))
total_files = sum(len(v) for v in files_by_ds.values())
sub = (f"Collection: <b>{health['collection']}</b> &nbsp;&nbsp; "
       f"Categories: <b>{len(files_by_ds)}</b> &nbsp;&nbsp; "
       f"Unique source files: <b>{total_files}</b> &nbsp;&nbsp; "
       f"Total chunks: <b>{health['points']:,}</b><br/>"
       f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;&middot;&nbsp; "
       f"Host: {RIG}")
story.append(Paragraph(sub, styles["SubHead"]))
story.append(Paragraph(
    "Depth indicator: the number of <i>distinct source PDFs</i> ingested per category. "
    "A category with 1 file has narrow coverage; a category with many files spans multiple acts, "
    "rules, and circulars. Chunk counts indicate how much retrievable text each file contributed "
    "after section-aware chunking (~180 words, 30 overlap, 1000-char cap).",
    styles["Body"]))
story.append(Spacer(1, 10))

# ---- Summary table: files per category ----
story.append(Paragraph("Summary — depth per category", styles["H2"]))
cats_sorted = sorted(files_by_ds.items(), key=lambda kv: -count_of.get(kv[0], 0))
sum_rows = [["Category", "Files", "Chunks", "Depth"]]

def depth_tag(n_files):
    if n_files >= 20: return "Wide"
    if n_files >= 5:  return "Medium"
    if n_files >= 2:  return "Narrow"
    return "Single-source"

for ds, files in cats_sorted:
    sum_rows.append([
        label_of.get(ds, ds),
        str(len(files)),
        f"{count_of.get(ds,0):,}",
        depth_tag(len(files)),
    ])
sum_tbl = Table(sum_rows, colWidths=[9.4*cm, 2*cm, 2.4*cm, 3.4*cm], repeatRows=1)
sum_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a4480")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("ALIGN", (1,0), (3,-1), "CENTER"),
    ("ALIGN", (0,0), (0,-1), "LEFT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f3f5f9")]),
    ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#bbb")),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(sum_tbl)

# ---- Per-category detail ----
story.append(PageBreak())
story.append(Paragraph("Per-category file listing", styles["H2"]))
story.append(Paragraph(
    "Each category below lists every distinct source file contributing to its retrievable "
    "corpus, sorted by chunk count (highest first). Hash-named files typically come from "
    "FEMA notifications where the original PDF filename is an RBI/MoF document ID.",
    styles["Body"]))
story.append(Spacer(1, 8))

for ds, files in cats_sorted:
    label = label_of.get(ds, ds)
    n_files = len(files)
    total_chunks = count_of.get(ds, 0)
    depth = depth_tag(n_files)

    # KeepTogether so the heading + first part of table stay together
    block = []
    block.append(Paragraph(
        f"{label} &nbsp;&middot;&nbsp; <font color='#555'>{ds}</font>",
        styles["CatHead"]))
    block.append(Paragraph(
        f"{n_files} file{'s' if n_files!=1 else ''} &nbsp;&middot;&nbsp; "
        f"{total_chunks:,} chunks &nbsp;&middot;&nbsp; depth: {depth}",
        styles["CatSub"]))

    rows = [["#", "Source file", "Chunks", "Share"]]
    files_sorted = sorted(files, key=lambda x: -x[1])
    for i, (fn, cnt) in enumerate(files_sorted, 1):
        share = cnt / total_chunks * 100 if total_chunks else 0
        pretty = prettify_filename(fn)
        note = filename_note(fn)
        cell = Paragraph(pretty + (("<br/>" + note) if note else ""), styles["FileCell"])
        rows.append([str(i), cell, f"{cnt:,}", f"{share:.1f}%"])

    tbl = Table(rows, colWidths=[0.8*cm, 11.4*cm, 2.2*cm, 2.4*cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2a5a9f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ALIGN", (0,0), (0,-1), "CENTER"),
        ("ALIGN", (2,0), (3,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7fb")]),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#bbb")),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    block.append(tbl)
    block.append(Spacer(1, 8))
    # For small tables, KeepTogether; for large ones (GST 41 files), let it flow
    if n_files <= 12:
        story.append(KeepTogether(block))
    else:
        story.extend(block)

# ---- Final notes ----
story.append(PageBreak())
story.append(Paragraph("Notes", styles["H2"]))
notes = [
    ("Single-source categories (1 file)",
     "9 categories rely on a single source PDF: Constitution, CPC, IBC, RERA, Disaster Mgmt, "
     "ADR, Telecom, Interpretation, Workplace. These cover their domain's principal act well "
     "but lack subordinate rules/circulars/regulations. For SaaS-grade coverage these should "
     "be expanded with rules, notifications, and guidance material."),
    ("Widest categories",
     "GST (Acts, Rules, Circulars) — 41 files (widest). FEMA Notifications — 16 files. "
     "Other Bare Acts — 20 files (a catch-all for contract, succession, stamp, limitation, etc.)."),
    ("Depth vs breadth trade-off",
     "A narrow but deep category (e.g. Income Tax: 2 files, 2,678 chunks) can answer questions "
     "about that one act exhaustively. A wide but shallow category (e.g. FEMA Notifications: "
     "16 files, 64 chunks) can reference many sources but each only shallowly. Phase 4's v3 "
     "ingester with unstructured[pdf] targets improved depth-per-file via better proviso and "
     "table handling."),
    ("Hash-named files",
     "FEMA master directions and some RBI notifications are named by their document hash on "
     "the RBI site (e.g. FEMA3RA16022026D88E275FB01B422FBA98D391984C828D). A post-ingest "
     "rename pass could map these to human-readable labels. For now the full hash is retained "
     "as the authoritative file identifier."),
]
for title, body in notes:
    story.append(Paragraph(f"<b>{title}</b>", styles["Body"]))
    story.append(Paragraph(body, styles["FileCell"]))
    story.append(Spacer(1, 6))

doc.build(story)
print(f"Wrote {OUT} ({os.path.getsize(OUT):,} bytes)")
