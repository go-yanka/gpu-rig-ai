"""Generate a PDF report of the Indian Legal AI ingested corpus."""
import json, urllib.request, os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether,
)

RIG = "http://192.168.1.107:8090"

def fetch(path):
    with urllib.request.urlopen(f"{RIG}{path}", timeout=20) as r:
        return json.loads(r.read())

print("Fetching /health and /meta from rig...")
health = fetch("/health")
meta = fetch("/meta")

# ---- Styles ----
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"],
                          fontSize=22, textColor=colors.HexColor("#1a4480"),
                          spaceAfter=4, alignment=0))
styles.add(ParagraphStyle(name="SubHead", parent=styles["Normal"],
                          fontSize=10, textColor=colors.HexColor("#555"),
                          spaceAfter=12))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"],
                          fontSize=14, textColor=colors.HexColor("#1a4480"),
                          spaceBefore=16, spaceAfter=8))
styles.add(ParagraphStyle(name="H3", parent=styles["Heading3"],
                          fontSize=11, textColor=colors.HexColor("#2a5a9f"),
                          spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"],
                          fontSize=9.5, leading=13))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"],
                          fontSize=8.5, textColor=colors.HexColor("#666"), leading=11))

# ---- Document ----
OUT = "D:/_gpu_rig_ai/corpus_ingested_report.pdf"
doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=1.8*cm, rightMargin=1.8*cm,
                        topMargin=1.6*cm, bottomMargin=1.6*cm,
                        title="Indian Legal AI — Ingested Corpus Report")
story = []

# Title block
story.append(Paragraph("Indian Legal AI — Ingested Corpus", styles["TitleBig"]))
sub = (f"Collection: <b>{health.get('collection','?')}</b> &nbsp;&nbsp; "
       f"Points: <b>{health.get('points','?'):,}</b> &nbsp;&nbsp; "
       f"BM25 IDF terms: <b>{health.get('idf_terms','?'):,}</b><br/>"
       f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;&nbsp; "
       f"Host: {RIG}")
story.append(Paragraph(sub, styles["SubHead"]))
story.append(Paragraph(
    f"<b>{len(meta['categories'])}</b> categories &nbsp;&middot;&nbsp; "
    f"<b>{len(meta['acts'])}</b> distinct act_names &nbsp;&middot;&nbsp; "
    f"<b>{health.get('points',0):,}</b> vector chunks "
    f"(dense BGE-384d + sparse BM25)", styles["Body"]))

# ---- Section 1: Categories ----
story.append(Paragraph("1. By Category", styles["H2"]))
story.append(Paragraph(
    "Each category maps to one <i>dataset</i> tag in the Qdrant payload. "
    "Counts reflect retrievable chunks, not source-file counts.",
    styles["Small"]))
story.append(Spacer(1, 4))

cats = sorted(meta["categories"], key=lambda c: -c["count"])
total_chunks = sum(c["count"] for c in cats)
cat_rows = [["#", "Category", "Dataset tag", "Chunks", "Share"]]
for i, c in enumerate(cats, 1):
    share = c["count"] / total_chunks * 100
    cat_rows.append([str(i), c["label"], c["value"],
                     f"{c['count']:,}", f"{share:.1f}%"])
cat_rows.append(["", Paragraph("<b>Total</b>", styles["Body"]), "",
                 f"{total_chunks:,}", "100.0%"])

cat_tbl = Table(cat_rows, colWidths=[0.8*cm, 7.6*cm, 4.6*cm, 2.4*cm, 1.8*cm],
                repeatRows=1)
cat_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a4480")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("ALIGN", (0,0), (0,-1), "CENTER"),
    ("ALIGN", (3,0), (4,-1), "RIGHT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.white, colors.HexColor("#f3f5f9")]),
    ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#dde4ef")),
    ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#bbb")),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(cat_tbl)

# ---- Section 2: Top acts ----
story.append(PageBreak())
story.append(Paragraph("2. Acts &mdash; Top 60 by chunk count", styles["H2"]))
story.append(Paragraph(
    f"Corpus contains {len(meta['acts'])} distinct <i>act_name</i> values. "
    "Shown below: the 60 most represented. Tail (82 acts with &le;20 chunks each) "
    "covers amendment-act fragments, individual circulars, and FEMA notifications.",
    styles["Small"]))
story.append(Spacer(1, 4))

acts_sorted = sorted(meta["acts"], key=lambda a: -a["count"])
top_n = 60
# 2-column layout
mid = (top_n + 1) // 2
left = acts_sorted[:mid]
right = acts_sorted[mid:top_n]
while len(right) < len(left):
    right.append({"value": "", "count": ""})

act_rows = [["#", "Act name", "Chunks", "#", "Act name", "Chunks"]]
for i in range(len(left)):
    l = left[i]
    r = right[i]
    act_rows.append([
        str(i+1), l["value"], f"{l['count']:,}" if l["count"] != "" else "",
        str(mid+i+1) if r["value"] else "", r["value"],
        f"{r['count']:,}" if r["count"] != "" else "",
    ])
act_tbl = Table(act_rows,
                colWidths=[0.8*cm, 6.8*cm, 1.6*cm, 0.8*cm, 6.8*cm, 1.6*cm],
                repeatRows=1)
act_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a4480")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8),
    ("ALIGN", (0,0), (0,-1), "CENTER"),
    ("ALIGN", (3,0), (3,-1), "CENTER"),
    ("ALIGN", (2,0), (2,-1), "RIGHT"),
    ("ALIGN", (5,0), (5,-1), "RIGHT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f3f5f9")]),
    ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#bbb")),
    ("LINEAFTER", (2,0), (2,-1), 0.8, colors.HexColor("#1a4480")),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(act_tbl)

# ---- Section 3: GST sub-corpus ----
story.append(PageBreak())
story.append(Paragraph("3. GST sub-corpus detail", styles["H2"]))
story.append(Paragraph(
    "The <b>GST (Acts, Rules, Circulars)</b> category contains 1,287 chunks. "
    "Grouped breakdown of the underlying sources:",
    styles["Body"]))
story.append(Spacer(1, 6))

gst_groups = [
    ("Acts", [
        ("CGST Act 2022", 289),
        ("IGST Act 2017", 18),
        ("UTGST Act 2017", 15),
        ("CGST Amendment Act 2023", 3),
        ("IGST Amendment Act 2023", 3),
    ]),
    ("Rules", [
        ("CGST Rules (01-Jul-2017)", 437),
        ("ITC Rules (17-May-2017)", 26),
        ("Return Rules (03-Jun-2017)", 15),
        ("Anti-Profiteering Rules", 11),
        ("Invoice Rules (17-May-2017)", 10),
        ("Payment Rules (17-May-2017)", 9),
        ("Refund Rules (17-May-2017)", 9),
        ("Account Record Rules (11-Jun-2017)", 8),
        ("Assessment & Audit Rules", 8),
        ("Valuation Rules (17-May-2017)", 8),
        ("Appeal & Revision Rules", 7),
        ("Transition Rules (04-Jun-2017)", 5),
        ("Account Record Format Rules", 4),
        ("Advance Ruling Authority Rules", 2),
        ("E-way Bill Rules", 1),
    ]),
    ("Rate schedules", [
        ("Chapter-wise Rate Schedule (18-May-2017)", 159),
        ("Services Rate Schedule", 19),
        ("GST Compensation Cess Rates", 4),
        ("Addendum Rate Schedule (22-May-2017)", 2),
    ]),
    ("Notifications", [
        ("Notification 3 Central Tax", 93),
    ]),
    ("Circulars (2024–2025)", [
        ("Circular 238-32-2024", 38),
        ("Services Circular 228-22-2024", 12),
        ("Circular 237-31-2024", 11),
        ("Circular 232", 9),
        ("Circular 55th GSTC Services", 8),
        ("Circular 230", 6),
        ("Circular 231", 6),
        ("CGST Circular 246-03-2025", 5),
        ("Circular 229-23-2024", 5),
        ("Circular 239-33-2024", 5),
        ("Circular 233", 4),
        ("Circular 250-2025", 3),
        ("Circular Co-insurance/Re-insurance", 3),
        ("Circular 249-2025", 2),
        ("Corrigendum 237-31-2024", 1),
    ]),
]
for group, items in gst_groups:
    subtotal = sum(c for _, c in items)
    story.append(Paragraph(f"{group} &mdash; {subtotal:,} chunks", styles["H3"]))
    rows = [["Source", "Chunks"]]
    for name, cnt in items:
        rows.append([name, f"{cnt:,}"])
    t = Table(rows, colWidths=[11.6*cm, 2*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2a5a9f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f3f5f9")]),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#bbb")),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

# ---- Section 4: Gaps ----
story.append(PageBreak())
story.append(Paragraph("4. Known gaps &amp; ingestion notes", styles["H2"]))

gap_items = [
    ("State GST Acts (SGSTs)",
     "Not ingested. Only central-level CGST, IGST, and UTGST are present. "
     "29 state-level SGST Acts would need to be added for full GST coverage."),
    ("Older Finance Acts (2020–2023)",
     "Only Finance Acts 2024, 2025, 2026 are ingested. Pre-2024 Finance Acts — "
     "which contain most historical amendments — are missing."),
    ("BNS 2023 under-indexed",
     "BNS 2023 has only 79 chunks vs IPC 1860's 480. Likely a PDF-extraction "
     "shortfall in the current ingester; queued for re-ingest in Phase 4 "
     "using unstructured[pdf] for better table/proviso handling."),
    ("FEMA Master Directions",
     "Present but with hash-named labels (e.g. FEMA3RA16022026D88E...). "
     "Human-readable labelling is a post-processing task."),
    ("IBBI subordinate regulations",
     "Only IBC Code 2016 ingested (147 chunks). IBBI's CIRP / Liquidation / "
     "Voluntary Liquidation regulations are absent."),
    ("SEBI regulations & circulars",
     "Only SEBI Act 1992 (52 chunks). SEBI's issuance of regulations "
     "(LODR, ICDR, AIF, etc.) and circular stream is not yet ingested."),
    ("Amendment-dated status tagging",
     "Chunks carry status=current/legacy, but fine-grained amendment-date "
     "provenance (e.g. \"this sub-section was inserted by Finance Act 2022\") "
     "is not yet tagged. Improves with Phase 4 v3 ingester."),
]
for title, body in gap_items:
    story.append(Paragraph(f"<b>{title}</b>", styles["Body"]))
    story.append(Paragraph(body, styles["Small"]))
    story.append(Spacer(1, 6))

# ---- Section 5: Technical footer ----
story.append(Paragraph("5. Technical pipeline", styles["H2"]))
tech_lines = [
    "<b>Vector store:</b> Qdrant 1.17.1, collection <i>indian_legal_t1_v2</i>, "
    "12,104 points, named vectors (dense 384d cosine + sparse BM25 with IDF modifier).",
    "<b>Embeddings:</b> BGE-small-en-v1.5 (384-dim) served via 4 llama.cpp workers on ports 9092–9095.",
    "<b>BM25 sparse:</b> MD5-hashed token indices, IDF from 20,401 terms, stored at "
    "<i>/opt/indian-legal-ai/rag/bm25_idf_v2.json</i>.",
    "<b>Reranker:</b> bge-reranker-v2-m3 on port 9096, prefix <i>\"Indian statutory law query: \"</i>, "
    "+1.5 boost for act-name matches.",
    "<b>LLM:</b> Llama 3.1 8B Instruct Q4_K_M on port 9086, grounded prompt, strict citation verifier "
    "(regex strip of unverified Section/§/Article/Rule claims).",
    "<b>Chunking:</b> section-aware, ~180 words with 30 overlap, 1000-char cap per chunk, "
    "metadata prepend (act_name / chapter / section / status).",
    "<b>Serving:</b> FastAPI + uvicorn on port 8090 at http://192.168.1.107:8090/.",
]
for line in tech_lines:
    story.append(Paragraph(line, styles["Body"]))
    story.append(Spacer(1, 4))

doc.build(story)
size = os.path.getsize(OUT)
print(f"Wrote {OUT} ({size:,} bytes)")
