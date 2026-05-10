#!/usr/bin/env python3
"""
fetch_gst_corpus.py  —  Build a curated GST dataset from authoritative public sources.

Writes JSONL files to /opt/indian-legal-ai/datasets/_curated/:

  gst_acts.jsonl      Tier 1  CGST, IGST, UTGST, Compensation Cess Acts + amendments (from PRS)
  gst_rates.jsonl     Tier 1  HSN-wise rate schedules (from archive.org CBIC cache)
  gst_wiki.jsonl      Tier 3  Wikipedia GST overview incl. 2023 online-gaming amendment
  gst_council.jsonl   Tier 2  GST Council homepage + press releases

Each record has the curated schema used by build_rag_worker_tiered.py:
    {"text": "...", "source": "...", "dataset": "gst_acts", "tier": 1}
"""

import os, re, json, time, io
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
import fitz  # pymupdf

OUT_DIR = "/opt/indian-legal-ai/datasets/_curated"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
HDRS = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/pdf,*/*"}

# ---------- target URLs ---------------------------------------------------

PRS_ACTS = [
    ("cgst_act_2017",         "CGST Act 2017",          "https://prsindia.org/files/bills_acts/acts_parliament/2017/the-central-goods-and-services-tax-act,-2017.pdf"),
    ("igst_act_2017",         "IGST Act 2017",          "https://prsindia.org/files/bills_acts/acts_parliament/2017/the-integrated-goods-and-services-tax-act,-2017.pdf"),
    ("utgst_act_2017",        "UTGST Act 2017",         "https://prsindia.org/files/bills_acts/acts_parliament/2017/the-union-territory-goods-and-services-tax-act,-2017.pdf"),
    ("gst_comp_cess_act_2017","GST Compensation Cess Act 2017", "https://prsindia.org/files/bills_acts/acts_parliament/2017/the-goods-and-services-tax-(compensation-to-states)-act,-2017.pdf"),
    ("cgst_amendment_2018",   "CGST Amendment Act 2018","https://prsindia.org/files/bills_acts/acts_parliament/2018/the-central-goods-and-services-tax-(amendment)-act,-2018.pdf"),
    ("cgst_amendment_2023",   "CGST Amendment Act 2023 (online gaming, casinos, horse racing)",
        "https://web.archive.org/web/2024/https://cbic-gst.gov.in/pdf/CGST-Amendment-Act-2023.pdf"),
    ("igst_amendment_2023",   "IGST Amendment Act 2023 (online money gaming / OIDAR)",
        "https://web.archive.org/web/2024/https://cbic-gst.gov.in/pdf/IGST-Amendment-Act-2023.pdf"),
]

ARCHIVE_HTML = [
    ("cbic_gst_acts_index", "CBIC — GST Acts index",                    "https://web.archive.org/web/2024/https://cbic-gst.gov.in/gst-acts.html"),
    ("cbic_gst_rates",      "CBIC — GST goods & services rates (HSN)",  "https://web.archive.org/web/2024/https://cbic-gst.gov.in/gst-goods-services-rates.html"),
    ("cbic_circulars",      "CBIC — Central tax circulars",             "https://web.archive.org/web/2024/https://cbic-gst.gov.in/central-tax-circulars.html"),
    ("cbic_notifications",  "CBIC — Central tax notifications",         "https://web.archive.org/web/2024/https://cbic-gst.gov.in/central-tax-notfns.html"),
    ("cbic_rate_notfns",    "CBIC — Central tax rate notifications",    "https://web.archive.org/web/2024/https://cbic-gst.gov.in/central-tax-notfns-rate.html"),
    ("cbic_faqs",           "CBIC — GST FAQs",                          "https://web.archive.org/web/2024/https://cbic-gst.gov.in/gst-faqs.html"),
    ("cbic_igst_acts",      "CBIC — IGST Act",                          "https://web.archive.org/web/2024/https://cbic-gst.gov.in/igst-act.html"),
]

WIKI_PAGES = [
    ("wiki_gst_india",      "Wikipedia — Goods and Services Tax (India)",        "https://en.wikipedia.org/wiki/Goods_and_Services_Tax_(India)"),
    ("wiki_gst_council",    "Wikipedia — GST Council",                           "https://en.wikipedia.org/wiki/GST_Council"),
    ("wiki_gstn",           "Wikipedia — GSTN",                                  "https://en.wikipedia.org/wiki/Goods_and_Services_Tax_Network"),
    ("wiki_online_gaming",  "Wikipedia — Online gambling in India",              "https://en.wikipedia.org/wiki/Online_gambling_in_India"),
    ("wiki_gst_search",     "Wikipedia — search 'GST online gaming 2023'",       "https://en.wikipedia.org/wiki/Special:Search?search=GST+online+gaming+amendment+2023"),
]

COUNCIL_HTML = [
    ("council_home",       "GST Council — home",          "https://gstcouncil.gov.in/"),
    ("council_members",    "GST Council — members",       "https://gstcouncil.gov.in/gst-council-member-list"),
    ("council_press",      "GST Council — press releases","https://gstcouncil.gov.in/press-release"),
]


# ---------- HTTP -----------------------------------------------------------

def fetch(url: str, timeout: int = 30) -> bytes | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HDRS, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 500:
                return r.content
            print(f"  HTTP {r.status_code} size={len(r.content)} (attempt {attempt+1})")
        except Exception as e:
            print(f"  {type(e).__name__}: {e} (attempt {attempt+1})")
        time.sleep(1 + attempt)
    return None


# ---------- PDF -> text ---------------------------------------------------

def pdf_to_text(pdf_bytes: bytes) -> str:
    parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    # Normalize whitespace but keep paragraph breaks
    text = "\n".join(parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------- HTML -> text --------------------------------------------------

def html_to_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    # Try main/article, fall back to body
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Drop wayback toolbar strings
    text = re.sub(r"Wayback Machine[\s\S]*?(?=\n\n|$)", "", text)
    text = re.sub(r"Internet Archive[\s\S]{0,200}", "", text)
    return text


# ---------- chunking ------------------------------------------------------

# Splitters preferred for statutes
SECTION_SPLIT = re.compile(
    r"(?=^\s*(?:Section|SECTION|Sec\.|Chapter|CHAPTER|Article|ARTICLE|Rule|RULE|Schedule)\s+[\dIVXLC]+[A-Z]?\.?\b)",
    re.MULTILINE,
)

def split_into_logical_chunks(text: str, target: int = 1400, hard_max: int = 2200) -> list[str]:
    # First try section-aware split
    parts = SECTION_SPLIT.split(text)
    parts = [p.strip() for p in parts if p and p.strip()]

    # Merge consecutive tiny parts and cut long parts
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 2 <= target:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    # Further split any oversize chunk on paragraph boundaries
    final: list[str] = []
    for c in chunks:
        if len(c) <= hard_max:
            final.append(c)
            continue
        paras = re.split(r"\n\n+", c)
        cur = ""
        for para in paras:
            if len(cur) + len(para) + 2 <= target:
                cur = (cur + "\n\n" + para).strip() if cur else para
            else:
                if cur:
                    final.append(cur)
                cur = para
                while len(cur) > hard_max:
                    final.append(cur[:hard_max])
                    cur = cur[hard_max:]
        if cur:
            final.append(cur)
    return [c for c in final if len(c) >= 120]


# ---------- writer --------------------------------------------------------

def write_jsonl(rows: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> wrote {len(rows):,} rows to {path}")


# ---------- pipeline ------------------------------------------------------

def process_pdf_group(dataset: str, tier: int, items: list[tuple]) -> list[dict]:
    rows = []
    for short, title, url in items:
        print(f"[{dataset}] fetching {title}")
        data = fetch(url)
        if not data:
            print(f"  SKIP: could not fetch")
            continue
        text = pdf_to_text(data)
        if not text.strip():
            print(f"  SKIP: empty text")
            continue
        chunks = split_into_logical_chunks(text)
        print(f"  {len(text):,} chars -> {len(chunks)} chunks")
        for i, ch in enumerate(chunks):
            rows.append({
                "text": ch,
                "source": f"{title} [§{i+1}]",
                "dataset": dataset,
                "tier": tier,
            })
    return rows


def process_html_group(dataset: str, tier: int, items: list[tuple]) -> list[dict]:
    rows = []
    for short, title, url in items:
        print(f"[{dataset}] fetching {title}")
        data = fetch(url)
        if not data:
            print(f"  SKIP: could not fetch")
            continue
        text = html_to_text(data)
        # Remove archive.org chrome that leaks in sometimes
        text = re.sub(r"(?mi)^.*web\.archive\.org.*$", "", text)
        if not text.strip() or len(text) < 300:
            print(f"  SKIP: too little text ({len(text)} chars)")
            continue
        chunks = split_into_logical_chunks(text)
        print(f"  {len(text):,} chars -> {len(chunks)} chunks")
        for i, ch in enumerate(chunks):
            rows.append({
                "text": ch,
                "source": f"{title} [§{i+1}]",
                "dataset": dataset,
                "tier": tier,
            })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n=== 1. GST Acts (PRS PDFs) [tier 1] ===")
    acts = process_pdf_group("gst_acts", 1, PRS_ACTS)
    write_jsonl(acts, f"{OUT_DIR}/gst_acts.jsonl")

    print("\n=== 2. CBIC rates + circulars (archive.org) [tier 1] ===")
    rates = process_html_group("gst_rates", 1, ARCHIVE_HTML)
    write_jsonl(rates, f"{OUT_DIR}/gst_rates.jsonl")

    print("\n=== 3. Wikipedia GST (incl. 2023 amendment) [tier 3] ===")
    wiki = process_html_group("gst_wiki", 3, WIKI_PAGES)
    write_jsonl(wiki, f"{OUT_DIR}/gst_wiki.jsonl")

    print("\n=== 4. GST Council pages [tier 2] ===")
    council = process_html_group("gst_council", 2, COUNCIL_HTML)
    write_jsonl(council, f"{OUT_DIR}/gst_council.jsonl")

    total = len(acts) + len(rates) + len(wiki) + len(council)
    print(f"\nTOTAL rows across 4 GST files: {total:,}")


if __name__ == "__main__":
    main()
