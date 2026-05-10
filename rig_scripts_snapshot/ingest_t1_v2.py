#!/usr/bin/env python3
"""Tier 1 v2 ingester — hierarchical + hybrid (dense + BM25 sparse) + status metadata.

New collection: indian_legal_t1_v2
  dense  : BGE-small 384d (existing 4 embedders on 9092-9095)
  sparse : BM25 (computed in-script, uploaded as Qdrant native sparse vector)

Payload schema:
  source, text, dataset, tier=1
  act_name, act_number, act_year, section_no, chapter_no, anchor
  status: 'current'|'legacy'|'amendment'
  is_amendment: bool
  primary_target: str            # for Finance Acts etc.
  effective_from: str YYYY-MM-DD
  effective_until: str           # for repealed acts
  file, pdf_sha256
  type: 'statute_chunk'|'mapping_bridge'|'concordance_chunk'

Run: python3 ingest_t1_v2.py --all
"""
import os, sys, re, json, time, uuid, hashlib, argparse, glob, math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error
import fitz

# ----- config -----
EMBED_PORTS = [9092, 9093, 9094, 9095]
QDRANT = "http://localhost:6333"
COLL = "indian_legal_t1_v2"
BATCH = 16
CHAR_CAP = 1000
WORDS_PER = 180
OVERLAP = 30
STAGE = "/opt/indian-legal-ai/gst_stage"
BM25_K1 = 1.5
BM25_B = 0.75

# ----- hierarchical section regex -----
SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:Section|SECTION|Sec\.?)\s+(\d+[A-Z]*)\.?\s*",
    re.I)
CHAPTER_RE = re.compile(
    r"(?:^|\n)\s*(?:CHAPTER|Chapter|PART|Part)\s+([IVXLC]+)\b", re.I)
ANCHOR_RE = re.compile(
    r"(?:^|\n)\s*(Section\s+\d+[A-Z]*|Rule\s+\d+[A-Z]*|Article\s+\d+[A-Z]*|"
    r"Chapter\s+[IVXLC]+|CHAPTER\s+[IVXLC]+|PART\s+[IVXLC]+|SCHEDULE\s+[IVXLC]*)\b",
    re.I)

# ----- metadata rules -----
# (category, file) -> overrides dict
FILE_OVERRIDES = {
    ("t1_income_tax", "income_tax_act_1961.pdf"): {
        "status": "legacy", "effective_until": "2026-03-31",
        "act_name": "Income Tax Act, 1961", "act_year": 1961, "act_number": "43 of 1961",
    },
    ("t1_income_tax", "income_tax_act_2025.pdf"): {
        "status": "current", "effective_from": "2026-04-01",
        "act_name": "Income Tax Act, 2025", "act_year": 2025, "act_number": "12 of 2025",
    },
}

# category-wide defaults
CAT_DEFAULTS = {
    "t1_old_criminal":        {"status": "legacy",  "effective_until": "2024-07-01",
                               "replaced_by": "Bharatiya Nyaya Sanhita 2023 / BNSS 2023 / BSA 2023"},
    "t1_criminal_codes_2023": {"status": "current", "effective_from": "2024-07-01"},
    "t1_finance_acts":        {"status": "current", "is_amendment": True,
                               "primary_target": "Income Tax Act, 1961"},
    "t1_pre_codified_labour": {"status": "legacy",
                               "replaced_by": "Labour Codes 2019/2020"},
    "t1_labour_codes":        {"status": "current"},
    "t1_bridges":             {"status": "current", "type_override": "concordance_chunk"},
}

CAT_TOPIC = {
    "t1_adr": "Alternative Dispute Resolution",
    "t1_banking": "Banking / Insurance / Benami",
    "t1_bridges": "Criminal code concordance (IPC↔BNS etc.)",
    "t1_civil_procedure": "Civil Procedure",
    "t1_commercial_acts": "Commercial law",
    "t1_companies_sebi": "Corporate / Securities",
    "t1_constitution": "Constitution of India",
    "t1_criminal_codes_2023": "New criminal codes (current)",
    "t1_criminal_special": "Special criminal laws (NDPS, UAPA, PC, Arms)",
    "t1_customs_excise": "Customs and Central Excise",
    "t1_data_privacy": "Data protection / Aadhaar",
    "t1_disaster_mgmt": "Disaster Management",
    "t1_environment": "Environment / Wildlife",
    "t1_fema_notifications": "FEMA notifications (RBI)",
    "t1_fema_rbi": "FEMA principal Act",
    "t1_finance_acts": "Finance Acts (annual amendments)",
    "t1_gst_circulars": "GST acts, rules, circulars",
    "t1_ibc": "Insolvency and Bankruptcy",
    "t1_income_tax": "Income Tax",
    "t1_interpretation": "General Clauses (statutory interpretation)",
    "t1_ip_acts": "Intellectual Property",
    "t1_labour_codes": "New Labour Codes (current)",
    "t1_old_criminal": "Legacy criminal codes (repealed)",
    "t1_other_bare_acts": "Miscellaneous bare acts",
    "t1_personal_law": "Personal law (marriage/succession)",
    "t1_pre_codified_labour": "Pre-2019 labour laws (repealed)",
    "t1_real_estate": "RERA",
    "t1_telecom": "Telecom",
    "t1_workplace": "POSH",
}

YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")

# ----- BM25 tokenizer -----
STOPWORDS = set("""a an the of in on at by for to from with and or but not is are was were be been
being have has had do does did can could would should shall will may might must this that these
those it its as if then than there here which who whom whose what when where why how""".split())

TOK_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")


def tokenize(text):
    out = []
    for m in TOK_RE.findall(text.lower()):
        if m in STOPWORDS or len(m) < 2:
            continue
        out.append(m)
    return out


def token_to_idx(tok):
    # 32-bit stable hash — Qdrant sparse vector indices are uint32
    h = hashlib.md5(tok.encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


# ----- HTTP helpers -----
def http_json(url, method="POST", body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed_batch(port, texts):
    return [e["embedding"] for e in http_json(
        f"http://localhost:{port}/v1/embeddings",
        body={"input": texts, "model": "bge"})["data"]]


def embed_batch_safe(port, texts):
    try:
        return embed_batch(port, texts)
    except urllib.error.HTTPError as e:
        if e.code != 400 or len(texts) == 1:
            return [None] * len(texts)
        mid = len(texts) // 2
        return embed_batch_safe(port, texts[:mid]) + embed_batch_safe(port, texts[mid:])


# ----- collection management -----
def ensure_collection():
    # delete if exists then create with named dense + sparse
    try:
        http_json(f"{QDRANT}/collections/{COLL}", method="DELETE")
    except urllib.error.HTTPError:
        pass
    http_json(f"{QDRANT}/collections/{COLL}", method="PUT", body={
        "vectors": {
            "dense": {"size": 384, "distance": "Cosine"}
        },
        "sparse_vectors": {
            "bm25": {"modifier": "idf"}  # Qdrant-native IDF modifier
        }
    })
    print(f"[ok] collection {COLL} created")


def create_payload_indexes():
    fields = [
        ("dataset", "keyword"), ("act_name", "keyword"), ("act_year", "integer"),
        ("status", "keyword"), ("is_amendment", "bool"), ("section_no", "keyword"),
        ("chapter_no", "keyword"), ("file", "keyword"), ("type", "keyword"),
        ("primary_target", "keyword"),
    ]
    for f, t in fields:
        try:
            http_json(f"{QDRANT}/collections/{COLL}/index", method="PUT",
                      body={"field_name": f, "field_schema": t})
            print(f"  [idx] {f} ({t})")
        except Exception as e:
            print(f"  [idx skip] {f}: {e}")


# ----- PDF -> chunks with hierarchy -----
def extract_hierarchy_chunks(path):
    """Walk pages, track current chapter + section, emit chunks with real hierarchy."""
    doc = fitz.open(path)
    pages = [p.get_text("text") for p in doc]
    doc.close()
    full = "\n\n".join(pages)
    full = re.sub(r"[ \t]+", " ", full)
    full = re.sub(r"\n{3,}", "\n\n", full)

    paras = [p.strip() for p in re.split(r"\n\s*\n", full) if p.strip()]
    chunks = []
    buf, buf_words = [], 0
    cur_chap, cur_sec, cur_anchor = "", "", ""

    def flush():
        nonlocal buf, buf_words
        if buf_words > 20:
            chunks.append({
                "text": " ".join(buf),
                "chapter_no": cur_chap,
                "section_no": cur_sec,
                "anchor": cur_anchor,
            })
        tail = " ".join(buf).split()[-OVERLAP:]
        buf = [" ".join(tail)] if tail else []
        buf_words = len(tail)

    for p in paras:
        chm = CHAPTER_RE.search(p)
        sm = SECTION_RE.search(p)
        am = ANCHOR_RE.search(p)
        # on chapter or section transition (and buf has content), flush
        if (chm or sm) and buf_words > 60:
            flush()
        if chm: cur_chap = chm.group(1).strip()
        if sm:  cur_sec = sm.group(1).strip()
        if am:  cur_anchor = am.group(1).strip()
        buf.append(p)
        buf_words += len(p.split())
        if buf_words >= WORDS_PER:
            flush()
    if buf_words > 20:
        chunks.append({
            "text": " ".join(buf),
            "chapter_no": cur_chap,
            "section_no": cur_sec,
            "anchor": cur_anchor,
        })
    return chunks


def derive_year(fname):
    m = YEAR_RE.search(fname)
    return int(m.group(0)) if m else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()


def build_payload(cat, fname, pdf_sha, ch, defaults, override):
    act_name = re.sub(r"\.pdf$", "", fname, flags=re.I).replace("_", " ")
    year = derive_year(fname)
    status = "current"
    payload = {
        "text": ch["text"],
        "dataset": cat,
        "tier": 1,
        "act_name": act_name,
        "act_year": year,
        "section_no": ch["section_no"],
        "chapter_no": ch["chapter_no"],
        "anchor": ch["anchor"],
        "file": fname,
        "pdf_sha256": pdf_sha,
        "type": defaults.get("type_override", "statute_chunk"),
        "status": status,
    }
    # apply category defaults
    for k, v in defaults.items():
        if k == "type_override": continue
        payload[k] = v
    # apply file-specific overrides
    payload.update(override)
    # build hierarchical source string: [act > chapter > section | status]
    hier = payload.get("act_name", "")
    if ch["chapter_no"]:
        hier += f" > Ch. {ch['chapter_no']}"
    if ch["section_no"]:
        hier += f" > Sec. {ch['section_no']}"
    hier += f" | {payload['status'].upper()}"
    payload["source"] = f"[{hier}] {fname}"
    # contextual prepend in text for richer dense embedding
    payload["text"] = f"[{hier}]\n{ch['text']}"
    if len(payload["text"]) > CHAR_CAP:
        payload["text"] = payload["text"][:CHAR_CAP]
    return payload


def point_id_for(payload):
    key = f"{payload['dataset']}|{payload['file']}|{payload['source']}|{payload['text'][:120]}"
    h = hashlib.sha1(key.encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


# ----- build IDF over corpus -----
def collect_chunks_all():
    """Pass 1: iterate all T1 dirs and extract chunks + payloads (no embed yet)."""
    all_items = []  # list of (payload, tokens)
    for cat in sorted(d for d in os.listdir(STAGE)
                       if d.startswith("t1_") and os.path.isdir(os.path.join(STAGE, d))):
        defaults = CAT_DEFAULTS.get(cat, {})
        pdfs = sorted(glob.glob(os.path.join(STAGE, cat, "*.pdf")))
        if not pdfs: continue
        print(f"  scan {cat}  ({len(pdfs)} pdfs)", flush=True)
        for p in pdfs:
            fname = os.path.basename(p)
            override = FILE_OVERRIDES.get((cat, fname), {})
            try:
                pdf_sha = sha256_file(p)
                chunks = extract_hierarchy_chunks(p)
            except Exception as e:
                print(f"    x {fname}: {e}")
                continue
            for ch in chunks:
                pay = build_payload(cat, fname, pdf_sha, ch, defaults, override)
                toks = tokenize(pay["text"])
                if not toks: continue
                all_items.append((pay, toks))
    return all_items


def compute_idf(items):
    """Standard IDF = log((N - df + 0.5)/(df + 0.5) + 1)."""
    N = len(items)
    df = Counter()
    for _, toks in items:
        for t in set(toks):
            df[t] += 1
    idf = {t: math.log((N - v + 0.5) / (v + 0.5) + 1) for t, v in df.items()}
    avg_dl = sum(len(t) for _, t in items) / max(N, 1)
    return idf, avg_dl


def bm25_sparse(tokens, idf, avg_dl):
    """Return Qdrant sparse-vector dict: {indices: [int], values: [float]}."""
    tf = Counter(tokens)
    dl = len(tokens)
    indices, values = [], []
    for tok, f in tf.items():
        if tok not in idf: continue
        num = f * (BM25_K1 + 1)
        den = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / max(avg_dl, 1))
        w = idf[tok] * num / max(den, 1e-6)
        if w <= 0: continue
        indices.append(token_to_idx(tok))
        values.append(float(w))
    return {"indices": indices, "values": values}


# ----- bridges.jsonl ingest (if present) -----
def load_bridges():
    path = os.path.join(STAGE, "t1_bridges", "bridges.jsonl")
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: items.append(json.loads(line))
            except: pass
    return items


def bridge_to_payload(b):
    old_act = b.get("old_act", "")
    new_act = b.get("new_act", "")
    hier = f"Bridge: {new_act} Sec {b.get('new_section','?')} ← {old_act} Sec {b.get('old_section','?')}"
    return {
        "text": b.get("context_prepend", "") + "\n" + b.get("text", ""),
        "dataset": "t1_bridges",
        "tier": 1,
        "act_name": f"{old_act} -> {new_act} bridge",
        "type": "mapping_bridge",
        "status": "current",
        "old_act": old_act, "old_section": b.get("old_section", ""),
        "old_title": b.get("old_title", ""),
        "new_act": new_act, "new_section": b.get("new_section", ""),
        "new_title": b.get("new_title", ""),
        "topic": b.get("topic", ""),
        "source": f"[{hier}]",
        "file": "bridges.jsonl",
    }


# ----- main pipeline -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--recreate", action="store_true", help="drop and recreate collection")
    args = ap.parse_args()

    t0 = time.time()
    print("=== T1 v2 ingestion ===")

    if args.recreate:
        ensure_collection()
        create_payload_indexes()

    print("\n[pass 1] collecting chunks + building IDF")
    items = collect_chunks_all()
    print(f"  total chunks: {len(items)}")

    # add bridges.jsonl records if present
    bridges = load_bridges()
    if bridges:
        print(f"  + {len(bridges)} mapping_bridge records from bridges.jsonl")
        for b in bridges:
            pay = bridge_to_payload(b)
            toks = tokenize(pay["text"])
            if toks:
                items.append((pay, toks))

    idf, avg_dl = compute_idf(items)
    # persist idf for query-time use
    idf_out = "/opt/indian-legal-ai/rag/bm25_idf_v2.json"
    os.makedirs(os.path.dirname(idf_out), exist_ok=True)
    with open(idf_out, "w") as f:
        json.dump({"idf": idf, "avg_dl": avg_dl, "bm25_k1": BM25_K1, "bm25_b": BM25_B}, f)
    print(f"  idf saved -> {idf_out} ({len(idf)} tokens, avg_dl={avg_dl:.1f})")

    print("\n[pass 2] dense embed + sparse compute + upsert")
    n = len(items)
    batches = [items[i:i + BATCH] for i in range(0, n, BATCH)]
    upserted = dropped = 0

    def embed_batch_task(args_):
        idx, batch = args_
        port = EMBED_PORTS[idx % len(EMBED_PORTS)]
        texts = [p["text"] for p, _ in batch]
        vecs = embed_batch_safe(port, texts)
        return idx, vecs

    with ThreadPoolExecutor(max_workers=len(EMBED_PORTS)) as ex:
        futs = {ex.submit(embed_batch_task, (i, b)): (i, b) for i, b in enumerate(batches)}
        for f in as_completed(futs):
            i, b = futs[f]
            idx, vecs = f.result()
            pts = []
            for (pay, toks), v in zip(b, vecs):
                if v is None:
                    dropped += 1
                    continue
                sparse = bm25_sparse(toks, idf, avg_dl)
                if not sparse["indices"]:
                    dropped += 1
                    continue
                pid = point_id_for(pay)
                pts.append({
                    "id": pid,
                    "vector": {"dense": v, "bm25": sparse},
                    "payload": pay,
                })
            if not pts: continue
            for attempt in range(3):
                try:
                    http_json(f"{QDRANT}/collections/{COLL}/points?wait=false",
                              method="PUT", body={"points": pts})
                    upserted += len(pts)
                    break
                except Exception as e:
                    print(f"  upsert retry {attempt}: {e}")
                    time.sleep(1 + attempt)
            if idx % 20 == 0:
                dt = time.time() - t0
                print(f"  batch {idx+1}/{len(batches)}  up={upserted}  drop={dropped}  "
                      f"rate={upserted/max(dt,1):.1f}/s  el={dt:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"\n=== DONE  upserted={upserted}/{n}  dropped={dropped}  elapsed={dt:.1f}s ===")

    # final counts by status / dataset
    print("\n=== VERIFY ===")
    cnt = http_json(f"{QDRANT}/collections/{COLL}/points/count",
                    body={"exact": True})["result"]["count"]
    print(f"total points: {cnt}")
    for st in ("current", "legacy"):
        c = http_json(f"{QDRANT}/collections/{COLL}/points/count",
                      body={"exact": True,
                            "filter": {"must": [{"key": "status",
                                                  "match": {"value": st}}]}})["result"]["count"]
        print(f"  status={st}: {c}")


if __name__ == "__main__":
    main()
