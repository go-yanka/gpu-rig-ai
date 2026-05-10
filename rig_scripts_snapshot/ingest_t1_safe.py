#!/usr/bin/env python3
"""Safe T1 ingester: PDF -> section-aware chunks -> 4-port BGE embed -> Qdrant.

Fixes the 512-token context overflow from parallel_ingest.py by:
 - smaller target chunk (180 words, 30 overlap)
 - hard char cap of 1000 (~= 350 tokens worst-case for legal English)
 - per-item embed retry that drops the offending chunk instead of failing whole batch
 - skips empty / duplicate chunks by content sha1
 - richer payload metadata (act_name from filename, category, year, source_path)

Invocation:
  python3 ingest_t1_safe.py --root /opt/indian-legal-ai/gst_stage --cat t1_adr --tier 1
  python3 ingest_t1_safe.py --root /opt/indian-legal-ai/gst_stage --all  --tier 1
"""
import os, sys, re, json, time, uuid, hashlib, argparse, glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error
import fitz

EMBED_PORTS = [9092, 9093, 9094, 9095]
QDRANT = "http://localhost:6333"
COLL = "indian_legal_full"
BATCH = 16
CHAR_CAP = 1000
WORDS_PER = 180
OVERLAP = 30

SECTION_RE = re.compile(
    r"(?:^|\n)\s*(Section\s+\d+[A-Z]*|Rule\s+\d+[A-Z]*|Article\s+\d+[A-Z]*|"
    r"Chapter\s+[IVXLC]+|CHAPTER\s+[IVXLC]+|PART\s+[IVXLC]+|SCHEDULE\s+[IVXLC]*)\b",
    re.I,
)


def embed_batch(port, texts):
    body = json.dumps({"input": texts, "model": "bge"}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/embeddings",
                                 data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    out = [None] * len(texts)
    for item in d["data"]:
        out[item["index"]] = item["embedding"]
    return out


def embed_batch_safe(port, texts):
    """Try batch; on HTTP 400 split in half and retry; on single-item failure drop it."""
    try:
        return embed_batch(port, texts)
    except urllib.error.HTTPError as e:
        if e.code != 400 or len(texts) == 1:
            # single-item 400 -> skip this chunk
            return [None] * len(texts)
        mid = len(texts) // 2
        left = embed_batch_safe(port, texts[:mid])
        right = embed_batch_safe(port, texts[mid:])
        return left + right


def qdrant_upsert(points):
    body = json.dumps({"points": points}).encode()
    req = urllib.request.Request(f"{QDRANT}/collections/{COLL}/points?wait=false",
                                 data=body, method="PUT")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def chunk_text(full_text):
    """Section-aware chunking with hard char cap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]
    out, buf, buf_words, anchor = [], [], 0, ""
    for p in paras:
        m = SECTION_RE.search(p)
        if m and buf_words > 90:
            out.append((anchor, " ".join(buf)))
            tail = " ".join(buf).split()[-OVERLAP:]
            buf = [" ".join(tail)]
            buf_words = len(tail)
            anchor = m.group(1).strip()
        elif m:
            anchor = m.group(1).strip()
        buf.append(p)
        buf_words += len(p.split())
        if buf_words >= WORDS_PER:
            out.append((anchor, " ".join(buf)))
            tail = " ".join(buf).split()[-OVERLAP:]
            buf = [" ".join(tail)]
            buf_words = len(tail)
    if buf_words > 20:
        out.append((anchor, " ".join(buf)))
    return out


YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")


def derive_year(fname):
    m = YEAR_RE.search(fname)
    return int(m.group(0)) if m else None


def pdf_to_chunks(path, category, label_prefix):
    fname = os.path.basename(path)
    doc = fitz.open(path)
    pages = [p.get_text("text") for p in doc]
    doc.close()
    full = "\n\n".join(pages)
    full = re.sub(r"[ \t]+", " ", full)
    full = re.sub(r"\n{3,}", "\n\n", full)
    year = derive_year(fname)
    act_name = re.sub(r"\.pdf$", "", fname, flags=re.I).replace("_", " ")
    base_src = f"{label_prefix} - {act_name[:100]}"
    chunks = chunk_text(full)
    out = []
    seen = set()
    for anchor, text in chunks:
        t = text.strip()
        if not t:
            continue
        if len(t) > CHAR_CAP:
            t = t[:CHAR_CAP]
        h = hashlib.sha1(t[:400].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        src = f"{base_src} [{anchor}]" if anchor else base_src
        out.append({
            "source": src,
            "text": t,
            "dataset": category,
            "act_name": act_name,
            "act_year": year,
            "anchor": anchor,
            "file": fname,
        })
    return out


def point_id_for(source, text):
    h = hashlib.sha1((source + "|" + text[:200]).encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


def ingest_chunks(chunks, dataset, tier, echo=True):
    n = len(chunks)
    if n == 0:
        print(f"[{dataset}] no chunks to ingest", flush=True)
        return 0
    batches = [chunks[i:i + BATCH] for i in range(0, n, BATCH)]
    print(f"[{dataset}] chunks={n} batches={len(batches)}", flush=True)
    t0 = time.time()
    upserted = 0
    dropped = 0

    def embed_one(args):
        idx, batch = args
        port = EMBED_PORTS[idx % len(EMBED_PORTS)]
        texts = [c["text"] for c in batch]
        vecs = embed_batch_safe(port, texts)
        return idx, vecs

    with ThreadPoolExecutor(max_workers=len(EMBED_PORTS)) as ex:
        futs = {ex.submit(embed_one, (i, b)): (i, b) for i, b in enumerate(batches)}
        for f in as_completed(futs):
            i, b = futs[f]
            idx, vecs = f.result()
            pts = []
            for c, v in zip(b, vecs):
                if v is None:
                    dropped += 1
                    continue
                pid = point_id_for(c["source"], c["text"])
                payload = {
                    "source": c["source"],
                    "text": c["text"],
                    "dataset": c["dataset"],
                    "tier": tier,
                    "act_name": c["act_name"],
                    "act_year": c["act_year"],
                    "anchor": c["anchor"],
                    "file": c["file"],
                }
                pts.append({"id": pid, "vector": v, "payload": payload})
            if not pts:
                continue
            for attempt in range(3):
                try:
                    qdrant_upsert(pts)
                    upserted += len(pts)
                    break
                except Exception as e:
                    print(f"  upsert retry {attempt} batch {idx}: {e}", flush=True)
                    time.sleep(1 + attempt)
            if echo and (idx % 10 == 0):
                dt = time.time() - t0
                print(f"  batch {idx+1}/{len(batches)}  upserted={upserted}  dropped={dropped}  "
                      f"rate={upserted/max(dt,1):.1f}/s", flush=True)

    dt = time.time() - t0
    print(f"[{dataset}] DONE upserted={upserted}/{n} dropped={dropped} in {dt:.0f}s "
          f"({upserted/max(dt,1):.1f}/s)", flush=True)
    return upserted


def ingest_dir(root, category, tier):
    d = os.path.join(root, category)
    if not os.path.isdir(d):
        print(f"skip missing dir {d}", flush=True)
        return 0
    pdfs = sorted(glob.glob(os.path.join(d, "*.pdf")))
    if not pdfs:
        print(f"[{category}] no pdfs", flush=True)
        return 0
    label_prefix = category.replace("t1_", "")
    all_chunks = []
    for f in pdfs:
        try:
            ch = pdf_to_chunks(f, category, label_prefix)
            all_chunks.extend(ch)
            print(f"  + {os.path.basename(f)}: {len(ch)} chunks", flush=True)
        except Exception as e:
            print(f"  x {os.path.basename(f)}: {e}", flush=True)
    return ingest_chunks(all_chunks, category, tier)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/opt/indian-legal-ai/gst_stage")
    ap.add_argument("--cat", help="single category dir name, e.g. t1_adr")
    ap.add_argument("--all", action="store_true", help="iterate every t1_* dir with PDFs")
    ap.add_argument("--tier", type=int, default=1)
    args = ap.parse_args()

    if args.all:
        cats = sorted(d for d in os.listdir(args.root)
                      if d.startswith("t1_") and os.path.isdir(os.path.join(args.root, d)))
    elif args.cat:
        cats = [args.cat]
    else:
        ap.error("need --cat or --all")

    t0 = time.time()
    grand = 0
    summary = []
    for c in cats:
        pdfs = glob.glob(os.path.join(args.root, c, "*.pdf"))
        if not pdfs:
            summary.append((c, 0, 0, "no-pdfs"))
            continue
        n = ingest_dir(args.root, c, args.tier)
        grand += n
        summary.append((c, len(pdfs), n, "ok"))

    dt = time.time() - t0
    print("\n" + "=" * 72)
    print(f"T1 INGEST DONE  total_upserted={grand}  elapsed={dt/60:.1f}min")
    print("=" * 72)
    for c, p, n, s in summary:
        print(f"  {c:<36} pdfs={p:>3}  chunks={n:>6}  {s}")


if __name__ == "__main__":
    main()
