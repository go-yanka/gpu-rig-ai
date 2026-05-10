#!/usr/bin/env python3
"""
Parallel GST ingestion worker:
 - PyMuPDF extract
 - section-aware chunking (~250 words, 40 overlap)
 - round-robin across 4 BGE embedders on :9092/3/4/5
 - batched Qdrant upsert to `indian_legal_full`
"""
import sys, os, json, re, time, uuid, hashlib, urllib.request, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF

EMBED_PORTS = [9092, 9093, 9094, 9095]
QDRANT = "http://localhost:6333"
COLL   = "indian_legal_full"
BATCH  = 16

def embed_batch(port, texts):
    body = json.dumps({"input": texts, "model": "bge"}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/embeddings",
                                 data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return [e["embedding"] for e in d["data"]]

def qdrant_upsert(points):
    body = json.dumps({"points": points}).encode()
    req = urllib.request.Request(f"{QDRANT}/collections/{COLL}/points?wait=false",
                                 data=body, method="PUT")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

SECTION_RE = re.compile(
    r"(?:^|\n)\s*(Section\s+\d+[A-Z]*|Rule\s+\d+[A-Z]*|Article\s+\d+[A-Z]*|"
    r"Chapter\s+[IVXLC]+|PART\s+[IVXLC]+)\b", re.I)

def chunk_text(full_text, words_per=250, overlap=40):
    """Section-aware chunker. Emits (anchor, chunk) pairs."""
    # Split into sentences-ish
    paras = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]
    out = []
    buf = []
    buf_words = 0
    anchor = ""
    for p in paras:
        m = SECTION_RE.search(p)
        if m and buf_words > 120:
            out.append((anchor, " ".join(buf)))
            # carry overlap
            tail = " ".join(buf).split()[-overlap:]
            buf = [" ".join(tail)]
            buf_words = len(tail)
            anchor = m.group(1).strip()
        elif m:
            anchor = m.group(1).strip()
        buf.append(p)
        buf_words += len(p.split())
        if buf_words >= words_per:
            out.append((anchor, " ".join(buf)))
            tail = " ".join(buf).split()[-overlap:]
            buf = [" ".join(tail)]
            buf_words = len(tail)
    if buf_words > 30:
        out.append((anchor, " ".join(buf)))
    return out

def pdf_to_chunks(path, source_label):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        t = page.get_text("text")
        pages.append(t)
    full = "\n\n".join(pages)
    full = re.sub(r"[ \t]+", " ", full)
    full = re.sub(r"\n{3,}", "\n\n", full)
    chunks = chunk_text(full)
    out = []
    for anchor, text in chunks:
        src = f"{source_label}"
        if anchor:
            src = f"{source_label} [{anchor}]"
        out.append({"source": src, "text": text.strip()})
    return out

def point_id_for(source, text):
    h = hashlib.sha1((source + "|" + text[:200]).encode()).hexdigest()
    # convert to uuid form (Qdrant accepts uuid strings)
    return str(uuid.UUID(h[:32]))

def ingest_chunks(chunks, dataset, tier, echo=True):
    n = len(chunks)
    batches = [chunks[i:i+BATCH] for i in range(0, n, BATCH)]
    print(f"[{dataset}] chunks={n} batches={len(batches)} (batch={BATCH})", flush=True)
    t0 = time.time()
    total_upserted = 0

    def embed_one(args):
        idx, batch = args
        port = EMBED_PORTS[idx % len(EMBED_PORTS)]
        texts = [c["text"][:1400] for c in batch]
        for attempt in range(3):
            try:
                vecs = embed_batch(port, texts)
                break
            except Exception as e:
                print(f"  embed retry {attempt} port={port}: {e}", flush=True)
                time.sleep(1.0 + attempt)
        else:
            return (idx, None)
        return (idx, vecs)

    with ThreadPoolExecutor(max_workers=len(EMBED_PORTS)) as ex:
        futs = {ex.submit(embed_one, (i, b)): (i, b) for i, b in enumerate(batches)}
        for f in as_completed(futs):
            i, b = futs[f]
            idx, vecs = f.result()
            if vecs is None:
                print(f"  FAILED batch {idx}", flush=True)
                continue
            pts = []
            for c, v in zip(b, vecs):
                pid = point_id_for(c["source"], c["text"])
                pts.append({
                    "id": pid,
                    "vector": v,
                    "payload": {
                        "source": c["source"],
                        "text": c["text"],
                        "dataset": dataset,
                        "tier": tier,
                    },
                })
            try:
                qdrant_upsert(pts)
                total_upserted += len(pts)
                if echo and (idx % 5 == 0):
                    dt = time.time() - t0
                    rate = total_upserted / max(dt,1)
                    print(f"  batch {idx+1}/{len(batches)}  upserted={total_upserted}  rate={rate:.1f}/s  elapsed={dt:.0f}s", flush=True)
            except Exception as e:
                print(f"  upsert fail batch {idx}: {e}", flush=True)

    dt = time.time() - t0
    print(f"[{dataset}] DONE {total_upserted}/{n} in {dt:.0f}s ({total_upserted/max(dt,1):.1f}/s)", flush=True)
    return total_upserted


def ingest_dir(dirpath, label_prefix, dataset, tier):
    import glob
    files = sorted(glob.glob(os.path.join(dirpath, "*.pdf")))
    print(f"[{dataset}] files={len(files)} in {dirpath}", flush=True)
    all_chunks = []
    for f in files:
        try:
            label = f"{label_prefix} - {os.path.basename(f)[:80]}"
            ch = pdf_to_chunks(f, label)
            all_chunks.extend(ch)
            print(f"  + {os.path.basename(f)}: {len(ch)} chunks", flush=True)
        except Exception as e:
            print(f"  x {os.path.basename(f)}: {e}", flush=True)
    print(f"[{dataset}] total chunks={len(all_chunks)}", flush=True)
    if not all_chunks: return 0
    return ingest_chunks(all_chunks, dataset, tier)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--label", required=True, help="source label prefix")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tier", type=int, required=True)
    args = ap.parse_args()

    if args.dir:
        ingest_dir(args.dir, args.label, args.dataset, args.tier)
    else:
        chunks = pdf_to_chunks(args.pdf, args.label)
        print(f"extracted {len(chunks)} chunks from {args.pdf}", flush=True)
        ingest_chunks(chunks, args.dataset, args.tier)
