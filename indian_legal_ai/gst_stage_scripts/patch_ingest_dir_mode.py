#!/usr/bin/env python3
"""Add dir-mode to /opt/indian-legal-ai/scripts/parallel_ingest.py"""
P = "/opt/indian-legal-ai/scripts/parallel_ingest.py"
s = open(P).read()

if "def ingest_dir" in s:
    print("already patched")
    raise SystemExit

add = r'''
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

'''

s = s.replace('if __name__ == "__main__":', add + 'if __name__ == "__main__":')
s = s.replace(
    'ap.add_argument("--pdf", required=True)',
    'ap.add_argument("--pdf", default=None)\n    ap.add_argument("--dir", default=None)'
)
s = s.replace(
    'chunks = pdf_to_chunks(args.pdf, args.label)\n    print(f"extracted {len(chunks)} chunks from {args.pdf}", flush=True)\n    ingest_chunks(chunks, args.dataset, args.tier)',
    'if args.dir:\n'
    '        ingest_dir(args.dir, args.label, args.dataset, args.tier)\n'
    '    else:\n'
    '        chunks = pdf_to_chunks(args.pdf, args.label)\n'
    '        print(f"extracted {len(chunks)} chunks from {args.pdf}", flush=True)\n'
    '        ingest_chunks(chunks, args.dataset, args.tier)'
)
open(P, "w").write(s)
print("patched, size:", len(s))
