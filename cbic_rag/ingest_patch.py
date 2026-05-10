"""One-shot patcher: rewrite ingest.main() to use a streaming producer+consumer
so chunk workers never block on main, and main never blocks on submit."""
import re, pathlib, sys

p = pathlib.Path("/opt/indian-legal-ai/rag/cbic_rag/ingest.py")
s = p.read_text()

# Locate main() and replace its body from the "with ProcessPoolExecutor..." block
# through the final elapsed print. We keep everything before (arg parsing, warmup).
old = re.search(
    r"    with ProcessPoolExecutor\(max_workers=args\.workers\).*?elapsed=\{elapsed:\.1f\}s'.*?flush=True\)",
    s, re.DOTALL)
if not old:
    print("ERROR: main() streaming block not found", file=sys.stderr); sys.exit(1)

new_block = r"""    # Streaming producer/consumer:
    # - ProcessPool runs chunk_worker with a rolling window of `args.workers * 4`
    #   submissions in flight at any time (not all 15k at once).
    # - Chunks land in a bounded Queue consumed by a dedicated embed+upsert thread,
    #   so the main loop never blocks on embed latency.
    import queue as _queue, threading as _threading
    chunk_q = _queue.Queue(maxsize=8)  # each item is a list of chunk dicts

    stop_sentinel = object()
    consumer_stats = {"total": 0, "batches": 0, "errors": 0}

    def _consumer():
        buf_local = []
        while True:
            item = chunk_q.get()
            if item is stop_sentinel:
                # final drain
                while buf_local:
                    batch = buf_local[:BATCH]; buf_local = buf_local[BATCH:]
                    texts = [b['text'] for b in batch]
                    try:
                        dense, sparse = embed_batch(texts)
                        n = upsert_chunks(qc, batch, dense, sparse)
                        consumer_stats["total"] += n
                        consumer_stats["batches"] += 1
                    except Exception as e:
                        consumer_stats["errors"] += 1
                        print(f'[embed/upsert err] {e}', flush=True)
                return
            buf_local.extend(item)
            while len(buf_local) >= BATCH:
                batch = buf_local[:BATCH]; buf_local = buf_local[BATCH:]
                texts = [b['text'] for b in batch]
                try:
                    dense, sparse = embed_batch(texts)
                    n = upsert_chunks(qc, batch, dense, sparse)
                    consumer_stats["total"] += n
                    consumer_stats["batches"] += 1
                except Exception as e:
                    consumer_stats["errors"] += 1
                    print(f'[embed/upsert err] {e}', flush=True)

    cthr = _threading.Thread(target=_consumer, daemon=True)
    cthr.start()

    WINDOW = max(args.workers * 4, 64)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        inflight = {}
        doc_iter = iter(docs)
        # prime window
        for _ in range(WINDOW):
            try:
                d = next(doc_iter)
            except StopIteration:
                break
            inflight[ex.submit(chunk_worker, d)] = d
        while inflight:
            done = None
            for f in list(inflight):
                if f.done():
                    done = f; break
            if done is None:
                # block on the oldest
                done = next(iter(as_completed(list(inflight.keys()))))
            del inflight[done]
            done_docs += 1
            try:
                doc_id, chunks = done.result()
                if chunks:
                    chunk_q.put(chunks)
            except Exception as _e:
                pass
            # top up window
            try:
                d = next(doc_iter)
                inflight[ex.submit(chunk_worker, d)] = d
            except StopIteration:
                pass
            if done_docs % 50 == 0:
                elapsed = time.time() - t0
                rate = done_docs / max(1, elapsed)
                qsize = chunk_q.qsize()
                print(f'[{done_docs}/{len(docs)}] upserted={consumer_stats["total"]} '
                      f'rate={rate:.2f} docs/s  q={qsize}  inflight={len(inflight)}',
                      flush=True)
    chunk_q.put(stop_sentinel)
    cthr.join()

    elapsed = time.time() - t0
    print(f'[DONE] docs={len(docs)} chunks={consumer_stats["total"]} elapsed={elapsed:.1f}s'
          f'  err_batches={consumer_stats["errors"]}', flush=True)"""

s2 = s[:old.start()] + new_block + s[old.end():]
p.write_text(s2)
print("patched main() with streaming pipeline; size:", len(s2))

# syntax check
import ast
ast.parse(s2)
print("syntax ok")
