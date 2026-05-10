#!/usr/bin/env python3
"""Cluster-wide linked_doc_ids backfill — manifest-driven (bypasses Qdrant scroll
which panics on corrupt segments in `red` collections, observed 2026-05-08 cbic_v2).

Strategy:
  1. Build sha256_en cluster map from scraper manifest (clusters of size > 1).
  2. For every chunk in ingest_manifest_v2 whose doc_id is in a cluster:
       - derive deterministic point_id from (doc_id, page, char_start)
       - target_links = full cluster - self
  3. Group point_ids by their target_links list, then call set_payload
     with the bulk `points=[ids...]` form (one HTTP per cluster member,
     not one per point).

Idempotent. Avoids any scroll over the collection.
"""
import os, sys, json, time, hashlib, sqlite3
from collections import defaultdict
from qdrant_client import QdrantClient

QDRANT_URL = 'http://127.0.0.1:6343'
COLLECTION = 'cbic_v2'
SCRAPER_MANIFEST = '/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite'
INGEST_MANIFEST = '/opt/indian-legal-ai/data/ingest_manifest_v2.sqlite'


def derive_pid(doc_id, page, char_start):
    key = f"{doc_id}|{page or 0}|{char_start or 0}".encode()
    return int(hashlib.sha256(key).hexdigest()[:15], 16) % (10**15)


def main():
    qc = QdrantClient(url=QDRANT_URL, timeout=120)

    # 1. Cluster map from scraper manifest
    m = sqlite3.connect(SCRAPER_MANIFEST)
    sha_to_dids = defaultdict(list)
    for did, sha in m.execute('SELECT doc_id, sha256_en FROM docs WHERE sha256_en IS NOT NULL'):
        sha_to_dids[sha].append(did)
    clusters = {sha: dids for sha, dids in sha_to_dids.items() if len(dids) > 1}
    cluster_doc_ids = set()
    did_to_full_cluster = {}
    for dids in clusters.values():
        full = sorted(set(dids))
        cluster_doc_ids.update(full)
        for d in full:
            did_to_full_cluster[d] = [x for x in full if x != d]
    print(f'clusters: {len(clusters)} clusters, {len(cluster_doc_ids)} doc_ids')

    # 2. Walk ingest manifest for chunks whose doc_id is in a cluster
    c = sqlite3.connect(INGEST_MANIFEST)
    rows = c.execute("""
        SELECT chunk_id, payload_json FROM chunks
        WHERE is_canonical=1 AND upserted=1
    """)
    # group: tuple(target_links) -> list of pids
    grouped = defaultdict(list)
    n_chunks_in_clusters = 0
    for chunk_id, pj in rows:
        try:
            p = json.loads(pj)
        except Exception:
            continue
        did = p.get('doc_id')
        if did not in did_to_full_cluster:
            continue
        target_links = tuple(did_to_full_cluster[did])
        page = p.get('page', 0)
        char_start = p.get('char_start', 0)
        pid = derive_pid(did, page, char_start)
        grouped[target_links].append(pid)
        n_chunks_in_clusters += 1
    print(f'chunks in clusters: {n_chunks_in_clusters}, distinct link-sets: {len(grouped)}')

    # 3. set_payload one call per link-set, bulk over all matching points
    t0 = time.time()
    n_done = 0; n_err = 0
    for i, (links, pids) in enumerate(grouped.items()):
        try:
            qc.set_payload(
                collection_name=COLLECTION,
                payload={'linked_doc_ids': list(links)},
                points=pids,
            )
            n_done += len(pids)
        except Exception as e:
            n_err += len(pids)
            if n_err <= 10:
                print(f'  set_payload fail (links={list(links)[:2]}... pids={len(pids)}): {type(e).__name__}: {str(e)[:200]}', flush=True)
        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f'  link-set {i+1}/{len(grouped)} done={n_done} err={n_err} rate={rate:.0f}/s elapsed={elapsed:.0f}s', flush=True)

    elapsed = time.time() - t0
    print(f'\nfinished: {n_done} chunks linked, {n_err} errors, {elapsed:.0f}s')


if __name__ == '__main__':
    main()
