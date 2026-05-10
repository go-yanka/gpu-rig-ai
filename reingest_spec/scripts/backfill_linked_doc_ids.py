#!/usr/bin/env python3
"""Backfill cluster-wide linked_doc_ids on every cbic_v2 chunk.

Why (codified 2026-05-08): the original chunker only populates linked_doc_ids
between doc_ids whose chunks the chunker actually processed. For shared-PDF
clusters where build_batch never picked all members (or some members were
deferred/dropped), the cluster relationship isn't fully recorded in the
chunk payload. This means G1 evaluator's linked_doc_ids fallback (Defect D fix)
misses cluster members it should have caught.

This script: for each sha256_en cluster of size > 1 in the scraper manifest,
update every chunk in cbic_v2 whose doc_id is in the cluster, setting
linked_doc_ids to the FULL cluster membership minus self. Recovers
"D-1 cluster member" hits in G1 without re-ingest or chunker-v3.

Idempotent — running again produces same result.
"""
import os, sys, sqlite3, json, time
from collections import defaultdict
from qdrant_client import QdrantClient

QDRANT_URL = 'http://127.0.0.1:6343'
COLLECTION = 'cbic_v2'
MANIFEST = '/opt/indian-legal-ai/data/scraped/cbic/_manifest.sqlite'

def main():
    qc = QdrantClient(url=QDRANT_URL, timeout=120)
    m = sqlite3.connect(MANIFEST)

    # Build sha -> [doc_ids] map for clusters of size > 1
    sha_to_dids = defaultdict(list)
    for did, sha in m.execute('SELECT doc_id, sha256_en FROM docs WHERE sha256_en IS NOT NULL'):
        sha_to_dids[sha].append(did)
    clusters = {sha: dids for sha, dids in sha_to_dids.items() if len(dids) > 1}
    cluster_doc_ids = set()
    for dids in clusters.values():
        cluster_doc_ids.update(dids)
    print(f'D-1 shared-PDF clusters: {len(clusters)} clusters, {len(cluster_doc_ids)} unique doc_ids in clusters')

    # For each cluster, build the full member set
    did_to_full_cluster = {}
    for dids in clusters.values():
        full = sorted(set(dids))
        for d in full:
            others = [x for x in full if x != d]
            did_to_full_cluster[d] = others
    print(f'doc_ids with cluster siblings: {len(did_to_full_cluster)}')

    # Now scroll cbic_v2 for chunks belonging to any cluster doc_id
    t0 = time.time()
    n_updated = 0
    n_scanned = 0
    next_offset = None
    BATCH = 256

    while True:
        scroll = qc.scroll(collection_name=COLLECTION, limit=BATCH,
                           offset=next_offset,
                           with_payload=['doc_id','linked_doc_ids'], with_vectors=False)
        points, next_offset = scroll
        if not points: break
        n_scanned += len(points)

        # Determine which points need update
        upd_payloads = []
        upd_ids = []
        for p in points:
            did = (p.payload or {}).get('doc_id')
            if did in did_to_full_cluster:
                target_links = did_to_full_cluster[did]
                current = (p.payload or {}).get('linked_doc_ids') or []
                if set(current) != set(target_links):
                    upd_ids.append(p.id)
                    upd_payloads.append({'linked_doc_ids': target_links})
        if upd_ids:
            from qdrant_client.http import models as qm
            for pid, pl in zip(upd_ids, upd_payloads):
                qc.set_payload(collection_name=COLLECTION, payload=pl,
                               points=[pid])
            n_updated += len(upd_ids)
        if n_scanned % 5120 == 0:
            elapsed = time.time() - t0
            print(f'  scanned={n_scanned} updated={n_updated} elapsed={elapsed:.0f}s', flush=True)
        if next_offset is None: break

    elapsed = time.time() - t0
    print(f'\nfinished: scanned={n_scanned}, updated linked_doc_ids on {n_updated} chunks in {elapsed:.0f}s')

if __name__ == '__main__':
    main()
