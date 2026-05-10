#!/usr/bin/env python3
"""SHA256-based chunk deduplication for v2 ingest.

Contract:
- Input: stream of chunk dicts (from chunker v2)
- Maintain hash -> canonical_chunk_id map
- On collision:
  * Don't create a new indexed point
  * Append (source_doc_id, source_chunk_ordinal) to canonical's `also_appears_in`
  * Return canonical_chunk_id so the caller can link
- Output: (canonical_chunk, is_new) per input chunk

Normalization for hash:
- Unicode NFKC
- Strip leading/trailing whitespace
- Collapse internal whitespace to single space
- Lowercase is OFF (legal text case matters — "Section 16" vs "section 16" are fine to dedupe? YES — gold queries mix cases. So lowercase ON for hash only.)
"""
import hashlib
import re
import unicodedata


WS = re.compile(r"\s+")


def canonical_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "")
    t = WS.sub(" ", t).strip()
    return t.lower()


def text_hash(t: str) -> str:
    return hashlib.sha256(canonical_text(t).encode("utf-8")).hexdigest()


class ChunkDeduper:
    """Call .add(chunk) for each chunk. Returns (canonical_chunk_id, is_new)."""

    def __init__(self):
        self.canon = {}   # hash -> canonical chunk dict
        self.stats = {"total": 0, "unique": 0, "duplicates": 0}

    def add(self, chunk: dict):
        """chunk must have: chunk_id, doc_id, text. May have other fields.
        Returns (canonical_chunk, is_new).
        """
        self.stats["total"] += 1
        h = text_hash(chunk.get("text", ""))
        chunk["_text_hash"] = h
        if h in self.canon:
            canon = self.canon[h]
            # Attach source link
            canon.setdefault("also_appears_in", []).append({
                "doc_id": chunk.get("doc_id"),
                "source_chunk_id": chunk.get("chunk_id"),
            })
            self.stats["duplicates"] += 1
            return canon, False
        # First occurrence → canonical
        chunk.setdefault("also_appears_in", [])
        self.canon[h] = chunk
        self.stats["unique"] += 1
        return chunk, True

    def canonical_chunks(self):
        return list(self.canon.values())


if __name__ == "__main__":
    # self-test
    d = ChunkDeduper()
    a = {"chunk_id": "a1", "doc_id": "docA", "text": "This is a legal provision."}
    b = {"chunk_id": "b1", "doc_id": "docB", "text": "This is a   legal provision.  "}  # same text, diff whitespace
    c = {"chunk_id": "c1", "doc_id": "docC", "text": "DIFFERENT."}
    for ch in (a, b, c): d.add(ch)
    assert d.stats["total"] == 3
    assert d.stats["unique"] == 2
    assert d.stats["duplicates"] == 1
    canon = d.canonical_chunks()
    assert len(canon) == 2
    assert len(canon[0]["also_appears_in"]) == 1
    assert canon[0]["also_appears_in"][0]["doc_id"] == "docB"
    print("SELF-TEST OK:", d.stats)
