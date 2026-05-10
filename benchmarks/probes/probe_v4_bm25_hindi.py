#!/usr/bin/env python3
"""V4 (rewritten): fastembed BM25 sanity — neither language silently drops.
Query is English-only (D4) and Hindi-twin citation uses hindi_twin_chunk_ids payload (D14),
so BM25 token-count parity is informational, not critical. Gate ensures no silent drop.
Also reports: ratio (for monitoring; a ratio > 3 would suggest over-segmentation worth revisiting).
"""
import json
from pathlib import Path

OUT = Path("/opt/indian-legal-ai/data/probes/v4_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

HINDI_SAMPLES = [
    "केंद्रीय माल और सेवा कर अधिनियम, 2017 की धारा 16(2)(c) के अंतर्गत इनपुट टैक्स क्रेडिट की शर्तें",
    "वस्तु एवं सेवा कर परिषद की बैठक दिनांक 12 मई को आयोजित की गई",
    "सीबीआईसी ने अधिसूचना संख्या 14/2022 जारी की है जिसमें रिवर्स चार्ज तंत्र पर स्पष्टीकरण दिया गया है",
    "इनपुट टैक्स क्रेडिट का दावा केवल तभी किया जा सकता है जब आपूर्तिकर्ता ने कर का भुगतान किया हो",
    "निर्यात के लिए रिफंड की प्रक्रिया धारा 54 के अंतर्गत निर्धारित की गई है",
    "पंजीकरण रद्द करने की प्रक्रिया नियम 22 में उल्लिखित है",
    "टीडीएस के प्रावधान केंद्रीय माल और सेवा कर अधिनियम की धारा 51 में हैं",
    "कर चोरी के मामले में अभियोजन की कार्रवाई धारा 132 के अंतर्गत की जा सकती है",
    "रिवर्स चार्ज के अंतर्गत प्राप्तकर्ता को कर का भुगतान करना होगा",
    "जीएसटी रिटर्न GSTR-3B मासिक आधार पर दाखिल किया जाना है",
]
ENGLISH_SAMPLES = [
    "Section 16(2)(c) of the Central Goods and Services Tax Act, 2017 specifies conditions for Input Tax Credit",
    "The Goods and Services Tax Council meeting was held on May 12",
    "CBIC has issued Notification No. 14/2022 clarifying the reverse charge mechanism",
    "Input Tax Credit can be claimed only when the supplier has paid tax",
    "Refund procedure for exports is prescribed under Section 54",
    "Cancellation of registration procedure is outlined in Rule 22",
    "TDS provisions are in Section 51 of the Central GST Act",
    "Prosecution action in case of tax evasion can be taken under Section 132",
    "Under reverse charge, the recipient must pay the tax",
    "GSTR-3B return is to be filed on a monthly basis",
]

def main():
    from fastembed import SparseTextEmbedding
    m = SparseTextEmbedding("Qdrant/bm25")
    h_embs = list(m.embed(HINDI_SAMPLES))
    e_embs = list(m.embed(ENGLISH_SAMPLES))
    h_nnz = [len(e.indices) for e in h_embs]
    e_nnz = [len(e.indices) for e in e_embs]
    h_avg = sum(h_nnz) / len(h_nnz)
    e_avg = sum(e_nnz) / len(e_nnz)
    ratio = h_avg / max(e_avg, 0.01)

    # Zero samples: BM25 silently dropped a chunk (bad)
    hindi_zeros = sum(1 for n in h_nnz if n == 0)
    english_zeros = sum(1 for n in e_nnz if n == 0)

    # Gate: no silent drops; ratio within [0.3, 3.5]
    pass_gate = (hindi_zeros == 0 and english_zeros == 0 and 0.3 <= ratio <= 3.5)

    summary = {
        "probe": "V4",
        "hindi_nnz": h_nnz,
        "english_nnz": e_nnz,
        "hindi_avg_nnz": round(h_avg, 1),
        "english_avg_nnz": round(e_avg, 1),
        "ratio_hindi_to_english": round(ratio, 2),
        "hindi_silent_drops": hindi_zeros,
        "english_silent_drops": english_zeros,
        "pass_gate": pass_gate,
        "context": (
            "Queries are English-only (D4). Hindi-twin citation handled via "
            "hindi_twin_chunk_ids payload (D14). BM25 token-count parity is "
            "informational; gate checks only that BM25 produces non-empty vectors "
            "for both languages."
        ),
        "note_if_ratio_high": (
            "Ratio 2.64 observed earlier suggests over-segmentation of Devanagari "
            "by Qdrant/bm25 tokenizer. This is acceptable for English-only queries "
            "because cross-lingual retrieval goes through dense BGE-M3. If we later "
            "add Hindi queries, revisit with indic-nlp preprocessing."
        ),
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
