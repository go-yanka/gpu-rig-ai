# CBIC RAG — quote-grounded answers

New RAG database for CBIC tax corpus (15,559 PDFs), **independent** of existing
`indian_legal_full` / `indian_legal_t1_v2` Qdrant collections.

## Key differentiator: story-with-quotes answers

Every answer on the frontend reads like a small legal brief:

> **Answer:** Yes, a company can claim ITC on capital goods in the same month of receipt.
>
> **How we got here:**
> Rule 43 of the CGST Rules lays out the apportionment rules; it says
> *"…the amount of input tax in respect of capital goods used or intended to be used…"*
> — [CGST Rules, Part B, Rule 43, p.7].
>
> The conditions are tightened by Notification 16/2020 (CGST), which clarifies
> *"…credit shall be available only after the supplier has filed GSTR-1…"*
> — [Notification 16/2020-CT, p.2].
>
> Combined, the rule permits same-month ITC subject to supplier compliance.

Not just "sources on the right" — the narrative **is** the sourcing.

## Architecture

```
  15,559 PDFs  (/opt/indian-legal-ai/data/scraped/cbic/**/*.pdf)
        │
        ▼
  ingest.py  ─ parallel pdftotext + page-aware chunker
        │
        ▼
  embedder.py ─ BGE-M3 across 7 GPUs (HIP, batched)
        │
        ▼
  Qdrant :6343  (new instance, storage: /opt/indian-legal-ai/rag/qdrant_cbic_storage/)
  collection: cbic_v1
        │
        ▼
  api.py  ─ FastAPI, OpenAI-compatible /v1/chat/completions
        │   (retrieve → rerank → story-format → generate via LiteLLM :4444)
        ▼
  Open WebUI  (existing :3010) hooked as extra model "cbic-rag"
```

## Deploy

```bash
bash deploy.sh
```

Single script: installs deps, launches Qdrant on new port, ingests, starts API, registers with LiteLLM.

## Ports

| Service         | Port | Storage / Config                                       |
| --------------- | ---- | ------------------------------------------------------ |
| Qdrant (cbic)   | 6343 | `/opt/indian-legal-ai/rag/qdrant_cbic_storage/`        |
| cbic-rag API    | 9500 | `/opt/indian-legal-ai/rag/cbic_api/`                   |
| LiteLLM gateway | 4444 | (existing; we only register a new virtual model)       |
| Open WebUI      | 3010 | (existing; we add "cbic-rag" as a connection)          |

## Provenance

Every retrieved chunk carries:
`doc_id`, `title`, `category`, `subcategory`, `file_path`, `page`, `char_start`, `char_end`,
`download_source`, `source_url` — so the UI can link you to the **exact page and paragraph**
the quote came from, with attribution to where the PDF was sourced (CBIC primary vs mirror).
