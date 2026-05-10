# CBIC RAG — status snapshot

## Recovery (previous task) — DONE
- Final coverage: **15,559 / 15,776  (98.62%)**
- 217 docs unrecoverable (every authoritative + mirror source exhausted):
  - 145 forms (obscure CBIC form PDFs, no web presence)
  - 34 notifications (old cx-tarr 2010-2016)
  - 23 rules/regulations
  - 11 orders, 11 instructions, 2 allied_acts, 2 others
- Provenance tagged in manifest:
  - `download_source='cbic_primary'`  → 15,382 (direct from taxinformation.cbic.gov.in)
  - `download_source='cbic_gst_site'` → 177 (via cbic-gst.gov.in master CGST PDF, Hindi fallback)
- New DB columns: `download_source`, `source_url`, `recovered_at`
- QA watchdog running; it will audit recovered files automatically

## cbic RAG (new task) — READY TO DEPLOY

Files written to `D:\_gpu_rig_ai\cbic_rag\`:

| File              | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| `README.md`       | Architecture + example story-with-quotes output |
| `deploy.sh`       | One-shot setup on the rig                       |
| `requirements.txt`| Python deps                                     |
| `chunker.py`      | Page-aware chunking (`pdftotext` + offsets)     |
| `ingest.py`       | Parallel PDF → BGE-M3 → Qdrant across 7 GPUs    |
| `retriever.py`    | Hybrid dense+sparse + optional cross-encoder    |
| `storyformat.py`  | **The key piece** — story-with-quotes builder   |
| `api.py`          | FastAPI (native + OpenAI-compat endpoints)      |

## Blocker right now

SSH to rig is wedged — but **not** the sshd daemon itself. Verbose diagnostic
shows:

```
Authenticated to 192.168.1.107 ([192.168.1.107]:22) using "publickey".
debug1: channel 0: new session [client-session]
debug1: Entering interactive session.
Timeout, server 192.168.1.107 not responding.
```

i.e. SSH **auth succeeds**, session channel opens, then the rig-side process
that would exec the shell/command is hung. Typical causes:

- **pam_systemd / logind stuck** (wait on dbus)
- **home dir on a hung mount** (NFS / SMB timeout) — your models are on a
  Windows SMB share; worth checking
- **fork/exec failing** — fd exhaustion or transient OOM in user slice
- **PAM env script reading a blocking file**

**How to unblock (try in order on the rig via web terminal / console):**

1. Restart logind + ssh:
   ```
   sudo systemctl restart systemd-logind
   sudo systemctl restart ssh
   ```

2. If that doesn't help, check journal:
   ```
   sudo journalctl -u ssh -n 80 --no-pager
   sudo journalctl -u systemd-logind -n 40 --no-pager
   dmesg | tail -30
   ```

3. Check for hung mounts:
   ```
   mount | grep -E 'cifs|nfs|smb'
   df -h   # any that hang?
   ```

4. Nuclear: `sudo reboot` (safe — all services auto-start; qdrant, litellm,
   open-webui, hermes all come back).

Once SSH is back, from Windows PowerShell:
```
scp -r D:\_gpu_rig_ai\cbic_rag rig:/opt/indian-legal-ai/rag/
ssh rig "bash /opt/indian-legal-ai/rag/cbic_rag/deploy.sh"
```

That's it — Qdrant (new instance on :6343), API on :9500, ingest kicks off in a
`screen` session using all 7 GPUs.

## Deploy flow the script executes

1. Creates `/opt/indian-legal-ai/rag/cbic_api/` + venv
2. Installs deps (requirements.txt)
3. Launches **separate Qdrant on :6343** with isolated storage at
   `/opt/indian-legal-ai/rag/qdrant_cbic_storage/` — your existing
   `indian_legal_full` + `indian_legal_t1_v2` collections on :6333 are untouched
4. Starts API on :9500
5. Runs ingestion inside `screen -S cbic-ingest`
   - BGE-M3 on 7 GPUs (HIP_VISIBLE_DEVICES rotated per worker)
   - Dense (1024-d) + sparse vectors per chunk
   - ~15,500 PDFs × ~8 chunks avg = ~120k points
   - Estimated runtime on 7× RX 6700 XT: **45-75 minutes**
6. Prints wiring instructions for Open WebUI + LiteLLM

## How answers will look (the key UX bit)

Every response on Open WebUI will be formatted like:

> **Answer:** Yes, ITC is available on capital goods in the same month of receipt, subject to conditions.
>
> **How we got here:**
>
> Rule 43 of the CGST Rules sets out the apportionment of ITC on capital goods:
> *"The amount of input tax in respect of capital goods used or intended to be used…"* [S1]
>
> Notification 16/2020 refines this by introducing the supplier-filing condition:
> *"credit shall be available only after the supplier has filed GSTR-1…"* [S2]
>
> **Conclusion:** Same-month ITC on capital goods is permitted if the supplier has complied with GSTR-1 filing.
>
> ---
> ### Sources
> - **[S1]** CGST Rules, 2017 — rules p.7 · [original](https://taxinformation.cbic.gov.in/…)
> - **[S2]** Notification 16/2020-CT — notifications p.2 · [original](https://taxinformation.cbic.gov.in/…)

Any quote the LLM can't actually find verbatim in the retrieved chunk gets
a visible ⚠️ marker — no silent hallucination.

## Next morning checklist

- [ ] SSH working? `ssh rig echo ok`
- [ ] Ingest screen running? `screen -r cbic-ingest`
- [ ] Coverage growing? `curl http://127.0.0.1:6343/collections/cbic_v1 | jq .result.points_count`
- [ ] API healthy? `curl http://127.0.0.1:9500/health`
- [ ] Open WebUI: add connection `http://<rig-ip>:9500/v1`, model `cbic-rag`, test a query
