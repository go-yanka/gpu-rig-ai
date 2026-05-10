# RunPod Deploy Plan — BGE-M3 Fine-Tune (CBIC v1)

Click-to-deploy recipe. Follow top-to-bottom the moment
`D:/_gpu_rig_ai/training/curated_pairs.jsonl` +
`D:/_gpu_rig_ai/training/hard_negatives.jsonl` exist.

---

## 0. Prereqs (one-time)

1. Account: https://www.runpod.io/console/signup  (Google login works).
2. Add $10 credit (Billing -> Add Funds). Run will burn ~$1.
3. Generate API key (optional, for `runpodctl`): https://www.runpod.io/console/user/settings
4. Install `runpodctl` on laptop (optional, else use scp):
   https://github.com/runpod/runpodctl/releases
   ```powershell
   choco install runpodctl    # or download the win64 exe and put on PATH
   runpodctl config --apiKey <KEY>
   ```
5. Add your SSH pubkey under **Settings -> SSH Public Keys**
   (laptop `%USERPROFILE%\.ssh\id_ed25519.pub`). Required for `scp` fallback.

---

## 1. Launch pod

Go to **Pods -> Deploy** ->

| Setting | Value |
| --- | --- |
| GPU | **1 x A100 PCIe 40GB** (NOT SXM — SXM is ~1.5x the price) |
| Region | any with stock; prefer US/EU for latency |
| Template | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (Community template, "RunPod Pytorch 2.4.0") |
| Container disk | 20 GB |
| Volume disk | **50 GB** mounted at `/workspace` (persists if you stop+start) |
| Expose | TCP 22 (SSH) — default on |
| Env vars | `HF_HOME=/workspace/hf_cache` <br> `TRANSFORMERS_CACHE=/workspace/hf_cache` <br> `HF_HUB_ENABLE_HF_TRANSFER=1` |
| Pod name | `bge-m3-cbic-ft` |

Click **Deploy On-Demand**. Wait ~60 s for "Running" state.

### Cost sanity check

A100 PCIe 40GB: ~**$1.19/hr** Community Cloud, ~$1.64/hr Secure Cloud
(prices as of 2026-04). Training ~20-30 min => **~$0.50-0.80 per run**.
Add ~10 min setup/download => round to **$1.00 budget**.

If A100 unavailable or >$1.50/hr, fall back to **RTX 4090 24GB** ~$0.44/hr,
expect ~35-45 min => still under $1.

---

## 2. Connect

From the pod's **Connect** button, copy either:
- Web terminal (fastest for sanity check)
- SSH over exposed TCP: `ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519`

Note pod IP + port; you need them for the upload script.

---

## 3. Setup (on pod)

Paste the contents of `setup.sh` into the pod shell, or:
```bash
curl -sSL https://raw.githubusercontent.com/<yours>/setup.sh | bash
# OR scp setup.sh over first, then: bash /workspace/setup.sh
```

---

## 4. Upload training data (from laptop)

```powershell
# from D:\_gpu_rig_ai\training\runpod\
.\upload.ps1 -PodIp <ip> -PodPort <port>
```

---

## 5. Train (on pod)

```bash
cd /workspace/cbic && bash run_training.sh 2>&1 | tee train.log
```

Expected wall-clock: 20-30 min. Watch for:
- `cbic-gold_cosine_Accuracy@5` printed after each epoch — should climb.
- Final `[done] model saved to /workspace/cbic/bge-m3-cbic-v1`.

---

## 6. Download model (from laptop)

```powershell
.\download.ps1 -PodIp <ip> -PodPort <port>
```

Model lands at `D:\_gpu_rig_ai\training\bge-m3-cbic-v1\`.

---

## 7. Stop pod

**Pods -> ... -> Terminate** (NOT "Stop" — Stop keeps billing the volume).
If you want to keep the volume for a v2 run, "Stop" is ~$0.07/GB/month
=> $3.50/mo for 50 GB. Usually not worth it; just re-upload next time.

---

## 8. Deploy to rig

See `deploy_to_rig.md`.
