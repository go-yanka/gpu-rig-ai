# Consult: ROCm on RX 5700 XT (gfx1010) — Alternatives If Not Viable

**Date:** 2026-04-22
**Audience:** External LLMs (Gemini 2.5 Pro, GPT-5, etc.)
**Ask:** Give sharp critique. If ROCm on gfx1010 is truly dead, what alternative stacks can achieve the same capabilities on these specific GPUs? Be concrete with version numbers, repos, and known-working paths.

---

## 1. The Hardware

- **5× AMD Radeon RX 5700 XT** — gfx1010, RDNA1, 8 GB VRAM each
- **1× AMD Radeon RX 6700 XT** — gfx1031, RDNA2, 12 GB VRAM (GPU 2, runs production chat, cannot disturb)
- Host: Ubuntu 22.04 (kernel 5.15), Python 3.10, 64 GB RAM, 4-core CPU
- All 6 GPUs sit on the same rig; we currently use Vulkan (llama.cpp / llama-server) successfully on all of them for inference and for BGE-M3 embedding (measured 5.2–9.6 chunks/s).

## 2. Why We Want ROCm (What Capabilities It Unlocks)

Vulkan gets us inference and embedding. It does NOT get us the following, which we need for a CBIC RAG re-ingestion targeting 95% retrieval quality:

1. **PyTorch-ROCm on the 5× 5700 XT for fine-tuning**
   - Train a **query-side LoRA adapter** on BGE-M3 (keeps the 150K document embeddings frozen — 30 min iteration instead of 4.3 hr re-embed).
   - Train with **MarginMSE loss** using the cross-encoder as a teacher — proven 3–7% recall lift over vanilla MNRL.
   - Without ROCm we cannot run `sentence-transformers` training at all on these GPUs.

2. **Cross-encoder reranker at batch size >1**
   - `bge-reranker-v2-m3` via `transformers` / `FlashRank`. Vulkan llama.cpp cannot run cross-encoder reranking efficiently (no BERT-family batch path).
   - Needed for both teacher-scoring during training AND for live query-time reranking.

3. **GPU-accelerated document parsers**
   - **Marker / Surya** (layout + OCR + table extraction) — these are PyTorch-native. ~471 image-only PDFs in the corpus need this.
   - CPU fallback (RapidOCR / tesseract) is 20–50× slower and user has explicitly rejected tesseract for this workload.

4. **Synthetic Q&A generator at full throughput**
   - Local qwen3-8B via `vllm` (ROCm backend) would 5–10× the throughput we get from llama-server Vulkan for the 10–20K training pair generation.

5. **Any HuggingFace model that isn't GGUF-convertible.**

**Net:** ROCm on the 5× 5700 XT rigs unlocks training + reranking + GPU OCR. Without it we can do inference-only workloads, which blocks roughly half of the planned re-ingestion pipeline.

GPU 2 (RX 6700 XT, gfx1031) appears to work with ROCm via `HSA_OVERRIDE_GFX_VERSION=10.3.0`, but it's reserved for production chat and has only 12 GB.

## 3. The Error / Current State

**ROCm 6.1.0 is installed on the host.** `rocminfo` reports all 6 GPUs including the 5 RDNA1 cards.

**PyTorch-ROCm refuses to work on gfx1010:**

- Official PyTorch ROCm wheels (2.6.0+rocm6.1, 2.5.x+rocm6.x) are compiled for gfx906/gfx908/gfx90a/gfx940/gfx942/gfx1030/gfx1100/gfx1101/gfx1102. **gfx1010 is not in the list.**
- Setting `HSA_OVERRIDE_GFX_VERSION=10.3.0` (to masquerade as gfx1030) lets `torch.cuda.is_available()` return True but kernel launches segfault or produce silent wrong results on RDNA1 (documented in AMD issue tracker and multiple community reports).
- AMD officially dropped gfx1010 support after ROCm 5.7. All 6.x releases target gfx1030+.

**Secondary install pain** (not the blocker, but symptomatic):
- `pip install torch --index-url https://download.pytorch.org/whl/rocm6.1` repeatedly fails with sha256 hash mismatches on the 2.7 GB torch wheel (pip issue #12177 — network-level truncation on large wheels).
- wget downloads the wheel fine, but then `pip install <local-wheel>` without `--no-deps` silently falls back to PyPI torch-2.11.0 (CUDA) because `pytorch-triton-rocm` isn't on PyPI.
- Workable install incantation exists (`uv pip install torch --torch-backend=rocm6` or `pip install --no-deps <local-wheel>` + manual deps), but even once installed, gfx1010 kernels are the blocker.

## 4. The Question for You

**Assume gfx1010 is dead for PyTorch-ROCm in 2026.** No amount of patching the torch source will make RDNA1 kernels reliable — the upstream composable_kernel / rocBLAS tuning targets don't include it.

Given that, what alternatives exist on these specific 5× RX 5700 XT cards to achieve:

**A. LoRA / MarginMSE training of BGE-M3 (bi-encoder, ~560M params)**
- Can we train via Vulkan backends? (IREE? SHARK? MLIR-based?)
- Does **ZLUDA** (CUDA translation layer) work on RDNA1 for PyTorch? What's the actual state in 2026?
- Is there a working **Vulkan compute backend for PyTorch** (beyond the experimental mobile one)?
- Would **JAX with Vulkan/SPIR-V** work? JAX-metal-style but for Vulkan?
- Practical option: do the training **on GPU 2 (gfx1031) only**, eat the 12 GB limit, use gradient accumulation — is BGE-M3 LoRA feasible on a single 12 GB card? (Back-of-envelope: BGE-M3 560M in fp16 = ~1.1 GB weights + LoRA rank-16 ~10 MB + optimizer states ~4 GB + activations for seq-len 512 batch 8 — estimate please.)

**B. Cross-encoder reranker inference (`bge-reranker-v2-m3`, ~568M)**
- Does **llama.cpp Vulkan** now support BERT-family cross-encoders? (It supports BGE-M3 embedding — is reranker next?)
- Is there an ONNX-Runtime path with DirectML-on-Linux or Vulkan EP that works on RDNA1?
- **FlashRank** — does it have a non-PyTorch backend?

**C. Marker / Surya OCR for image-only PDFs**
- These are heavily PyTorch-coupled. ONNX export path for Surya / Marker models on Vulkan ORT?
- Alternative: **docTR** (TensorFlow + ONNX) — does it work on Vulkan ORT? Quality comparable?
- Practical fallback: CPU **PaddleOCR** / **RapidOCR** — realistic throughput on 4-core CPU for 471 image PDFs?

**D. vllm-style high-throughput generation for synthetic Q&A**
- `vllm` requires CUDA or ROCm (gfx1030+). Alternatives?
- **llama.cpp Vulkan with parallel requests** (`-np 8`) — realistic tokens/sec on 5700 XT running Qwen3-8B Q4_K_M? Is this "good enough" vs the 5–10× vllm would give us?
- **TGI with exllamav2 Vulkan** — does this exist?

**E. Community ROCm forks**
- **TheRock** (AMD's next-gen build system) — does it build usable gfx1010 binaries in 2026?
- **V6ser/TheRock-gfx1031** and similar community forks — any gfx1010 analog that actually works?
- Rebuilding ROCm 5.7 from source and pinning PyTorch 2.1 — is the quality/compat gap vs 2026 tooling too painful (flash-attn, modern sentence-transformers, etc.)?

## 5. What "Good Enough" Looks Like

- We do NOT need training to be fast. A 12-hour LoRA run on CPU would be acceptable if it works. But most CPU-only trainers are memory-bound, not compute-bound, and swap-thrash on 64 GB RAM with 560M-param models.
- We DO need reranker inference to be < 500 ms for 50 candidates × 512 tokens at query time. Vulkan llama.cpp would need ~20 tok/s throughput for BERT-class — is that realistic?
- OCR throughput floor: 471 PDFs in <8 hours. That's ~60 s/PDF average. Even CPU RapidOCR may hit this.

## 6. Concrete Deliverable We Want From You

For each capability (A through E), tell us:
1. **Is there a working path on RX 5700 XT in 2026?** (Yes / No / "Yes but painful")
2. **Specific repo + version + install command** if yes.
3. **Expected performance** vs the ROCm-ideal path (rough order of magnitude).
4. **Known failure modes** to watch for.
5. **Your overall recommendation:** given all 5 capabilities, should we (a) buy/swap a 6700 XT or 7800 XT to replace a 5700 XT, (b) rent cloud GPU time (RunPod MI300 or A100) for training + reranking only and keep the rig Vulkan-only, (c) source-build some community ROCm fork, or (d) redesign the plan to not need these capabilities at all?

Budget for option (a): ~₹30–40k / ~$400 for one GPU swap. Budget for option (b): ~$50–100 for a one-time training run. Budget for option (c): engineering time, which we have if the success probability is >50%.

Be specific. We've already been burned by plausible-sounding paths that don't actually work on RDNA1.
