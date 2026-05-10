
# ──────────────────────────────────────────────────────────────────────
# Embed pool admin (added 2026-04-25 — per-GPU named workers + swap)
# ──────────────────────────────────────────────────────────────────────

@app.get('/admin/embed_pool')
def admin_embed_pool():
    """Live per-GPU pool stats: state, weight, calls, errors, p50/p95 latency.
    Use this during gates/ingestion to verify all 6 GPUs are sharing load."""
    try:
        from embedder_direct import get_pool
        return get_pool().health()
    except Exception as e:
        raise HTTPException(500, f'embed_pool unavailable: {e}')


@app.post('/admin/embed_pool/add_gpu/{gpu_id}')
def admin_embed_pool_add(gpu_id: int):
    """Hot-add a GPU to the embed pool (e.g. GPU 2 swap during phase 3-4-5).
    Caller is responsible for ensuring GPU 2 has been unloaded from qwen3 first."""
    try:
        from embedder_direct import get_pool
        return get_pool().add_gpu(gpu_id)
    except Exception as e:
        raise HTTPException(500, f'add_gpu failed: {e}')


@app.post('/admin/embed_pool/remove_gpu/{gpu_id}')
def admin_embed_pool_remove(gpu_id: int):
    """Hot-remove a GPU from the embed pool (e.g. before reloading qwen3 on GPU 2)."""
    try:
        from embedder_direct import get_pool
        return get_pool().remove_gpu(gpu_id)
    except Exception as e:
        raise HTTPException(500, f'remove_gpu failed: {e}')

