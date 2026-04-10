# AI Rig — Instructions for Ritu's Job Agent

## What's Available

A 7-GPU AI inference rig running Gemma 4 E4B on all GPUs, behind a single load-balanced endpoint.

| Detail | Value |
|--------|-------|
| **API Endpoint** | `http://192.168.1.107:4000/v1/chat/completions` |
| **Model** | Gemma 4 E4B (7.5B params, Q4_K_M, thinking mode) |
| **GPUs** | 7x AMD Radeon (6x RX 5700 XT 8GB + 1x RX 6700 XT 12GB) |
| **Speed** | ~63 tokens/sec per GPU |
| **Context** | 16,384 tokens per instance |
| **Parallel capacity** | 7 concurrent requests (1 per GPU) |
| **Load balancing** | nginx least_conn on port 4000 → 7 GPU instances |
| **Tool calling** | 100% (8/8 tests passed, use OpenAI tools format) |
| **JSON extraction** | Excellent (tested with job posting extraction) |
| **Thinking mode** | Enabled (model reasons before answering) |
| **API format** | OpenAI-compatible |
| **Auth** | None required (API key: `sk-none`) |

## Single Endpoint — One URL For Everything

```
http://192.168.1.107:4000/v1/chat/completions
```

This is the ONLY URL you need. nginx load balances across all 7 GPUs automatically. Send as many concurrent requests as you want — nginx routes each to the least-busy GPU.

## How To Use

### Basic Request

```python
import httpx

response = httpx.post(
    "http://192.168.1.107:4000/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "Extract job fields as JSON."},
            {"role": "user", "content": "Senior ML Engineer at Google, Mountain View..."}
        ],
        "max_tokens": 4000
    },
    timeout=120.0
)
result = response.json()
content = result["choices"][0]["message"]["content"]
```

### With OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.107:4000/v1",
    api_key="sk-none"
)

response = client.chat.completions.create(
    model="gemma-4-E4B-it-Q4_K_M.gguf",
    messages=[{"role": "user", "content": "Analyze this job posting..."}],
    max_tokens=4000
)
```

### With Tool Calling

```python
response = httpx.post(
    "http://192.168.1.107:4000/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Search for latest Python release"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }],
        "max_tokens": 4000
    },
    timeout=120.0
)

# Response contains structured tool_calls:
# {"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {"name": "web_search", "arguments": "{\"query\":\"...\"}"}}]}}
```

### Parallel Processing (7 jobs at once)

```python
import asyncio
import httpx

async def process_job(client, job_description):
    response = await client.post(
        "http://192.168.1.107:4000/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "Extract as JSON: {\"title\",\"company\",\"location\",\"salary\",\"skills\":[]}"},
                {"role": "user", "content": job_description}
            ],
            "max_tokens": 4000
        },
        timeout=120.0
    )
    return response.json()["choices"][0]["message"]["content"]

async def process_batch(jobs):
    async with httpx.AsyncClient() as client:
        tasks = [process_job(client, job) for job in jobs]
        return await asyncio.gather(*tasks)

# Process 7 jobs simultaneously — each goes to a different GPU
results = asyncio.run(process_batch(job_descriptions))
```

## Wake / Sleep

The rig supports Wake-on-LAN and auto-sleep. Use `rig_client.py` (already in `agent/rig_client.py`):

```python
from agent.rig_client import ensure_rig_ready, rig_sleep, RIG_API_URL

# Before processing — wakes rig if sleeping, waits for GPUs
status = ensure_rig_ready(min_gpus=6)
if not status:
    raise RuntimeError("Rig unavailable")

# Process jobs using RIG_API_URL (= http://192.168.1.107:4000/v1/chat/completions)
# ...

# After processing — suspend to RAM (models stay loaded, wakes in ~20s)
rig_sleep()
```

### CLI usage:
```bash
python agent/rig_client.py --wake      # Wake + wait for GPUs
python agent/rig_client.py --status    # Check status
python agent/rig_client.py --sleep     # Suspend to RAM
python agent/rig_client.py --shutdown  # Full power off
```

## Important Configuration Notes

### max_tokens: Use 4000+
Gemma 4 has thinking mode — it reasons internally before answering. With low max_tokens (200-500), the thinking consumes the entire budget and the answer is empty. Always use `max_tokens: 4000` minimum.

### Tool calling format
Use the OpenAI `tools` array in the request body (NOT text descriptions in the system prompt). The server automatically parses Gemma 4's native tool format into clean JSON.

### Timeout
Set HTTP timeout to 120s. With thinking mode and 16K context, complex responses can take 30-60 seconds.

### No streaming needed
Non-streaming works perfectly. If you want streaming, use `"stream": true` — it works but thinking tokens are hidden.

## Model Capabilities (Tested & Proven)

| Capability | Score | Evidence |
|-----------|-------|---------|
| Tool calling (single) | 100% | 8/8 tests passed |
| Tool calling (parallel) | Works | Called 2 tools in one response |
| Tool calling (choose correct) | Works | Picked calculator vs web_search correctly |
| Tool NOT called when unnecessary | Works | Answered "2+2=4" without calling tools |
| JSON extraction (simple) | Excellent | Perfect job posting extraction |
| JSON extraction (complex nested) | Excellent | Nested objects + arrays |
| Code generation | Excellent | Python, JavaScript, bug fixes |
| Reasoning (Bayes theorem) | Correct | Step-by-step with LaTeX |
| Logic puzzles | Correct | Systematic elimination |
| Multi-step math | Correct | $80 → $64 → $54.40 |
| Multi-turn memory | Works | Remembers name + location |
| Instruction following | Perfect | Exactly 5 items, numbered |
| Safety refusal | Works | Refuses harmful requests |

Full test report: `D:/_gpu_rig_ai/GEMMA4_TEST_REPORT.md`

## Architecture

```
Ritu's Job Agent (Windows PC 192.168.1.223)
    │
    │  POST http://192.168.1.107:4000/v1/chat/completions
    │
    ▼
nginx load balancer (port 4000, least_conn)
    │
    ├── GPU 0 (RX 5700 XT)  port 8080  ← Gemma 4 E4B, 63 tok/s
    ├── GPU 1 (RX 5700 XT)  port 8081  ← Gemma 4 E4B, 63 tok/s
    ├── GPU 2 (RX 6700 XT)  port 8082  ← Gemma 4 E4B, 63 tok/s
    ├── GPU 3 (RX 5700 XT)  port 8083  ← Gemma 4 E4B, 63 tok/s
    ├── GPU 4 (RX 5700 XT)  port 8084  ← Gemma 4 E4B, 63 tok/s
    ├── GPU 5 (RX 5700 XT)  port 8085  ← Gemma 4 E4B, 63 tok/s
    └── GPU 6 (RX 5700 XT)  port 8086  ← Gemma 4 E4B, 63 tok/s
```

## Performance

| Metric | Value |
|--------|-------|
| Single GPU speed | 63 tok/s generation, 250 tok/s prompt |
| 7-GPU throughput | ~440 tok/s total (7 × 63) |
| Context window | 16,384 tokens per request |
| Model load time | ~60s from SSD (on boot/wake) |
| Wake from sleep | ~20s (models stay in VRAM) |
| Concurrent requests | 7 (one per GPU) |
| Typical job analysis | 2-5 seconds per job |
| 100 jobs parallel | ~70 seconds (vs 8+ min on 1 GPU) |

## Rig SSH Access
```
ssh user@192.168.1.107    # key auth or password: 1
```

## Files
- `D:/_gpu_rig_ai/rig_client.py` — wake/sleep/status client
- `D:/_ritu_job_agent/agent/rig_client.py` — copy in agent project
- `D:/_gpu_rig_ai/GEMMA4_TEST_REPORT.md` — full test evidence
- `D:/_gpu_rig_ai/MODEL_COMPARISON_REPORT.md` — model comparison
