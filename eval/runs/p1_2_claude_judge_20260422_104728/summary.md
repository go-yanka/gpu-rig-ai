# CBIC RAG eval — Claude judge
- date: 2026-04-22T12:21:05
- gold: D:\_gpu_rig_ai\eval\gold_set.yaml
- items: 170
- api: http://192.168.1.107:9500
- judge: claude-cli (Max subscription)

## Headline
**Total: 387.0 / 1557.0 = 24.86%**
Deterministic-only (no judge): 233.0 / 1047.0 = 22.25%
Mean Claude judge score: 0.91 / 3 (n=170)

## Per category
| category | n | pts | max | pct | mean_claude |
|---|---:|---:|---:|---:|---:|
| central_excise | 10 | 31.0 | 93.0 | 33.33% | 1.00 |
| customs | 45 | 95.0 | 426.0 | 22.30% | 0.96 |
| gst | 81 | 201.0 | 730.0 | 27.53% | 0.90 |
| others | 14 | 16.0 | 123.0 | 13.01% | 0.71 |
| service_tax | 20 | 44.0 | 185.0 | 23.78% | 0.90 |