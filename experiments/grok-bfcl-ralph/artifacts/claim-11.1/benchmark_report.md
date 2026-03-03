# Grok BFCL Benchmark Report

- Generated (UTC): 2026-03-03T05:29:42.392787+00:00
- Model: `grok-4-latest`
- Runtime Root: `/Users/kim/Downloads/ai-sdk-tool-call-middleware-main/experiments/grok-bfcl-ralph/runtime-tune-v1`
- Categories: `multiple, parallel, parallel_multiple, simple_python`
- Cases per category: `3`
- Run-id mode: `enabled`

## Scoreboard

- Baseline: `grok-4-latest (Prompt Baseline)`
- RALPH: `grok-4-latest (Prompt + RALPH Loop)`

| Metric | Baseline | RALPH | Delta (pp) |
|---|---:|---:|---:|
| Overall Acc | 7.50 | 8.33 | +0.83 |
| Non-Live AST Acc | N/A | N/A | N/A |
| Live Acc | 0.00 | 0.00 | +0.00 |
| Multi Turn Acc | 0.00 | 0.00 | +0.00 |
| Relevance Detection | N/A | N/A | N/A |
| Irrelevance Detection | N/A | N/A | N/A |

## Headline

- Verdict: `improved`
- Wins: `1` | Losses: `0` | Ties: `2` | Unknown: `3`
- Best gain: `Overall Acc` (+0.83 pp)
- Missing metrics: `Non-Live AST Acc, Relevance Detection, Irrelevance Detection`
