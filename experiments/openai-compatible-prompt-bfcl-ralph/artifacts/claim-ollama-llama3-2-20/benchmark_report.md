# OpenAI-Compatible Prompt-Mode BFCL Benchmark Report

- Generated (UTC): 2026-03-06T11:48:20.387988+00:00
- Provider: `Ollama`
- Model: `llama3.2:latest`
- Runtime Root: `/Users/kim/github_repos/ai-sdk-tool-calling-lab/experiments/openai-compatible-prompt-bfcl-ralph/runtime-claim-ollama-llama3-2-20`
- Categories: `multiple, parallel, parallel_multiple, simple_python`
- Cases per category: `20`
- Run-id mode: `enabled`

## Scoreboard

- Baseline: `llama3.2:latest (Prompt Baseline)`
- RALPH: `llama3.2:latest (Prompt + RALPH Loop)`

| Metric | Baseline | RALPH | Delta (pp) |
|---|---:|---:|---:|
| Overall Acc | 7.50 | 7.62 | +0.12 |
| Non-Live AST Acc | N/A | N/A | N/A |
| Live Acc | 0.00 | 0.00 | +0.00 |
| Multi Turn Acc | 0.00 | 0.00 | +0.00 |
| Relevance Detection | N/A | N/A | N/A |
| Irrelevance Detection | N/A | N/A | N/A |

## Headline

- Verdict: `improved`
- Wins: `1` | Losses: `0` | Ties: `2` | Unknown: `3`
- Best gain: `Overall Acc` (+0.12 pp)
- Missing metrics: `Non-Live AST Acc, Relevance Detection, Irrelevance Detection`
