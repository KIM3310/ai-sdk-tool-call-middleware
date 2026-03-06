# Hiring Brief

This repository is not a toy demo. It shows three concrete engineering muscles that matter for big-tech and frontier-LLM roles:

1. Evaluation-driven LLM engineering
- Built prompt-mode BFCL benchmarking flows for Grok, OpenAI-compatible models, Kiro CLI, and a matrix runner that classifies `improved / flat / regressed / failed`.
- Kept claims precise: the Grok experiment is about BFCL prompt-mode function calling, not native xAI tools API performance.
- Added salvage logic so small or partially hung benchmark runs can still recover usable summaries instead of silently failing.

2. Production-minded backend and tooling
- Turned raw benchmark scripts into an operator-facing local service (`BenchLab`) with job launch, cancellation, runtime inspection, and per-job log viewing.
- Hardened service flows with typed APIs, structured artifacts, runtime guards, and automated tests.
- Reduced routing complexity and closed the loop with formatter, lint, typecheck, and test green status.

3. Applied systems product sense
- Built `StagePilot`, a multi-agent orchestration skeleton with planning, benchmarking, insights, what-if simulation, and operator notification flows.
- Exposed both browser-facing and API-facing surfaces, not just library code.
- Preserved separation between package surface, experiment surface, and demo-service surface.

## Why this is a strong hiring signal

Most LLM repos show one of these:
- a prompt hack
- a benchmark notebook
- an API wrapper

This repo shows the full stack:
- model/middleware layer
- evaluation harness
- failure recovery
- service layer
- UI for operators
- docs and test discipline

That combination maps well to roles such as:
- Applied AI Engineer
- LLM Evals / Reliability Engineer
- AI Product Engineer
- Inference / Tool-use Platform Engineer
- Frontier model post-training / scaffolding engineer

## Concrete evidence in this repo

### Parser / middleware package
- Published package surface: `@ai-sdk-tool/parser`
- Handles tool-call parsing for models that do not natively support `tools` or behave better with prompt-shaped protocols.

### BenchLab service
- Local operator UI/API for matrix experiments.
- Supports:
  - config discovery
  - launching benchmark and preflight jobs
  - canceling running jobs
  - runtime report inspection
  - stdout/stderr log viewing from the browser

### Prompt-mode BFCL research
- Grok-specific prompt experiment.
- OpenAI-compatible generalized runner.
- Kiro CLI runner.
- Matrix runner for multi-model comparisons.

### StagePilot
- Multi-agent orchestration slice with:
  - intake planning
  - benchmark endpoint
  - insight derivation
  - what-if simulation
  - operator notification bridge

## Results worth mentioning

Use only claims that are explicitly true in the repo:

- StagePilot benchmark artifact shows:
  - baseline: `29.17%`
  - middleware: `62.50%`
  - middleware + ralph-loop: `100%`
  - source: `docs/benchmarks/stagepilot-latest.json`

- Local BFCL prompt-mode matrix runs show that RALPH is not universally positive.
  - Example: some local Llama runs improved, some Qwen/Gemma/Phi runs regressed or stayed flat.
  - This is a positive hiring signal because it shows honest evaluation instead of one-way cherry-picking.

- BenchLab and StagePilot now pass repository quality gates:
  - `npm run check:biome`
  - `npm run check:types`
  - `npm test`

## What to say in interviews

Say:
- “I separate package claims from experiment claims.”
- “I treat evals as products: reproducible, inspectable, and debuggable.”
- “When BFCL aggregation broke on tiny sample sizes, I added recovery from per-category score artifacts rather than hand-waving the failure.”
- “I turned benchmark scripts into an operator-facing service so the system could actually be used.”

Do not say:
- “I improved Grok native tool calling.”
- “RALPH always improves function calling.”
- “This proves universal gains across models.”

## Good role fit

Best fit:
- teams building model tooling, evals, orchestration, or reliability
- teams that care about correctness under messy real-world model behavior
- teams that need engineers who can bridge research experiments and usable services

Weaker fit:
- purely academic training roles with no product or systems ownership
- traditional CRUD-only backend roles where this work will be underused

## Suggested demo order

1. Show the parser package in `README.md`.
2. Show `BenchLab` and explain why a benchmark needs an operator surface.
3. Show matrix results and emphasize honest regressions as well as improvements.
4. Show the tiny-run salvage fix as an example of engineering under imperfect tooling.
5. Show `StagePilot` to prove you can build product-shaped systems, not just eval scripts.

## One-sentence summary

This repository demonstrates that the author can take LLM behavior from raw prompting experiments to reproducible evaluation infrastructure and then all the way to a usable service with tests, recovery paths, and honest reporting.
