# Resume Bullets

Use the version that matches the role. Do not paste all of them at once.

## Applied AI / Frontier LLM

- Built an evaluation-driven LLM tooling lab around prompt-mode tool use, spanning parser middleware, BFCL benchmark runners, multi-model matrix orchestration, and operator-facing inspection surfaces.
- Generalized a Grok-specific BFCL prompt experiment into reusable runners for OpenAI-compatible models and Kiro CLI, then classified outcomes across models as improved, flat, regressed, or failed.
- Hardened small-sample benchmark reliability by adding salvage logic that reconstructs summaries from per-category BFCL score artifacts when aggregate evaluation breaks or stalls.
- Converted raw benchmark scripts into a local service with job launch, cancellation, runtime report inspection, and browser-accessible stdout/stderr logs.

## AI Infra / Platform

- Designed and implemented a local evaluation service for LLM benchmarking, including typed API routes, runtime artifact management, failure handling, and test coverage.
- Added recovery paths for flaky benchmark infrastructure, preserving partial results and converting otherwise failed runs into inspectable, reproducible outputs.
- Drove repository quality to green with formatter, lint, typecheck, and full test suite validation across `184` test files / `1691` tests.

## Product / Full-Stack AI Engineer

- Built `StagePilot`, a product-shaped multi-agent orchestration slice with planning, benchmarking, insight generation, what-if simulation, and operator notification flows.
- Exposed both library and service surfaces for LLM workflows, bridging backend orchestration, browser UI, operational tooling, and experiment reproducibility.
- Documented scope boundaries and experimental caveats clearly to avoid overstating model gains or conflating prompt-mode behavior with native tool APIs.

## Short project line

- Created an LLM tooling lab that combines tool-call middleware, eval infrastructure, recovery-aware benchmarking, and operator-facing services for prompt-mode function calling systems.
