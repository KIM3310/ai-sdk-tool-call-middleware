# Interview Talk Track

## 30-second version

I built a repository that covers three layers of LLM engineering: tool-call middleware for models that do not reliably support native tools, BFCL-based prompt-mode benchmarking across multiple providers, and service surfaces that make those experiments operable and debuggable. The strongest part is not just the benchmark itself, but that I turned it into a reproducible system with tests, logging, and recovery paths.

## 2-minute version

I started from a parser/middleware package for AI SDK models that need prompt-shaped tool protocols. From there I added research-style BFCL experiments for Grok and generalized the same approach to OpenAI-compatible models and Kiro CLI. Then I built a matrix runner to compare many models and classify whether a prompt strategy improved, stayed flat, regressed, or failed.

The important engineering move was refusing to stop at scripts. I added `BenchLab`, a local operator UI/API that can launch benchmark jobs, cancel them, inspect runtime reports, and read stdout/stderr logs. I also hardened the benchmark path itself. For example, very small BFCL runs could fail during aggregate scoring because the upstream evaluator expected more than one latency datapoint. Instead of accepting that as a dead run, I added a fallback that reconstructs a usable summary from per-category score artifacts and marks the run as salvaged.

That project shows how I work: precise claims, eval-first thinking, and productizing research code so it can actually be used and debugged.

## 5-minute walkthrough

1. Problem
- Many models do not reliably support native tool calling, so prompt-shaped tool protocols are still operationally useful.
- Raw prompt experiments are not enough; they need reproducible evaluation and inspectable runtime artifacts.

2. What I built
- Middleware layer for parsing tool calls in AI SDK workflows.
- BFCL prompt-mode runners for Grok, OpenAI-compatible providers, and Kiro CLI.
- Matrix orchestration to compare many models at once.
- BenchLab service to operate and inspect runs.
- StagePilot as a separate product-shaped multi-agent example.

3. Hard problem I solved
- Tiny benchmark runs exposed brittleness in upstream aggregation.
- I added fallback summary recovery from per-category score JSONs so a run with valid model outputs does not get lost because of an evaluator-side stats assumption.

4. Why it matters
- It proves I can bridge experimentation, infrastructure, and usable product surfaces.
- It also shows that I do not overclaim results: some models improved, some stayed flat, and some regressed.

## STAR story

### Situation
- I had a repo with useful LLM middleware and growing benchmark experiments, but the experiment layer was too script-heavy and fragile for serious reuse.

### Task
- Turn it into something that is reproducible, debuggable, and credible to other engineers, while keeping claims technically honest.

### Action
- Added generalized benchmark runners and a model matrix.
- Built a local BenchLab API/UI for operating runs.
- Added failure recovery, including salvage from partial BFCL artifacts.
- Reworked routing and validation until formatter, lint, typecheck, and tests all passed.

### Result
- The repo now demonstrates package engineering, evaluation infrastructure, and serviceization in one place.
- Full repo checks are green.
- The project is easier to demo, easier to trust, and easier to discuss in a hiring loop.

## Good answers to likely questions

### “What’s the most impressive technical decision here?”
- Separating claims from surfaces. I kept the package, experiments, and services distinct so the repo stays truthful and maintainable.

### “What bug are you proud of fixing?”
- The small-sample BFCL failure. The model outputs were valid, but aggregate scoring broke. I added a recovery path that reconstructs usable summaries from score artifacts instead of discarding the run.

### “How do you evaluate LLM systems?”
- I prefer reproducible benchmark harnesses with runtime artifacts, not screenshot metrics. I also care about negative results because they prevent bad product decisions.

### “How do you think about product vs research?”
- Research gives you hypotheses. Product engineering gives you observability, failure handling, and operational ergonomics. Good AI work needs both.

## What to emphasize for frontier LLM roles

- evaluation rigor
- honest handling of regressions
- failure recovery
- model behavior instrumentation
- bridging prompt systems and usable tooling

## What to emphasize for big-tech platform roles

- API/service design
- reliability and recovery paths
- operator tooling
- quality gates and test discipline
- scope control and clear documentation
