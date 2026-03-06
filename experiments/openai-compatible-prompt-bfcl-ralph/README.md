# OpenAI-Compatible BFCL Prompt-Mode RALPH Benchmark

BFCL v4에서 **OpenAI-compatible prompt mode** 기준으로 baseline prompt와 `RALPH loop` 2-pass prompt를 비교하는 실험입니다.

- 대상: native tool calling이 없거나, 있어도 prompt mode로 비교하고 싶은 모델
- 실행 스크립트: `run_openai_compatible_prompt_bfcl_ralph.py`
- 방식: BFCL `is_fc_model=False` 경로로 baseline vs RALPH 비교

여러 모델을 한 번에 돌리고 분류하려면 `../prompt-bfcl-ralph-matrix`를 사용합니다.

## Checked-In Success Snapshot

- `llama3.2:latest` on Ollama (`20` cases/category): `7.50 -> 7.62`, `+0.12pp`, `+1.6%` relative
- Evidence:
  - `artifacts/claim-ollama-llama3-2-20/summary.json`
  - `artifacts/claim-ollama-llama3-2-20/benchmark_report.md`
  - `artifacts/claim-ollama-llama3-2-20/data_overall.csv`
  - `artifacts/claim-ollama-llama3-2-20/error_forensics.json`

![Ollama Llama 3.2 Prompt-Mode BFCL Improvement](https://raw.githubusercontent.com/KIM3310/ai-sdk-tool-calling-lab/main/experiments/openai-compatible-prompt-bfcl-ralph/artifacts/claim-ollama-llama3-2-20/benchmark-ollama-llama3-2-20.svg)

## Required Environment

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export OPENAI_COMPATIBLE_API_KEY="sk-..."
export OPENAI_COMPATIBLE_BASE_URL="https://your-provider.example/v1"
export OPENAI_COMPATIBLE_MODEL="your-model-id"
```

Optional metadata:

```bash
export OPENAI_COMPATIBLE_PROVIDER="OpenRouter"
export OPENAI_COMPATIBLE_DOCS_URL="https://openrouter.ai/models"
export OPENAI_COMPATIBLE_PROVIDER_LICENSE="Proprietary"
export OPENAI_COMPATIBLE_DEFAULT_HEADERS='{"HTTP-Referer":"https://example.com","X-Title":"bfcl-ralph"}'
```

## BFCL Setup

```bash
git clone https://github.com/ShishirPatil/gorilla.git /Users/kim/Downloads/gorilla
cd /Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard
/opt/homebrew/bin/python3.11 -m venv .venv311
.venv311/bin/pip install -e .
.venv311/bin/pip install soundfile
```

## Run Benchmark

```bash
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  "$REPO_ROOT/experiments/openai-compatible-prompt-bfcl-ralph/run_openai_compatible_prompt_bfcl_ralph.py" \
  --bfcl-root /Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard \
  --runtime-root "$REPO_ROOT/experiments/openai-compatible-prompt-bfcl-ralph/runtime" \
  --provider-name "${OPENAI_COMPATIBLE_PROVIDER:-OpenAI-Compatible}" \
  --model-name "$OPENAI_COMPATIBLE_MODEL" \
  --base-url "$OPENAI_COMPATIBLE_BASE_URL" \
  --categories simple_python,multiple,parallel,parallel_multiple \
  --cases-per-category 3 \
  --num-threads 1 \
  --report-markdown benchmark_report.md \
  --error-report-json error_forensics.json
```

If your provider does not expose a standard `GET /models` endpoint, add `--skip-model-check`.

## Preflight Only

```bash
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  "$REPO_ROOT/experiments/openai-compatible-prompt-bfcl-ralph/run_openai_compatible_prompt_bfcl_ralph.py" \
  --preflight-only \
  --bfcl-root /Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard \
  --runtime-root "$REPO_ROOT/experiments/openai-compatible-prompt-bfcl-ralph/runtime" \
  --provider-name "${OPENAI_COMPATIBLE_PROVIDER:-OpenAI-Compatible}" \
  --model-name "$OPENAI_COMPATIBLE_MODEL" \
  --base-url "$OPENAI_COMPATIBLE_BASE_URL" \
  --cases-per-category 3
```

## Debug & Test

```bash
cd "$REPO_ROOT/experiments/openai-compatible-prompt-bfcl-ralph"
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python -m py_compile run_openai_compatible_prompt_bfcl_ralph.py
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python -m unittest -q test_run_openai_compatible_prompt_bfcl_ralph.py
```

## Notes

- 이 실험은 native tools API가 아니라 BFCL prompt mode를 비교합니다.
- `RALPH loop`는 internal checklist 생성 후 final function-call 출력만 남기는 2-pass prompting 전략입니다.
- 상대 향상폭은 모델, provider, category 구성에 따라 달라지므로 Grok의 `+11.1%`를 그대로 기대하면 안 됩니다.
