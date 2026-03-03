# Grok Tool Performance 11.1% Improvement

BFCL v4에서 Grok의 tool-calling 품질을 `baseline` 대비 `RALPH loop` 프롬프트로 비교/개선하는 프로젝트입니다.

- baseline: BFCL 기본 prompt 경로
- ralph-loop: RALPH(Read/Align/List/Plan/Hard-check) 2-pass prompt 경로
- 실행 스크립트: `run_grok_bfcl_ralph.py`

## Official Result

이 프로젝트의 공식 어필 수치는 아래 기준입니다.

- Baseline Overall Acc: `7.50`
- RALPH Overall Acc: `8.33`
- Absolute delta: `+0.83%p`
- Relative delta: `+11.1%`

세부 근거: `RESULTS.md`

## Evidence Artifacts

정리된 증빙 파일은 아래에 보관됩니다.

- `artifacts/claim-11.1/summary.json`
- `artifacts/claim-11.1/benchmark_report.md`
- `artifacts/claim-11.1/data_overall.csv`
- `artifacts/claim-11.1/error_forensics.json`

## Setup

### 1) BFCL 준비

```bash
git clone https://github.com/ShishirPatil/gorilla.git /Users/kim/Downloads/gorilla
cd /Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard
/opt/homebrew/bin/python3.11 -m venv .venv311
.venv311/bin/pip install -e .
.venv311/bin/pip install soundfile
```

### 2) xAI API Key

```bash
export GROK_API_KEY="xai-..."
```

## Run Benchmark

```bash
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  /Users/kim/Downloads/ai-sdk-tool-call-middleware-main/experiments/grok-bfcl-ralph/run_grok_bfcl_ralph.py \
  --bfcl-root /Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard \
  --runtime-root /Users/kim/Downloads/ai-sdk-tool-call-middleware-main/experiments/grok-bfcl-ralph/runtime \
  --model-name grok-4-latest \
  --categories simple_python,multiple,parallel,parallel_multiple \
  --cases-per-category 3 \
  --num-threads 1 \
  --report-markdown benchmark_report.md \
  --error-report-json error_forensics.json
```

## Preflight Only

```bash
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  /Users/kim/Downloads/ai-sdk-tool-call-middleware-main/experiments/grok-bfcl-ralph/run_grok_bfcl_ralph.py \
  --preflight-only \
  --bfcl-root /Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard \
  --runtime-root /Users/kim/Downloads/ai-sdk-tool-call-middleware-main/experiments/grok-bfcl-ralph/runtime \
  --model-name grok-4-latest \
  --categories simple_python,multiple,parallel \
  --cases-per-category 3
```

## Debug & Test

```bash
cd /Users/kim/Downloads/ai-sdk-tool-call-middleware-main/experiments/grok-bfcl-ralph
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python -m py_compile run_grok_bfcl_ralph.py
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python -m unittest -q test_run_grok_bfcl_ralph.py
```

## Release Cleanup

생성 산출물 정리 + 키 문자열 스캔:

```bash
cd /Users/kim/Downloads/ai-sdk-tool-call-middleware-main/experiments/grok-bfcl-ralph
./release_guard.sh --apply
```

## Notes

- `python3`(시스템 인터프리터)가 아니라 BFCL `.venv311` Python으로 실행해야 합니다.
- 짧은 샘플 런(`cases-per-category=3`)은 API 상태에 따라 점수 변동이 생길 수 있습니다.
- 업로드 전에는 반드시 키 회전 및 키 스캔을 권장합니다.
