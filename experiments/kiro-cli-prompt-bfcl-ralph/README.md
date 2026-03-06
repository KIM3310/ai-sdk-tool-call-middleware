# Kiro CLI Prompt-Mode BFCL RALPH

이 실험은 `kiro-cli chat`를 BFCL prompt-mode 함수호출 벤치마크에 연결해서:

- baseline prompt 성능
- RALPH loop prompt 성능
- delta(pp)

를 비교합니다.

쉽게 말하면:

- 같은 문제를 `Kiro CLI`에 그냥 시키는 버전
- 먼저 체크리스트를 만들고 다시 답하게 하는 `RALPH` 버전

을 나란히 시험 보는 실험입니다.

## Important Cost Note

- `--preflight-only`는 로그인, CLI 설치, BFCL import, 카테고리 설정만 확인합니다.
- `--preflight-only`는 **Kiro credits를 쓰지 않도록 설계**했습니다.
- 실제 benchmark run은 `kiro-cli chat`를 여러 번 호출하므로 **Kiro credits를 소비할 수 있습니다**.
- 돈이 나가면 안 되는 상황이면 먼저 `Kiro Free`인지와 overage가 꺼져 있는지 Kiro 대시보드에서 확인한 뒤 실행하세요.

공식 참고:

- Pricing: `https://kiro.dev/pricing/`
- CLI: `https://kiro.dev/cli/`

## Run Preflight Only

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  "$REPO_ROOT/experiments/kiro-cli-prompt-bfcl-ralph/run_kiro_cli_prompt_bfcl_ralph.py" \
  --bfcl-root /Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard \
  --runtime-root "$REPO_ROOT/experiments/kiro-cli-prompt-bfcl-ralph/runtime" \
  --model-name kiro-cli-default \
  --preflight-only
```

## Run Benchmark

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  "$REPO_ROOT/experiments/kiro-cli-prompt-bfcl-ralph/run_kiro_cli_prompt_bfcl_ralph.py" \
  --bfcl-root /Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard \
  --runtime-root "$REPO_ROOT/experiments/kiro-cli-prompt-bfcl-ralph/runtime" \
  --model-name kiro-cli-default \
  --cases-per-category 3
```

선택 옵션:

- `--kiro-model <name>`
- `--kiro-agent <name>`
- `--trust-tools ''`
- `--request-timeout-sec 120`

## Output

- `runtime/summary.json`
- `runtime/benchmark_report.md`
- `runtime/error_forensics.json`
- `runtime/score/data_overall.csv`
- `runtime/stdout.log`
- `runtime/stderr.log`
