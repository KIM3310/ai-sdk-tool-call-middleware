# Prompt-Mode BFCL RALPH Matrix

이 프로젝트는 여러 모델을 한 번에 돌려서:

- baseline prompt 성능
- RALPH loop prompt 성능
- `향상됨 / 유지 / 악화 / 실패`

를 한 번에 정리하는 대규모 실험 오케스트레이터입니다.

쉽게 말하면:

- `grok-prompt-bfcl-ralph`는 한 학생만 시험 보는 버전
- `openai-compatible-prompt-bfcl-ralph`는 다른 학생 한 명도 시험 볼 수 있는 버전
- `prompt-bfcl-ralph-matrix`는 **학생 여러 명을 한 교실에서 한꺼번에 채점하는 버전**

## What It Runs

- `grok` kind: [../grok-prompt-bfcl-ralph/run_grok_prompt_bfcl_ralph.py](/Users/kim/github_repos/ai-sdk-tool-calling-lab/experiments/grok-prompt-bfcl-ralph/run_grok_prompt_bfcl_ralph.py)
- `openai-compatible` kind: [../openai-compatible-prompt-bfcl-ralph/run_openai_compatible_prompt_bfcl_ralph.py](/Users/kim/github_repos/ai-sdk-tool-calling-lab/experiments/openai-compatible-prompt-bfcl-ralph/run_openai_compatible_prompt_bfcl_ralph.py)
- `kiro-cli` kind: [../kiro-cli-prompt-bfcl-ralph/run_kiro_cli_prompt_bfcl_ralph.py](/Users/kim/github_repos/ai-sdk-tool-calling-lab/experiments/kiro-cli-prompt-bfcl-ralph/run_kiro_cli_prompt_bfcl_ralph.py)

## Config

모델 목록은 `models.example.json`에 넣습니다.

```json
{
  "models": [
    {
      "id": "grok-4-latest",
      "kind": "grok",
      "enabled": true,
      "model_name": "grok-4-latest",
      "api_key_env": "GROK_API_KEY"
    }
  ]
}
```

OpenAI-compatible 예시:

```json
{
  "id": "openrouter-qwen3-coder",
  "kind": "openai-compatible",
  "enabled": true,
  "provider_name": "OpenRouter",
  "model_name": "qwen/qwen3-coder:free",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key_env": "OPENROUTER_API_KEY"
}
```

Kiro CLI 예시:

```json
{
  "id": "kiro-cli-default",
  "kind": "kiro-cli",
  "enabled": true,
  "provider_name": "Kiro CLI",
  "model_name": "kiro-cli-default",
  "cli_path": "kiro-cli",
  "trust_tools": ""
}
```

## Run

BFCL 가상환경에서 실행하는 게 안전합니다.

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/run_prompt_bfcl_ralph_matrix.py" \
  --models-file "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/models.example.json" \
  --bfcl-root /Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard \
  --runtime-root "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/runtime" \
  --cases-per-category 3
```

특정 모델만 선택:

```bash
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/run_prompt_bfcl_ralph_matrix.py" \
  --models-file "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/models.example.json" \
  --model-ids grok-4-latest,openrouter-qwen3-coder
```

사전 점검만 실행:

```bash
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/run_prompt_bfcl_ralph_matrix.py" \
  --models-file "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/models.example.json" \
  --preflight-only
```

로컬 Ollama만 바로 벤치마크:

```bash
/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard/.venv311/bin/python \
  "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/run_prompt_bfcl_ralph_matrix.py" \
  --models-file "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/models.ollama.local.json" \
  --runtime-root "$REPO_ROOT/experiments/prompt-bfcl-ralph-matrix/runtime-ollama" \
  --cases-per-category 5
```

## Output

매트릭스 실행 후 아래 파일이 생깁니다.

- `runtime/matrix_summary.json`
- `runtime/matrix_results.csv`
- `runtime/matrix_report.md`
- `runtime/runs/<model-id>/summary.json`
- `runtime/runs/<model-id>/benchmark_report.md`
- `runtime/runs/<model-id>/stdout.log`
- `runtime/runs/<model-id>/stderr.log`

자동 복구가 발동한 경우:

- `matrix_results.csv`에 `salvaged`, `salvage_note`가 기록됩니다.
- `matrix_report.md`에 `## Salvaged` 섹션이 추가됩니다.

## Classification Rules

- `improved`: Overall Acc delta가 양수
- `flat`: delta가 `--flat-threshold-pp` 이내
- `regressed`: Overall Acc delta가 음수
- `failed`: child runner가 실패하거나 필수 env가 없음

## Notes

- 기본 제공 `models.example.json`은 비용 방지를 위해 전부 `enabled: false`입니다.
- 실제 라이브 실행용 설정은 보통 `models.local.json`처럼 로컬 파일로 두고, API 키와 로컬 엔드포인트를 붙여서 돌립니다.
- 돈을 쓰지 않는 점검용 예시는 `models.zero-cost.local.json`에 들어 있습니다. 이 파일은 `Kiro preflight + Ollama local models` 조합만 포함합니다.
- 실제 무료 로컬 벤치마크 전용 구성은 `models.ollama.local.json`에 들어 있습니다.
- Ollama 로컬 구성은 실제 인증이 필요 없어서 `api_key: "dummy"`를 직접 넣어 두었습니다.
- 현재 무과금 로컬 예시에는 `llama3.2`, `llama3.1:8b`, `qwen2.5:1.5b`, `qwen3.5:4b`, `phi3`, `gemma3:4b`가 들어 있습니다.
- 이 프로젝트는 live API 호출이 들어갈 수 있으므로, 어떤 모델을 켤지 먼저 정해서 돌리는 게 맞습니다.
- 현재 매트릭스가 직접 지원하는 대상은 `grok`과 `openai-compatible` HTTP 엔드포인트입니다.
- `Kiro CLI`는 이제 별도 runner를 통해 연결되지만, `--preflight-only` 외의 실제 benchmark는 credits를 소비할 수 있습니다.
- 매트릭스 runner는 이제 child runner가 결과 파일 생성 후 멈추는 경우 `eval-only salvage`를 시도합니다.
