#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHARED_PATH = (
    Path(__file__).resolve().parents[1]
    / "grok-prompt-bfcl-ralph"
    / "run_grok_prompt_bfcl_ralph.py"
)
SHARED = runpy.run_path(str(SHARED_PATH), run_name="prompt_bfcl_ralph_shared")

DEFAULT_CATEGORIES = SHARED["DEFAULT_CATEGORIES"]
RALPH_SYSTEM_PROMPT_SUFFIX = SHARED["RALPH_SYSTEM_PROMPT_SUFFIX"]
RALPH_PREFLIGHT_PROMPT = SHARED["RALPH_PREFLIGHT_PROMPT"]
RALPH_FINAL_PROMPT = SHARED["RALPH_FINAL_PROMPT"]
format_missing_dependency_error = SHARED["format_missing_dependency_error"]
parse_categories = SHARED["parse_categories"]
resolve_runtime_categories = SHARED["resolve_runtime_categories"]
bootstrap_bfcl_imports = SHARED["bootstrap_bfcl_imports"]
create_run_ids_map = SHARED["create_run_ids_map"]
write_run_ids_file = SHARED["write_run_ids_file"]
run_generation_and_eval = SHARED["run_generation_and_eval"]
load_score_rows = SHARED["load_score_rows"]
build_summary = SHARED["build_summary"]
compute_metric_outcomes = SHARED["compute_metric_outcomes"]
resolve_report_markdown_path = SHARED["resolve_report_markdown_path"]
resolve_error_report_json_path = SHARED["resolve_error_report_json_path"]
clear_stale_output_file = SHARED["clear_stale_output_file"]

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def env_or_none(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark prompt-mode baseline vs RALPH-loop prompt on BFCL "
            "through Kiro CLI chat."
        )
    )
    parser.add_argument(
        "--bfcl-root",
        type=Path,
        default=Path("/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard"),
        help="Path to BFCL root directory (contains bfcl_eval package).",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runtime",
        help="Output/runtime root for BFCL result/, score/, metadata, run-id file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=env_or_none("KIRO_BFCL_MODEL_NAME") or "kiro-cli-default",
        help="BFCL display model ID for this Kiro CLI run.",
    )
    parser.add_argument(
        "--provider-name",
        type=str,
        default=env_or_none("KIRO_PROVIDER_NAME") or "Kiro CLI",
        help="Provider label shown in BFCL score rows and reports.",
    )
    parser.add_argument(
        "--provider-docs-url",
        type=str,
        default=env_or_none("KIRO_PROVIDER_DOCS_URL") or "https://kiro.dev/cli/",
        help="Optional documentation URL used in BFCL metadata.",
    )
    parser.add_argument(
        "--provider-license",
        type=str,
        default=env_or_none("KIRO_PROVIDER_LICENSE") or "Proprietary",
        help="License label recorded in BFCL metadata.",
    )
    parser.add_argument(
        "--cli-path",
        type=str,
        default=env_or_none("KIRO_CLI_PATH") or "kiro-cli",
        help="Kiro CLI executable path.",
    )
    parser.add_argument(
        "--kiro-model",
        type=str,
        default=env_or_none("KIRO_MODEL"),
        help="Optional Kiro CLI --model override.",
    )
    parser.add_argument(
        "--kiro-agent",
        type=str,
        default=env_or_none("KIRO_AGENT"),
        help="Optional Kiro CLI --agent override.",
    )
    parser.add_argument(
        "--trust-tools",
        type=str,
        default=env_or_none("KIRO_TRUST_TOOLS") or "",
        help="Value passed to Kiro CLI --trust-tools. Empty string means trust no tools.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
        help=(
            "Comma-separated BFCL categories. Default: "
            "simple_python,multiple,parallel,parallel_multiple"
        ),
    )
    parser.add_argument(
        "--cases-per-category",
        type=int,
        default=20,
        help="If > 0, evaluate first N test IDs per category via --run-ids mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="Benchmark metadata only. Kiro CLI does not expose a temperature flag.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Parallel BFCL worker count.",
    )
    parser.add_argument(
        "--skip-login-check",
        action="store_true",
        help="Skip the Kiro CLI login check. Not recommended.",
    )
    parser.add_argument(
        "--include-input-log",
        action="store_true",
        help="Enable BFCL verbose inference input logs.",
    )
    parser.add_argument(
        "--allow-agentic-run-ids",
        action="store_true",
        help="Allow memory/web_search categories when --cases-per-category > 0.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run environment/category/login diagnostics and exit without consuming credits.",
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=None,
        help="Optional markdown report output path. Relative paths resolve from --runtime-root.",
    )
    parser.add_argument(
        "--error-report-json",
        type=Path,
        default=None,
        help="Optional JSON file for inference error forensic summary.",
    )
    parser.add_argument(
        "--error-report-top-n",
        type=int,
        default=5,
        help="How many top error reasons to print per model when failures are present.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=120.0,
        help="Per-request timeout (seconds) for each kiro-cli chat subprocess.",
    )
    parser.add_argument(
        "--max-step-limit",
        type=int,
        default=20,
        help="Override BFCL MAXIMUM_STEP_LIMIT to cap per-turn tool-call loops.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.cases_per_category < 0:
        raise SystemExit("--cases-per-category must be >= 0.")
    if args.num_threads <= 0:
        raise SystemExit("--num-threads must be >= 1.")
    if args.temperature < 0:
        raise SystemExit("--temperature must be >= 0.")
    if not args.model_name or not args.model_name.strip():
        raise SystemExit("--model-name must not be blank.")
    if not args.provider_name or not args.provider_name.strip():
        raise SystemExit("--provider-name must not be blank.")
    if not args.cli_path or not args.cli_path.strip():
        raise SystemExit("--cli-path must not be blank.")
    if args.error_report_top_n <= 0:
        raise SystemExit("--error-report-top-n must be >= 1.")
    if args.request_timeout_sec <= 0:
        raise SystemExit("--request-timeout-sec must be > 0.")
    if args.max_step_limit <= 0:
        raise SystemExit("--max-step-limit must be >= 1.")


def resolve_cli_path(cli_path: str) -> str:
    candidate = cli_path.strip()
    if not candidate:
        raise SystemExit("--cli-path must not be blank.")
    if "/" in candidate:
        path = Path(candidate).expanduser()
        if not path.exists():
            raise SystemExit(f"Kiro CLI executable not found: {path}")
        return str(path)

    resolved = shutil.which(candidate)
    if not resolved:
        raise SystemExit(
            f"Could not find Kiro CLI on PATH: {candidate}. Install it or pass --cli-path."
        )
    return resolved


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text).replace("\r", "")


def sanitize_kiro_output(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "Opening browser..." in line:
            continue
        if "Press (^) + C to cancel" in line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def extract_kiro_error(stdout: str, stderr: str) -> str:
    combined = []
    for source in (stderr, stdout):
        for raw_line in strip_ansi(source).splitlines():
            line = raw_line.strip()
            if line:
                combined.append(line)
    if not combined:
        return "kiro-cli failed without output"
    return combined[-1][-400:]


def build_kiro_chat_command(
    *,
    cli_path: str,
    prompt: str,
    kiro_model: str | None,
    kiro_agent: str | None,
    trust_tools: str,
) -> list[str]:
    command = [
        cli_path,
        "chat",
        "--no-interactive",
        "--wrap",
        "never",
        f"--trust-tools={trust_tools}",
    ]
    if isinstance(kiro_agent, str) and kiro_agent.strip():
        command.extend(["--agent", kiro_agent.strip()])
    if isinstance(kiro_model, str) and kiro_model.strip():
        command.extend(["--model", kiro_model.strip()])
    command.append(prompt)
    return command


def check_kiro_login(cli_path: str) -> str:
    proc = subprocess.run(
        [cli_path, "whoami"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        message = extract_kiro_error(proc.stdout, proc.stderr)
        raise SystemExit(
            "Kiro CLI login check failed: "
            f"{message}. Run `kiro-cli login --license free --use-device-flow` first."
        )

    identity = sanitize_kiro_output(proc.stdout)
    if not identity:
        raise SystemExit("Kiro CLI login check returned an empty identity.")
    return identity


def ensure_runtime_layout(
    runtime_root: Path,
    *,
    cli_path: str,
    kiro_model: str | None,
    kiro_agent: str | None,
    trust_tools: str,
) -> None:
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "result").mkdir(exist_ok=True)
    (runtime_root / "score").mkdir(exist_ok=True)

    for stale_file in [
        runtime_root / "summary.json",
        runtime_root / "test_case_ids_to_generate.json",
    ]:
        if stale_file.exists():
            stale_file.unlink()

    for stale_csv in (runtime_root / "score").glob("*.csv"):
        stale_csv.unlink()

    for stale_path in (runtime_root / "score").iterdir():
        if stale_path.is_dir():
            shutil.rmtree(stale_path)

    env_lines = [
        f"KIRO_CLI_PATH={cli_path}",
        f"KIRO_MODEL={kiro_model or ''}",
        f"KIRO_AGENT={kiro_agent or ''}",
        f"KIRO_TRUST_TOOLS={trust_tools}",
    ]
    (runtime_root / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")


def run_preflight_checks(
    *,
    bfcl_root: Path,
    runtime_root: Path,
    provider_name: str,
    model_name: str,
    categories: list[str],
    cases_per_category: int,
    allow_agentic_run_ids: bool,
    cli_path: str,
    skip_login_check: bool,
) -> None:
    checks: list[tuple[str, str]] = []

    def add(status: str, message: str) -> None:
        checks.append((status, message))

    add("PASS", f"BFCL root found: {bfcl_root}")

    try:
        bootstrap_bfcl_imports(bfcl_root, runtime_root)
        from bfcl_eval.utils import load_dataset_entry
    except ModuleNotFoundError as exc:
        raise SystemExit(format_missing_dependency_error(exc)) from exc

    add("PASS", "BFCL imports available")

    run_ids_enabled = cases_per_category > 0
    resolved_categories = resolve_runtime_categories(
        categories=categories,
        run_ids_enabled=run_ids_enabled,
        allow_agentic_run_ids=allow_agentic_run_ids,
    )
    add(
        "PASS",
        f"Category validation passed ({len(resolved_categories)} categories)",
    )

    if run_ids_enabled:
        run_ids_map = create_run_ids_map(resolved_categories, cases_per_category)
        total_ids = sum(len(ids) for ids in run_ids_map.values())
        add(
            "PASS",
            f"Run-id sampling ready ({total_ids} IDs across {len(run_ids_map)} categories)",
        )
    else:
        add("INFO", "Run-id sampling disabled (cases_per_category=0)")
        for category in resolved_categories:
            try:
                entries = load_dataset_entry(category, include_prereq=False)
            except Exception as exc:
                raise SystemExit(
                    f"Dataset probe failed for category '{category}': {exc}"
                ) from exc
            add("PASS", f"Dataset probe ok for {category} ({len(entries)} entries)")

    add("PASS", f"Kiro CLI found: {cli_path}")

    if skip_login_check:
        add("WARN", "Skipped Kiro CLI login check (--skip-login-check)")
    else:
        identity = check_kiro_login(cli_path)
        add("PASS", f"Kiro CLI login check passed ({identity})")

    print("\n=== Preflight Report ===")
    for status, message in checks:
        print(f"[{status}] {message}")
    print(f"[INFO] Provider: {provider_name}")
    print(f"[INFO] Model Label: {model_name}")
    print(f"[INFO] Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("Preflight completed successfully.")


def render_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def render_prompt_transcript(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = render_message_content(message.get("content", ""))
        name = message.get("name")
        if isinstance(name, str) and name.strip():
            blocks.append(f"[{role} name={name.strip()}]\n{content}")
        else:
            blocks.append(f"[{role}]\n{content}")

    blocks.append(
        "[SYSTEM]\n"
        "Follow the conversation exactly. "
        "Answer only the latest user request while preserving all earlier system instructions."
    )
    return "\n\n".join(blocks).strip()


def register_custom_models(
    *,
    model_name: str,
    cli_path: str,
    kiro_model: str | None,
    kiro_agent: str | None,
    trust_tools: str,
    request_timeout_sec: float,
    provider_name: str,
    provider_docs_url: str | None,
    provider_license: str,
) -> tuple[str, str, str, str]:
    try:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
        from bfcl_eval.constants.enums import ModelStyle
        from bfcl_eval.model_handler.base_handler import BaseHandler
        from bfcl_eval.model_handler.utils import (
            format_execution_results_prompting,
            system_prompt_pre_processing_chat_model,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(format_missing_dependency_error(exc)) from exc

    class KiroCLIPromptHandler(BaseHandler):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.model_style = ModelStyle.OPENAI_COMPLETIONS
            self.cli_path = cli_path
            self.kiro_model = kiro_model
            self.kiro_agent = kiro_agent
            self.trust_tools = trust_tools
            self.request_timeout_sec = request_timeout_sec

        def _run_kiro_chat(self, prompt: str) -> tuple[str, float]:
            command = build_kiro_chat_command(
                cli_path=self.cli_path,
                prompt=prompt,
                kiro_model=self.kiro_model,
                kiro_agent=self.kiro_agent,
                trust_tools=self.trust_tools,
            )
            start_time = time.time()
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.request_timeout_sec,
            )
            end_time = time.time()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Kiro CLI request failed: {extract_kiro_error(proc.stdout, proc.stderr)}"
                )

            content = sanitize_kiro_output(proc.stdout)
            if not content:
                raise RuntimeError("Kiro CLI returned empty output.")

            return content, end_time - start_time

        def _query_prompting(self, inference_data: dict):
            prompt = render_prompt_transcript(inference_data["message"])
            inference_data["inference_input_log"] = {
                "message": repr(inference_data["message"]),
                "rendered_prompt": prompt,
            }

            content, latency = self._run_kiro_chat(prompt)
            return {
                "content": content,
                "input_token": 0,
                "output_token": 0,
            }, latency

        def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
            functions: list = test_entry["function"]
            test_entry_id: str = test_entry["id"]

            test_entry["question"][0] = system_prompt_pre_processing_chat_model(
                test_entry["question"][0], functions, test_entry_id
            )

            return {"message": []}

        def _parse_query_response_prompting(self, api_response: Any) -> dict:
            return {
                "model_responses": api_response["content"],
                "model_responses_message_for_chat_history": {
                    "role": "assistant",
                    "content": api_response["content"],
                },
                "input_token": api_response.get("input_token", 0),
                "output_token": api_response.get("output_token", 0),
            }

        def add_first_turn_message_prompting(
            self, inference_data: dict, first_turn_message: list[dict]
        ) -> dict:
            inference_data["message"].extend(first_turn_message)
            return inference_data

        def _add_next_turn_user_message_prompting(
            self, inference_data: dict, user_message: list[dict]
        ) -> dict:
            inference_data["message"].extend(user_message)
            return inference_data

        def _add_assistant_message_prompting(
            self, inference_data: dict, model_response_data: dict
        ) -> dict:
            inference_data["message"].append(
                model_response_data["model_responses_message_for_chat_history"]
            )
            return inference_data

        def _add_execution_results_prompting(
            self, inference_data: dict, execution_results: list[str], model_response_data: dict
        ) -> dict:
            formatted_results_message = format_execution_results_prompting(
                inference_data, execution_results, model_response_data
            )
            inference_data["message"].append(
                {"role": "user", "content": formatted_results_message}
            )
            return inference_data

    class KiroCLIRalphLoopPromptHandler(KiroCLIPromptHandler):
        def _query_prompting(self, inference_data: dict):
            base_messages = deepcopy(inference_data["message"])

            analysis_messages = deepcopy(base_messages)
            analysis_messages.append({"role": "system", "content": RALPH_PREFLIGHT_PROMPT})
            analysis_prompt = render_prompt_transcript(analysis_messages)
            analysis_text, analysis_latency = self._run_kiro_chat(analysis_prompt)

            final_messages = deepcopy(base_messages)
            if analysis_text.strip():
                final_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Internal RALPH checklist (do not quote this in output):\n"
                            f"{analysis_text[:1200]}"
                        ),
                    }
                )
            final_messages.append({"role": "system", "content": RALPH_FINAL_PROMPT})

            final_prompt = render_prompt_transcript(final_messages)
            inference_data["inference_input_log"] = {
                "base_message": repr(base_messages),
                "ralph_analysis_prompt": RALPH_PREFLIGHT_PROMPT,
                "ralph_final_prompt": RALPH_FINAL_PROMPT,
                "rendered_final_prompt": final_prompt,
            }

            final_text, final_latency = self._run_kiro_chat(final_prompt)
            total_latency = analysis_latency + final_latency

            lowered = final_text.lower().strip()
            looks_empty_or_refusal = (
                lowered in {"", "[]"}
                or "cannot comply" in lowered
                or "i cannot" in lowered
                or "i'm sorry" in lowered
            )
            if looks_empty_or_refusal:
                fallback_prompt = render_prompt_transcript(base_messages)
                final_text, fallback_latency = self._run_kiro_chat(fallback_prompt)
                total_latency += fallback_latency

            return {
                "content": final_text,
                "input_token": 0,
                "output_token": 0,
                "ralph_analysis": analysis_text,
            }, total_latency

        def _parse_query_response_prompting(self, api_response: Any) -> dict:
            response_data = super()._parse_query_response_prompting(api_response)
            ralph_analysis = api_response.get("ralph_analysis")
            if isinstance(ralph_analysis, str) and ralph_analysis.strip():
                response_data["ralph_analysis"] = ralph_analysis
            return response_data

    baseline_registry = f"{model_name}-prompt-baseline"
    ralph_registry = f"{model_name}-prompt-ralph-loop"
    baseline_display = f"{model_name} (Prompt Baseline)"
    ralph_display = f"{model_name} (Prompt + RALPH Loop)"

    model_url = provider_docs_url or "https://kiro.dev/cli/"

    MODEL_CONFIG_MAPPING[baseline_registry] = ModelConfig(
        model_name=model_name,
        display_name=baseline_display,
        url=model_url,
        org=provider_name,
        license=provider_license,
        model_handler=KiroCLIPromptHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    )

    MODEL_CONFIG_MAPPING[ralph_registry] = ModelConfig(
        model_name=model_name,
        display_name=ralph_display,
        url=model_url,
        org=provider_name,
        license=provider_license,
        model_handler=KiroCLIRalphLoopPromptHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    )

    return baseline_registry, ralph_registry, baseline_display, ralph_display


def build_markdown_report(
    *,
    summary: dict[str, Any],
    provider_name: str,
    model_name: str,
    baseline_display: str,
    ralph_display: str,
    runtime_root: Path,
    run_ids_enabled: bool,
) -> str:
    metrics = summary["metrics_percent_point"]
    outcomes = compute_metric_outcomes(metrics)

    lines = [
        "# Kiro CLI Prompt-Mode BFCL Benchmark Report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Provider: `{provider_name}`",
        f"- Model: `{model_name}`",
        f"- Runtime Root: `{runtime_root}`",
        f"- Categories: `{', '.join(summary['categories'])}`",
        f"- Cases per category: `{summary['cases_per_category']}`",
        f"- Run-id mode: `{'enabled' if run_ids_enabled else 'disabled'}`",
        "",
        "## Scoreboard",
        "",
        f"- Baseline: `{baseline_display}`",
        f"- RALPH: `{ralph_display}`",
        "",
        "| Metric | Baseline | RALPH | Delta (pp) |",
        "|---|---:|---:|---:|",
    ]

    for metric, values in metrics.items():
        base_val = values["baseline"]
        ralph_val = values["ralph"]
        delta = values["delta"]
        base_txt = "N/A" if base_val is None else f"{base_val:.2f}"
        ralph_txt = "N/A" if ralph_val is None else f"{ralph_val:.2f}"
        delta_txt = "N/A" if delta is None else f"{delta:+.2f}"
        lines.append(f"| {metric} | {base_txt} | {ralph_txt} | {delta_txt} |")

    wins = outcomes["wins"]
    losses = outcomes["losses"]
    ties = outcomes["ties"]
    unknown = outcomes["unknown"]

    verdict = "mixed"
    if len(wins) > len(losses):
        verdict = "improved"
    elif len(wins) < len(losses):
        verdict = "regressed"
    elif wins and losses:
        verdict = "balanced"

    lines.extend(
        [
            "",
            "## Headline",
            "",
            f"- Verdict: `{verdict}`",
            f"- Wins: `{len(wins)}` | Losses: `{len(losses)}` | "
            f"Ties: `{len(ties)}` | Unknown: `{len(unknown)}`",
        ]
    )

    if wins:
        best_metric, best_delta = wins[0]
        lines.append(f"- Best gain: `{best_metric}` ({best_delta:+.2f} pp)")
    if losses:
        worst_metric, worst_delta = losses[0]
        lines.append(f"- Biggest drop: `{worst_metric}` ({worst_delta:+.2f} pp)")
    if unknown:
        lines.append(f"- Missing metrics: `{', '.join(unknown)}`")

    return "\n".join(lines) + "\n"


def print_summary(summary: dict[str, Any], provider_name: str, model_name: str) -> None:
    print("\n=== Prompt-Mode BFCL Benchmark Summary ===")
    print(f"provider: {provider_name}")
    print(f"model: {model_name}")
    print(f"categories: {', '.join(summary['categories'])}")
    print(f"cases_per_category: {summary['cases_per_category']}")
    print("\nmetric | baseline | ralph | delta(pp)")
    print("-" * 52)
    for metric, values in summary["metrics_percent_point"].items():
        baseline = values["baseline"]
        ralph = values["ralph"]
        delta = values["delta"]
        baseline_txt = "N/A" if baseline is None else f"{baseline:.2f}"
        ralph_txt = "N/A" if ralph is None else f"{ralph:.2f}"
        delta_txt = "N/A" if delta is None else f"{delta:+.2f}"
        print(f"{metric:22} | {baseline_txt:>8} | {ralph_txt:>6} | {delta_txt:>8}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    cli_path = resolve_cli_path(args.cli_path)

    SHARED["validate_bfcl_root"](args.bfcl_root)
    categories = parse_categories(args.categories)
    if not categories:
        raise SystemExit("No categories provided.")

    run_ids_enabled = args.cases_per_category > 0
    error_report_json_path = resolve_error_report_json_path(
        runtime_root=args.runtime_root,
        error_report_json=args.error_report_json,
    )

    if args.preflight_only:
        run_preflight_checks(
            bfcl_root=args.bfcl_root,
            runtime_root=args.runtime_root,
            provider_name=args.provider_name,
            model_name=args.model_name,
            categories=categories,
            cases_per_category=args.cases_per_category,
            allow_agentic_run_ids=args.allow_agentic_run_ids,
            cli_path=cli_path,
            skip_login_check=args.skip_login_check,
        )
        return

    if not args.skip_login_check:
        check_kiro_login(cli_path)

    ensure_runtime_layout(
        args.runtime_root,
        cli_path=cli_path,
        kiro_model=args.kiro_model,
        kiro_agent=args.kiro_agent,
        trust_tools=args.trust_tools,
    )
    bootstrap_bfcl_imports(args.bfcl_root, args.runtime_root)
    clear_stale_output_file(error_report_json_path)

    categories = resolve_runtime_categories(
        categories=categories,
        run_ids_enabled=run_ids_enabled,
        allow_agentic_run_ids=args.allow_agentic_run_ids,
    )

    if run_ids_enabled:
        run_ids_map = create_run_ids_map(categories, args.cases_per_category)
        run_ids_path = write_run_ids_file(args.runtime_root, run_ids_map)
        print(f"Wrote run-id file: {run_ids_path}")

    baseline_registry, ralph_registry, baseline_display, ralph_display = register_custom_models(
        model_name=args.model_name,
        cli_path=cli_path,
        kiro_model=args.kiro_model,
        kiro_agent=args.kiro_agent,
        trust_tools=args.trust_tools,
        request_timeout_sec=args.request_timeout_sec,
        provider_name=args.provider_name,
        provider_docs_url=args.provider_docs_url,
        provider_license=args.provider_license,
    )
    print("Registered custom models:", baseline_registry, ralph_registry)

    try:
        run_generation_and_eval(
            baseline_registry=baseline_registry,
            ralph_registry=ralph_registry,
            categories=categories,
            temperature=args.temperature,
            num_threads=args.num_threads,
            include_input_log=args.include_input_log,
            run_ids_enabled=run_ids_enabled,
            runtime_root=args.runtime_root,
            error_report_json=error_report_json_path,
            error_report_top_n=args.error_report_top_n,
            max_step_limit=args.max_step_limit,
        )
    except SystemExit:
        if error_report_json_path is not None and error_report_json_path.exists():
            print(f"Saved error forensic report: {error_report_json_path}")
        raise

    score_csv = args.runtime_root / "score" / "data_overall.csv"
    baseline_row, ralph_row = load_score_rows(score_csv, baseline_display, ralph_display)
    summary = build_summary(
        baseline_row=baseline_row,
        ralph_row=ralph_row,
        categories=categories,
        cases_per_category=args.cases_per_category,
    )

    summary_path = args.runtime_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, provider_name=args.provider_name, model_name=args.model_name)
    print(f"\nSaved summary: {summary_path}")
    print(f"Overall CSV: {score_csv}")

    report_markdown_path = resolve_report_markdown_path(
        runtime_root=args.runtime_root,
        report_markdown=args.report_markdown,
    )
    if report_markdown_path is not None:
        report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_report = build_markdown_report(
            summary=summary,
            provider_name=args.provider_name,
            model_name=args.model_name,
            baseline_display=baseline_display,
            ralph_display=ralph_display,
            runtime_root=args.runtime_root.resolve(),
            run_ids_enabled=run_ids_enabled,
        )
        report_markdown_path.write_text(markdown_report, encoding="utf-8")
        print(f"Saved markdown report: {report_markdown_path}")

    if error_report_json_path is not None:
        print(f"Saved error forensic report: {error_report_json_path}")


if __name__ == "__main__":
    main()
