#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
import threading
import urllib.error
import urllib.request
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
resolve_categories_for_run_ids = SHARED["resolve_categories_for_run_ids"]


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
            "for an OpenAI-compatible model."
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
        help="Output/runtime root for BFCL result/, score/, .env, run-id file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=env_or_none("OPENAI_COMPATIBLE_MODEL"),
        help="Prompt-mode model ID. Defaults to OPENAI_COMPATIBLE_MODEL.",
    )
    parser.add_argument(
        "--provider-name",
        type=str,
        default=env_or_none("OPENAI_COMPATIBLE_PROVIDER") or "OpenAI-Compatible",
        help="Provider label shown in BFCL score rows and reports.",
    )
    parser.add_argument(
        "--provider-docs-url",
        type=str,
        default=env_or_none("OPENAI_COMPATIBLE_DOCS_URL"),
        help="Optional provider/model documentation URL used in BFCL metadata.",
    )
    parser.add_argument(
        "--provider-license",
        type=str,
        default=env_or_none("OPENAI_COMPATIBLE_PROVIDER_LICENSE") or "Unknown",
        help="License label recorded in BFCL metadata.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Optional. If omitted, reads OPENAI_COMPATIBLE_API_KEY first, "
            "then OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=env_or_none("OPENAI_COMPATIBLE_BASE_URL", "OPENAI_BASE_URL"),
        help=(
            "OpenAI-compatible base URL. Defaults to OPENAI_COMPATIBLE_BASE_URL "
            "or OPENAI_BASE_URL."
        ),
    )
    parser.add_argument(
        "--default-headers-json",
        type=str,
        default=env_or_none(
            "OPENAI_COMPATIBLE_DEFAULT_HEADERS",
            "OPENAI_DEFAULT_HEADERS",
        ),
        help="Optional JSON object passed to OPENAI_DEFAULT_HEADERS.",
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
        help="Sampling temperature for both runs.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Parallel API calls per model.",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip the generic GET /models accessibility check.",
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
        help="Run environment/category/model diagnostics and exit.",
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
        help="Per-request timeout (seconds) for OpenAI-compatible chat.completions calls.",
    )
    parser.add_argument(
        "--max-step-limit",
        type=int,
        default=20,
        help="Override BFCL MAXIMUM_STEP_LIMIT to cap per-turn tool-call loops.",
    )
    return parser.parse_args()


def validate_json_object(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise SystemExit(f"--default-headers-json must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--default-headers-json must decode to a JSON object.")
    return json.dumps(payload, separators=(",", ":"))


def validate_args(args: argparse.Namespace) -> None:
    if args.cases_per_category < 0:
        raise SystemExit("--cases-per-category must be >= 0.")
    if args.num_threads <= 0:
        raise SystemExit("--num-threads must be >= 1.")
    if args.temperature < 0:
        raise SystemExit("--temperature must be >= 0.")
    if not args.model_name or not args.model_name.strip():
        raise SystemExit(
            "--model-name is required. Pass it explicitly or set OPENAI_COMPATIBLE_MODEL."
        )
    if not args.base_url or not args.base_url.strip():
        raise SystemExit(
            "--base-url is required. Pass it explicitly or set "
            "OPENAI_COMPATIBLE_BASE_URL / OPENAI_BASE_URL."
        )
    if not args.provider_name or not args.provider_name.strip():
        raise SystemExit("--provider-name must not be blank.")
    if args.error_report_top_n <= 0:
        raise SystemExit("--error-report-top-n must be >= 1.")
    if args.request_timeout_sec <= 0:
        raise SystemExit("--request-timeout-sec must be > 0.")
    if args.max_step_limit <= 0:
        raise SystemExit("--max-step-limit must be >= 1.")


def require_api_key(cli_value: str | None) -> str:
    key = (
        cli_value
        if cli_value is not None and cli_value.strip()
        else env_or_none("OPENAI_COMPATIBLE_API_KEY", "OPENAI_API_KEY")
    )
    if not key:
        raise SystemExit(
            "API key is required. Set OPENAI_COMPATIBLE_API_KEY / OPENAI_API_KEY "
            "or pass --api-key."
        )
    return key


def require_base_url(cli_value: str | None) -> str:
    base_url = cli_value.strip() if isinstance(cli_value, str) else ""
    if not base_url:
        raise SystemExit(
            "Base URL is required. Set OPENAI_COMPATIBLE_BASE_URL / OPENAI_BASE_URL "
            "or pass --base-url."
        )
    return base_url.rstrip("/")


def build_request_headers(api_key: str, default_headers_json: str | None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if default_headers_json:
        headers.update(json.loads(default_headers_json))
    return headers


def check_model_access(
    api_key: str,
    base_url: str,
    model_name: str,
    default_headers_json: str | None,
) -> None:
    req = urllib.request.Request(
        f"{base_url}/models",
        headers=build_request_headers(api_key, default_headers_json),
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Unexpected status from model API: {resp.status}")
            payload_text = resp.read().decode("utf-8", "ignore")
            try:
                payload = json.loads(payload_text)
            except Exception:
                payload = None

            if isinstance(payload, dict):
                data = payload.get("data")
                if isinstance(data, list):
                    visible_model_ids = {
                        item.get("id")
                        for item in data
                        if isinstance(item, dict) and isinstance(item.get("id"), str)
                    }
                    if visible_model_ids and model_name not in visible_model_ids:
                        visible = ", ".join(sorted(visible_model_ids)[:20])
                        raise SystemExit(
                            f"Model '{model_name}' is not visible from {base_url}/models. "
                            f"Visible IDs: {visible}"
                        )
    except urllib.error.HTTPError as exc:
        body = exc.read(300).decode("utf-8", "ignore")
        raise SystemExit(
            f"Model API precheck failed: HTTP {exc.code}. Body: {body}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Model API precheck failed: {exc}") from exc


def ensure_runtime_layout(
    runtime_root: Path,
    api_key: str,
    base_url: str,
    default_headers_json: str | None,
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
        f"OPENAI_API_KEY={api_key}",
        f"OPENAI_BASE_URL={base_url}",
        f"OPENAI_DEFAULT_HEADERS={default_headers_json or ''}",
        "SERP_API_KEY=",
        "LOCAL_SERVER_ENDPOINT=localhost",
        "LOCAL_SERVER_PORT=1053",
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
    skip_model_check: bool,
    api_key: str,
    base_url: str,
    default_headers_json: str | None,
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

    if skip_model_check:
        add("WARN", f"Skipped {provider_name} GET /models check (--skip-model-check)")
    else:
        check_model_access(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            default_headers_json=default_headers_json,
        )
        add("PASS", f"{provider_name} model visibility check passed")

    print("\n=== Preflight Report ===")
    for status, message in checks:
        print(f"[{status}] {message}")
    print(f"[INFO] Provider: {provider_name}")
    print(f"[INFO] Base URL: {base_url}")
    print(f"[INFO] Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("Preflight completed successfully.")


def register_custom_models(
    *,
    model_name: str,
    request_timeout_sec: float,
    provider_name: str,
    provider_docs_url: str | None,
    provider_license: str,
) -> tuple[str, str, str, str]:
    try:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
        from bfcl_eval.model_handler.api_inference.openai_completion import (
            OpenAICompletionsHandler,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(format_missing_dependency_error(exc)) from exc

    class OpenAICompatibleTimeoutPromptHandler(OpenAICompletionsHandler):
        def generate_with_backoff(self, **kwargs):
            kwargs.setdefault("timeout", request_timeout_sec)

            result_box: dict[str, Any] = {}
            error_box: dict[str, Exception] = {}
            done = threading.Event()

            def _call_parent() -> None:
                try:
                    result_box["result"] = super(
                        OpenAICompatibleTimeoutPromptHandler, self
                    ).generate_with_backoff(**kwargs)
                except Exception as exc:
                    error_box["error"] = exc
                finally:
                    done.set()

            worker = threading.Thread(target=_call_parent, daemon=True)
            worker.start()

            hard_wait = max(request_timeout_sec + 5.0, 10.0)
            if not done.wait(hard_wait):
                raise TimeoutError(
                    "OpenAI-compatible request exceeded hard timeout "
                    f"({request_timeout_sec:.1f}s)."
                )

            if "error" in error_box:
                raise error_box["error"]

            return result_box["result"]

    class OpenAICompatibleRalphLoopPromptHandler(OpenAICompatibleTimeoutPromptHandler):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._thread_local = threading.local()

        def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
            inference_data = super()._pre_query_processing_prompting(test_entry)

            first_turn = test_entry["question"][0]
            if first_turn and first_turn[0].get("role") == "system":
                first_turn[0]["content"] = (
                    f"{first_turn[0]['content']}\n\n{RALPH_SYSTEM_PROMPT_SUFFIX}"
                )
            else:
                first_turn.insert(
                    0,
                    {"role": "system", "content": RALPH_SYSTEM_PROMPT_SUFFIX},
                )
            return inference_data

        def _query_prompting(self, inference_data: dict):
            def _response_text(resp: Any) -> str:
                try:
                    txt = resp.choices[0].message.content
                except Exception:
                    return ""
                if isinstance(txt, str):
                    return txt.strip()
                return str(txt or "").strip()

            base_messages = deepcopy(inference_data["message"])

            analysis_messages = deepcopy(base_messages)
            analysis_messages.append({"role": "system", "content": RALPH_PREFLIGHT_PROMPT})
            analysis_response, analysis_latency = self.generate_with_backoff(
                messages=analysis_messages,
                model=self.model_name,
                temperature=self.temperature,
                store=False,
            )

            analysis_text = _response_text(analysis_response)
            self._thread_local.last_ralph_analysis = analysis_text

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

            inference_data["inference_input_log"] = {
                "base_message": repr(base_messages),
                "ralph_analysis_prompt": RALPH_PREFLIGHT_PROMPT,
                "ralph_final_prompt": RALPH_FINAL_PROMPT,
            }

            final_response, final_latency = self.generate_with_backoff(
                messages=final_messages,
                model=self.model_name,
                temperature=self.temperature,
                store=False,
            )
            total_latency = analysis_latency + final_latency

            final_text = _response_text(final_response).lower()
            looks_empty_or_refusal = (
                final_text in {"", "[]"}
                or "cannot comply" in final_text
                or "i cannot" in final_text
                or "i'm sorry" in final_text
            )
            if looks_empty_or_refusal:
                fallback_response, fallback_latency = self.generate_with_backoff(
                    messages=base_messages,
                    model=self.model_name,
                    temperature=self.temperature,
                    store=False,
                )
                total_latency += fallback_latency
                return fallback_response, total_latency

            return final_response, total_latency

        def _parse_query_response_prompting(self, api_response: Any) -> dict:
            response_data = super()._parse_query_response_prompting(api_response)
            last_ralph_analysis = getattr(self._thread_local, "last_ralph_analysis", "")
            if last_ralph_analysis:
                response_data["ralph_analysis"] = last_ralph_analysis
            self._thread_local.last_ralph_analysis = ""
            return response_data

    baseline_registry = f"{model_name}-prompt-baseline"
    ralph_registry = f"{model_name}-prompt-ralph-loop"
    baseline_display = f"{model_name} (Prompt Baseline)"
    ralph_display = f"{model_name} (Prompt + RALPH Loop)"

    model_url = provider_docs_url or os.getenv("OPENAI_BASE_URL") or "Unknown"

    MODEL_CONFIG_MAPPING[baseline_registry] = ModelConfig(
        model_name=model_name,
        display_name=baseline_display,
        url=model_url,
        org=provider_name,
        license=provider_license,
        model_handler=OpenAICompatibleTimeoutPromptHandler,
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
        model_handler=OpenAICompatibleRalphLoopPromptHandler,
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
        "# OpenAI-Compatible Prompt-Mode BFCL Benchmark Report",
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
    default_headers_json = validate_json_object(args.default_headers_json)
    api_key = require_api_key(args.api_key)
    base_url = require_base_url(args.base_url)

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
            skip_model_check=args.skip_model_check,
            api_key=api_key,
            base_url=base_url,
            default_headers_json=default_headers_json,
        )
        return

    if not args.skip_model_check:
        check_model_access(
            api_key=api_key,
            base_url=base_url,
            model_name=args.model_name,
            default_headers_json=default_headers_json,
        )

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    if default_headers_json:
        os.environ["OPENAI_DEFAULT_HEADERS"] = default_headers_json
    else:
        os.environ.pop("OPENAI_DEFAULT_HEADERS", None)

    ensure_runtime_layout(args.runtime_root, api_key, base_url, default_headers_json)
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
