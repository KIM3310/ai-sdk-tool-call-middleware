#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GROK_RUNNER = (
    REPO_ROOT
    / "experiments"
    / "grok-prompt-bfcl-ralph"
    / "run_grok_prompt_bfcl_ralph.py"
)
OPENAI_COMPATIBLE_RUNNER = (
    REPO_ROOT
    / "experiments"
    / "openai-compatible-prompt-bfcl-ralph"
    / "run_openai_compatible_prompt_bfcl_ralph.py"
)
KIRO_CLI_RUNNER = (
    REPO_ROOT
    / "experiments"
    / "kiro-cli-prompt-bfcl-ralph"
    / "run_kiro_cli_prompt_bfcl_ralph.py"
)


DEFAULT_CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
]

DEFAULT_REQUEST_TIMEOUT_SEC = 120.0
DEFAULT_CHILD_POLL_INTERVAL_SEC = 5.0
DEFAULT_SALVAGE_STALL_SEC = 120.0
_RUNNER_MODULE_CACHE: dict[str, dict[str, Any]] = {}
TRACKED_METRICS = [
    "Overall Acc",
    "Non-Live AST Acc",
    "Live Acc",
    "Multi Turn Acc",
    "Relevance Detection",
    "Irrelevance Detection",
]


@dataclass
class ChildRunOutcome:
    returncode: int
    salvaged: bool = False
    salvage_note: str | None = None
    failure_note: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a large prompt-mode BFCL RALPH matrix across Grok and "
            "OpenAI-compatible models, then classify improved/regressed results."
        )
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=Path(__file__).resolve().parent / "models.example.json",
        help="JSON file containing the model matrix configuration.",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runtime",
        help="Root directory for per-model runs and aggregate reports.",
    )
    parser.add_argument(
        "--bfcl-root",
        type=Path,
        default=Path("/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard"),
        help="Path to BFCL root directory.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python interpreter used for child benchmark runs.",
    )
    parser.add_argument(
        "--model-ids",
        type=str,
        default=None,
        help="Optional comma-separated subset of model ids to run.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated BFCL categories.",
    )
    parser.add_argument(
        "--cases-per-category",
        type=int,
        default=3,
        help="If > 0, evaluate first N test IDs per category.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="Sampling temperature for child runs.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Parallel API calls per child model run.",
    )
    parser.add_argument(
        "--max-step-limit",
        type=int,
        default=20,
        help="BFCL maximum step limit override.",
    )
    parser.add_argument(
        "--flat-threshold-pp",
        type=float,
        default=0.0,
        help="Treat deltas within this absolute percent-point threshold as flat.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run each configured model in preflight mode only.",
    )
    parser.add_argument(
        "--include-input-log",
        action="store_true",
        help="Enable BFCL verbose inference input logs.",
    )
    parser.add_argument(
        "--allow-agentic-run-ids",
        action="store_true",
        help="Allow memory/web-search categories when run-id sampling is enabled.",
    )
    parser.add_argument(
        "--skip-model-checks",
        action="store_true",
        help="Skip provider /models prechecks in child runners.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the matrix on the first failed model run.",
    )
    parser.add_argument(
        "--child-poll-interval-sec",
        type=float,
        default=DEFAULT_CHILD_POLL_INTERVAL_SEC,
        help="How often to poll child benchmark processes.",
    )
    parser.add_argument(
        "--salvage-stall-sec",
        type=float,
        default=DEFAULT_SALVAGE_STALL_SEC,
        help=(
            "If all expected result files are present but the child process still "
            "has not finished after this many idle seconds, run eval-only salvage."
        ),
    )
    parser.add_argument(
        "--child-max-runtime-sec",
        type=float,
        default=0.0,
        help=(
            "Optional hard cap for each child benchmark runtime. "
            "Set to 0 to disable."
        ),
    )
    return parser.parse_args()


def parse_categories(raw: str) -> list[str]:
    categories = [item.strip() for item in raw.split(",") if item.strip()]
    if not categories:
        raise SystemExit("No categories provided.")
    return categories


def load_models_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"--models-file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--models-file must contain a JSON object.")
    models = payload.get("models")
    if not isinstance(models, list):
        raise SystemExit("--models-file must contain a top-level 'models' array.")
    return models


def require_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise SystemExit(f"Field '{field_name}' must be a boolean.")


def validate_model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise SystemExit("Each model entry must be a JSON object.")

    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise SystemExit("Each model entry needs a non-empty string 'id'.")

    kind = entry.get("kind")
    if kind not in {"grok", "openai-compatible", "kiro-cli"}:
        raise SystemExit(
            f"Model '{model_id}' has invalid kind '{kind}'. "
            "Use 'grok', 'openai-compatible', or 'kiro-cli'."
        )

    model_name = entry.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise SystemExit(f"Model '{model_id}' needs a non-empty 'model_name'.")

    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise SystemExit(f"Model '{model_id}' field 'enabled' must be boolean.")

    if kind == "openai-compatible":
        provider_name = entry.get("provider_name")
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise SystemExit(
                f"OpenAI-compatible model '{model_id}' needs 'provider_name'."
            )
        base_url = entry.get("base_url")
        base_url_env = entry.get("base_url_env")
        if not isinstance(base_url, str) and not isinstance(base_url_env, str):
            raise SystemExit(
                f"OpenAI-compatible model '{model_id}' needs 'base_url' or 'base_url_env'."
            )

    if kind == "kiro-cli":
        provider_name = entry.get("provider_name")
        if provider_name is not None and (
            not isinstance(provider_name, str) or not provider_name.strip()
        ):
            raise SystemExit(
                f"Kiro CLI model '{model_id}' field 'provider_name' must be a non-empty string."
            )

    return entry


def select_models(
    models: list[dict[str, Any]], requested_ids: set[str] | None
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_entry in models:
        entry = validate_model_entry(raw_entry)
        model_id = entry["id"]
        if model_id in seen_ids:
            raise SystemExit(f"Duplicate model id in config: {model_id}")
        seen_ids.add(model_id)

        if not entry.get("enabled", True):
            continue
        if requested_ids is not None and model_id not in requested_ids:
            continue
        selected.append(entry)

    if requested_ids is not None:
        missing = sorted(requested_ids - seen_ids)
        if missing:
            raise SystemExit(f"Unknown --model-ids requested: {', '.join(missing)}")

    if not selected:
        raise SystemExit("No enabled models selected for this matrix run.")

    return selected


def env_value_from_entry(entry: dict[str, Any], field: str) -> str | None:
    direct = entry.get(field)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    env_name = entry.get(f"{field}_env")
    if isinstance(env_name, str) and env_name.strip():
        value = os.getenv(env_name)
        if value is not None and value.strip():
            return value.strip()
    return None


def build_child_env(entry: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    child_env = os.environ.copy()
    missing: list[str] = []
    kind = entry["kind"]

    if kind == "grok":
        api_key = env_value_from_entry(entry, "api_key")
        env_name = entry.get("api_key_env", "GROK_API_KEY")
        if not api_key:
            missing.append(str(env_name))
        else:
            child_env["GROK_API_KEY"] = api_key
            child_env["OPENAI_API_KEY"] = api_key
            child_env["XAI_API_KEY"] = api_key
        return child_env, missing

    if kind == "kiro-cli":
        return child_env, missing

    api_key = env_value_from_entry(entry, "api_key")
    env_name = entry.get("api_key_env", "OPENAI_COMPATIBLE_API_KEY")
    if not api_key:
        missing.append(str(env_name))
    else:
        child_env["OPENAI_COMPATIBLE_API_KEY"] = api_key
        child_env["OPENAI_API_KEY"] = api_key

    base_url = env_value_from_entry(entry, "base_url")
    if not base_url:
        missing.append(str(entry.get("base_url_env", "base_url")))

    headers = env_value_from_entry(entry, "default_headers_json")
    if headers:
        child_env["OPENAI_COMPATIBLE_DEFAULT_HEADERS"] = headers
        child_env["OPENAI_DEFAULT_HEADERS"] = headers

    return child_env, missing


def load_runner_module(kind: str) -> dict[str, Any]:
    cached = _RUNNER_MODULE_CACHE.get(kind)
    if cached is not None:
        return cached

    if kind == "grok":
        runner_path = GROK_RUNNER
    elif kind == "openai-compatible":
        runner_path = OPENAI_COMPATIBLE_RUNNER
    elif kind == "kiro-cli":
        runner_path = KIRO_CLI_RUNNER
    else:  # pragma: no cover - validate_model_entry rejects this
        raise SystemExit(f"Unsupported model kind for runner lookup: {kind}")

    module = runpy.run_path(
        str(runner_path),
        run_name=f"prompt_bfcl_matrix_{kind.replace('-', '_')}",
    )
    _RUNNER_MODULE_CACHE[kind] = module
    return module


def resolve_registry_names(
    model_name: str,
) -> tuple[str, str, str, str]:
    baseline_registry = f"{model_name}-prompt-baseline"
    ralph_registry = f"{model_name}-prompt-ralph-loop"
    baseline_display = f"{model_name} (Prompt Baseline)"
    ralph_display = f"{model_name} (Prompt + RALPH Loop)"
    return baseline_registry, ralph_registry, baseline_display, ralph_display


def resolve_effective_categories(args: argparse.Namespace) -> list[str]:
    shared = load_runner_module("grok")
    return shared["resolve_runtime_categories"](
        categories=parse_categories(args.categories),
        run_ids_enabled=args.cases_per_category > 0,
        allow_agentic_run_ids=args.allow_agentic_run_ids,
    )


def build_expected_result_paths(
    *,
    entry: dict[str, Any],
    args: argparse.Namespace,
    runtime_root: Path,
) -> list[Path]:
    shared = load_runner_module("grok")
    shared["bootstrap_bfcl_imports"](args.bfcl_root, runtime_root)

    from bfcl_eval.utils import (
        get_directory_structure_by_category,
        get_file_name_by_category,
    )

    baseline_registry, ralph_registry, _baseline_display, _ralph_display = (
        resolve_registry_names(entry["model_name"])
    )
    expected_paths: list[Path] = []
    categories = resolve_effective_categories(args)

    for registry in [baseline_registry, ralph_registry]:
        model_root = runtime_root / "result" / registry.replace("/", "_")
        for category in categories:
            expected_paths.append(
                model_root
                / Path(get_directory_structure_by_category(category))
                / get_file_name_by_category(category, is_result_file=True)
            )

    return expected_paths


def latest_existing_mtime(paths: list[Path]) -> float | None:
    existing = [path.stat().st_mtime for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing)


def should_attempt_eval_only_salvage(
    *,
    expected_result_paths: list[Path],
    summary_path: Path,
    stall_sec: float,
    now: float | None = None,
) -> bool:
    if summary_path.exists():
        return False
    if not expected_result_paths:
        return False
    if any(not path.exists() for path in expected_result_paths):
        return False

    latest_mtime = latest_existing_mtime(expected_result_paths)
    if latest_mtime is None:
        return False

    check_now = time.time() if now is None else now
    return (check_now - latest_mtime) >= stall_sec


def clear_runtime_log_files(runtime_root: Path) -> tuple[Path, Path]:
    stdout_path = runtime_root / "stdout.log"
    stderr_path = runtime_root / "stderr.log"
    for path in [stdout_path, stderr_path]:
        if path.exists():
            path.unlink()
    return stdout_path, stderr_path


def read_log_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def clear_score_outputs(runtime_root: Path, *, remove_nested: bool = False) -> None:
    score_root = runtime_root / "score"
    score_root.mkdir(parents=True, exist_ok=True)
    for stale_csv in score_root.glob("*.csv"):
        stale_csv.unlink()
    if remove_nested:
        for stale_path in score_root.iterdir():
            if stale_path.is_dir():
                shutil.rmtree(stale_path)


def terminate_process(proc: subprocess.Popen[str], timeout_sec: float = 5.0) -> None:
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout_sec)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def register_custom_models_for_entry(
    runner_module: dict[str, Any],
    entry: dict[str, Any],
) -> tuple[str, str, str, str]:
    request_timeout_sec = float(
        entry.get("request_timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC)
    )
    if entry["kind"] == "grok":
        return runner_module["register_custom_models"](
            model_name=entry["model_name"],
            request_timeout_sec=request_timeout_sec,
        )

    if entry["kind"] == "openai-compatible":
        return runner_module["register_custom_models"](
            model_name=entry["model_name"],
            request_timeout_sec=request_timeout_sec,
            provider_name=entry["provider_name"],
            provider_docs_url=entry.get("provider_docs_url"),
            provider_license=entry.get("provider_license", "Unknown"),
        )

    if entry["kind"] == "kiro-cli":
        cli_path = env_value_from_entry(entry, "cli_path") or "kiro-cli"
        trust_tools_raw = entry.get("trust_tools")
        trust_tools = trust_tools_raw if isinstance(trust_tools_raw, str) else ""
        return runner_module["register_custom_models"](
            model_name=entry["model_name"],
            cli_path=cli_path,
            kiro_model=env_value_from_entry(entry, "kiro_model"),
            kiro_agent=env_value_from_entry(entry, "kiro_agent"),
            trust_tools=trust_tools,
            request_timeout_sec=request_timeout_sec,
            provider_name=entry.get("provider_name", "Kiro CLI"),
            provider_docs_url=entry.get("provider_docs_url"),
            provider_license=entry.get("provider_license", "Unknown"),
        )

    raise SystemExit(f"Unsupported model kind for salvage registration: {entry['kind']}")


def build_salvaged_markdown_report(
    *,
    runner_module: dict[str, Any],
    entry: dict[str, Any],
    summary: dict[str, Any],
    baseline_display: str,
    ralph_display: str,
    runtime_root: Path,
    run_ids_enabled: bool,
) -> str:
    if entry["kind"] == "grok":
        return runner_module["build_markdown_report"](
            summary=summary,
            model_name=entry["model_name"],
            baseline_display=baseline_display,
            ralph_display=ralph_display,
            runtime_root=runtime_root.resolve(),
            run_ids_enabled=run_ids_enabled,
        )

    return runner_module["build_markdown_report"](
        summary=summary,
        provider_name=entry.get("provider_name", "Unknown"),
        model_name=entry["model_name"],
        baseline_display=baseline_display,
        ralph_display=ralph_display,
        runtime_root=runtime_root.resolve(),
        run_ids_enabled=run_ids_enabled,
    )


def load_bfcl_scoring_helpers() -> dict[str, Any]:
    from bfcl_eval.eval_checker.eval_runner_helper import (  # type: ignore
        calculate_percentage_weighted_accuracy,
        calculate_unweighted_accuracy,
        calculate_weighted_accuracy,
        get_category_score,
    )

    return {
        "calculate_percentage_weighted_accuracy": calculate_percentage_weighted_accuracy,
        "calculate_unweighted_accuracy": calculate_unweighted_accuracy,
        "calculate_weighted_accuracy": calculate_weighted_accuracy,
        "get_category_score": get_category_score,
    }


def category_name_from_score_path(path: Path) -> str | None:
    prefix = "BFCL_v4_"
    suffix = "_score.json"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def load_registry_category_scores(
    *, runtime_root: Path, registry_name: str
) -> dict[str, dict[str, Any]]:
    score_root = runtime_root / "score" / registry_name
    if not score_root.exists():
        return {}

    scores: dict[str, dict[str, Any]] = {}
    for path in score_root.rglob("BFCL_v4_*_score.json"):
        category_name = category_name_from_score_path(path)
        if category_name is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        accuracy = payload.get("accuracy")
        correct_count = payload.get("correct_count")
        total_count = payload.get("total_count")
        if not isinstance(accuracy, (int, float)):
            continue
        if not isinstance(correct_count, int) or not isinstance(total_count, int):
            continue
        scores[category_name] = {
            "accuracy": float(accuracy),
            "correct_count": correct_count,
            "total_count": total_count,
        }
    return scores


def to_percent_points(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value) * 100, 2)
    return None


def compute_tracked_metrics_from_category_scores(
    *,
    category_scores: dict[str, dict[str, Any]],
    helpers: dict[str, Any],
) -> dict[str, float | None]:
    get_category_score = helpers["get_category_score"]
    calculate_unweighted_accuracy = helpers["calculate_unweighted_accuracy"]
    calculate_weighted_accuracy = helpers["calculate_weighted_accuracy"]
    calculate_percentage_weighted_accuracy = helpers[
        "calculate_percentage_weighted_accuracy"
    ]

    python_simple_ast_non_live = get_category_score(category_scores, "simple_python")
    python_multiple_ast_non_live = get_category_score(category_scores, "multiple")
    python_parallel_ast_non_live = get_category_score(category_scores, "parallel")
    python_parallel_multiple_ast_non_live = get_category_score(
        category_scores, "parallel_multiple"
    )
    java_simple_ast_non_live = get_category_score(category_scores, "simple_java")
    javascript_simple_ast_non_live = get_category_score(
        category_scores, "simple_javascript"
    )
    irrelevance_non_live = get_category_score(category_scores, "irrelevance")

    simple_ast_non_live = calculate_unweighted_accuracy(
        [
            python_simple_ast_non_live,
            java_simple_ast_non_live,
            javascript_simple_ast_non_live,
        ]
    )
    multiple_ast_non_live = python_multiple_ast_non_live
    parallel_ast_non_live = python_parallel_ast_non_live
    parallel_multiple_ast_non_live = python_parallel_multiple_ast_non_live

    summary_ast_non_live = calculate_unweighted_accuracy(
        [
            simple_ast_non_live,
            multiple_ast_non_live,
            parallel_ast_non_live,
            parallel_multiple_ast_non_live,
        ]
    )
    overall_accuracy_non_live = calculate_unweighted_accuracy(
        [
            simple_ast_non_live,
            multiple_ast_non_live,
            parallel_ast_non_live,
            parallel_multiple_ast_non_live,
        ],
        display_na_if_category_missing=False,
    )

    python_simple_ast_live = get_category_score(category_scores, "live_simple")
    python_multiple_ast_live = get_category_score(category_scores, "live_multiple")
    python_parallel_ast_live = get_category_score(category_scores, "live_parallel")
    python_parallel_multiple_ast_live = get_category_score(
        category_scores, "live_parallel_multiple"
    )
    irrelevance_live = get_category_score(category_scores, "live_irrelevance")
    relevance_live = get_category_score(category_scores, "live_relevance")

    summary_ast_live = calculate_weighted_accuracy(
        [
            python_simple_ast_live,
            python_multiple_ast_live,
            python_parallel_ast_live,
            python_parallel_multiple_ast_live,
        ]
    )
    overall_accuracy_live = calculate_weighted_accuracy(
        [
            python_simple_ast_live,
            python_multiple_ast_live,
            python_parallel_ast_live,
            python_parallel_multiple_ast_live,
        ],
        display_na_if_category_missing=False,
    )

    multi_turn_base = get_category_score(category_scores, "multi_turn_base")
    multi_turn_miss_func = get_category_score(category_scores, "multi_turn_miss_func")
    multi_turn_miss_param = get_category_score(category_scores, "multi_turn_miss_param")
    multi_turn_long_context = get_category_score(
        category_scores, "multi_turn_long_context"
    )
    overall_accuracy_multi_turn = calculate_unweighted_accuracy(
        [
            multi_turn_base,
            multi_turn_miss_func,
            multi_turn_miss_param,
            multi_turn_long_context,
        ],
        display_na_if_category_missing=False,
    )

    web_search_base = get_category_score(category_scores, "web_search_base")
    web_search_no_snippet = get_category_score(category_scores, "web_search_no_snippet")
    summary_web_search = calculate_unweighted_accuracy(
        [
            web_search_base,
            web_search_no_snippet,
        ]
    )
    memory_kv = get_category_score(category_scores, "memory_kv")
    memory_vector = get_category_score(category_scores, "memory_vector")
    memory_rec_sum = get_category_score(category_scores, "memory_rec_sum")
    summary_memory = calculate_unweighted_accuracy(
        [
            memory_kv,
            memory_vector,
            memory_rec_sum,
        ]
    )
    overall_accuracy_agentic = calculate_unweighted_accuracy(
        [
            summary_web_search,
            summary_memory,
        ],
        display_na_if_category_missing=False,
    )

    total_irrelevance = calculate_unweighted_accuracy(
        [irrelevance_non_live, irrelevance_live]
    )
    total_relevance = relevance_live

    total_overall_accuracy = calculate_percentage_weighted_accuracy(
        [
            overall_accuracy_non_live,
            overall_accuracy_live,
            total_irrelevance,
            overall_accuracy_multi_turn,
            overall_accuracy_agentic,
        ],
        [10, 10, 10, 30, 40],
        display_na_if_category_missing=False,
    )

    return {
        "Overall Acc": to_percent_points(total_overall_accuracy["display_accuracy"]),
        "Non-Live AST Acc": to_percent_points(summary_ast_non_live["display_accuracy"]),
        "Live Acc": to_percent_points(overall_accuracy_live["display_accuracy"]),
        "Multi Turn Acc": to_percent_points(
            overall_accuracy_multi_turn["display_accuracy"]
        ),
        "Relevance Detection": to_percent_points(total_relevance["display_accuracy"]),
        "Irrelevance Detection": to_percent_points(
            total_irrelevance["display_accuracy"]
        ),
    }


def build_summary_from_category_score_dicts(
    *,
    baseline_scores: dict[str, dict[str, Any]],
    ralph_scores: dict[str, dict[str, Any]],
    categories: list[str],
    cases_per_category: int,
    helpers: dict[str, Any],
) -> dict[str, Any] | None:
    if not baseline_scores or not ralph_scores:
        return None

    baseline_metrics = compute_tracked_metrics_from_category_scores(
        category_scores=baseline_scores,
        helpers=helpers,
    )
    ralph_metrics = compute_tracked_metrics_from_category_scores(
        category_scores=ralph_scores,
        helpers=helpers,
    )

    metrics: dict[str, dict[str, float | None]] = {}
    for metric in TRACKED_METRICS:
        baseline_value = baseline_metrics.get(metric)
        ralph_value = ralph_metrics.get(metric)
        delta = None
        if baseline_value is not None and ralph_value is not None:
            delta = round(ralph_value - baseline_value, 4)
        metrics[metric] = {
            "baseline": baseline_value,
            "ralph": ralph_value,
            "delta": delta,
        }

    return {
        "categories": categories,
        "cases_per_category": cases_per_category,
        "metrics_percent_point": metrics,
    }


def attempt_score_json_summary_salvage(
    *,
    baseline_registry: str,
    ralph_registry: str,
    runtime_root: Path,
    categories: list[str],
    cases_per_category: int,
) -> dict[str, Any] | None:
    helpers = load_bfcl_scoring_helpers()
    baseline_scores = load_registry_category_scores(
        runtime_root=runtime_root,
        registry_name=baseline_registry,
    )
    ralph_scores = load_registry_category_scores(
        runtime_root=runtime_root,
        registry_name=ralph_registry,
    )
    return build_summary_from_category_score_dicts(
        baseline_scores=baseline_scores,
        ralph_scores=ralph_scores,
        categories=categories,
        cases_per_category=cases_per_category,
        helpers=helpers,
    )


def attempt_eval_only_salvage(
    *,
    entry: dict[str, Any],
    args: argparse.Namespace,
    runtime_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    if args.preflight_only:
        return None, "eval-only salvage is unavailable in preflight mode"

    runner_module = load_runner_module(entry["kind"])
    runner_module["bootstrap_bfcl_imports"](args.bfcl_root, runtime_root)

    try:
        from dotenv import load_dotenv
        from bfcl_eval.constants.eval_config import DOTENV_PATH
        from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on BFCL install
        return None, f"eval-only salvage failed: {exc}"

    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    categories = resolve_effective_categories(args)
    baseline_registry, ralph_registry, baseline_display, ralph_display = (
        register_custom_models_for_entry(runner_module, entry)
    )
    clear_score_outputs(runtime_root)

    try:
        evaluation_main(
            [baseline_registry, ralph_registry],
            categories,
            result_dir=None,
            score_dir=None,
            partial_eval=args.cases_per_category > 0,
        )

        score_csv = runtime_root / "score" / "data_overall.csv"
        baseline_row, ralph_row = runner_module["load_score_rows"](
            score_csv,
            baseline_display,
            ralph_display,
        )
        summary = runner_module["build_summary"](
            baseline_row=baseline_row,
            ralph_row=ralph_row,
            categories=categories,
            cases_per_category=args.cases_per_category,
        )
        (runtime_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_markdown = build_salvaged_markdown_report(
            runner_module=runner_module,
            entry=entry,
            summary=summary,
            baseline_display=baseline_display,
            ralph_display=ralph_display,
            runtime_root=runtime_root,
            run_ids_enabled=args.cases_per_category > 0,
        )
        (runtime_root / "benchmark_report.md").write_text(
            report_markdown,
            encoding="utf-8",
        )
        return summary, None
    except Exception as exc:
        fallback_summary = attempt_score_json_summary_salvage(
            baseline_registry=baseline_registry,
            ralph_registry=ralph_registry,
            runtime_root=runtime_root,
            categories=categories,
            cases_per_category=args.cases_per_category,
        )
        if fallback_summary is None:
            return None, f"eval-only salvage failed: {exc}"

        (runtime_root / "summary.json").write_text(
            json.dumps(fallback_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_markdown = build_salvaged_markdown_report(
            runner_module=runner_module,
            entry=entry,
            summary=fallback_summary,
            baseline_display=baseline_display,
            ralph_display=ralph_display,
            runtime_root=runtime_root,
            run_ids_enabled=args.cases_per_category > 0,
        )
        (runtime_root / "benchmark_report.md").write_text(
            report_markdown,
            encoding="utf-8",
        )
        return fallback_summary, None


def run_child_process_monitored(
    *,
    entry: dict[str, Any],
    command: list[str],
    child_env: dict[str, str],
    args: argparse.Namespace,
    runtime_root: Path,
) -> ChildRunOutcome:
    stdout_path, stderr_path = clear_runtime_log_files(runtime_root)
    expected_result_paths: list[Path] = []
    if not args.preflight_only:
        expected_result_paths = build_expected_result_paths(
            entry=entry,
            args=args,
            runtime_root=runtime_root,
        )

    summary_path = runtime_root / "summary.json"
    started_monotonic = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=child_env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

        while True:
            returncode = proc.poll()
            if returncode is not None:
                break

            if (
                not args.preflight_only
                and args.salvage_stall_sec >= 0
                and should_attempt_eval_only_salvage(
                    expected_result_paths=expected_result_paths,
                    summary_path=summary_path,
                    stall_sec=args.salvage_stall_sec,
                )
            ):
                summary, salvage_error = attempt_eval_only_salvage(
                    entry=entry,
                    args=args,
                    runtime_root=runtime_root,
                )
                if summary is not None:
                    terminate_process(proc)
                    return ChildRunOutcome(
                        returncode=0,
                        salvaged=True,
                        salvage_note=(
                            "Recovered via eval-only salvage after the child runner "
                            "stalled with all expected result files present."
                        ),
                    )
                terminate_process(proc)
                return ChildRunOutcome(
                    returncode=1,
                    failure_note=salvage_error
                    or "eval-only salvage failed after child stall",
                )

            if (
                args.child_max_runtime_sec > 0
                and (time.monotonic() - started_monotonic) >= args.child_max_runtime_sec
            ):
                if not args.preflight_only and expected_result_paths and all(
                    path.exists() for path in expected_result_paths
                ):
                    summary, salvage_error = attempt_eval_only_salvage(
                        entry=entry,
                        args=args,
                        runtime_root=runtime_root,
                    )
                    if summary is not None:
                        terminate_process(proc)
                        return ChildRunOutcome(
                            returncode=0,
                            salvaged=True,
                            salvage_note=(
                                "Recovered via eval-only salvage after the child "
                                "runner exceeded the max runtime."
                            ),
                        )
                    terminate_process(proc)
                    return ChildRunOutcome(
                        returncode=1,
                        failure_note=salvage_error
                        or "eval-only salvage failed after max runtime was exceeded",
                    )

                terminate_process(proc)
                return ChildRunOutcome(
                    returncode=1,
                    failure_note=(
                        "Child benchmark exceeded max runtime "
                        f"({args.child_max_runtime_sec:.1f}s)."
                    ),
                )

            time.sleep(args.child_poll_interval_sec)

    if not args.preflight_only and not summary_path.exists():
        if expected_result_paths and all(path.exists() for path in expected_result_paths):
            summary, salvage_error = attempt_eval_only_salvage(
                entry=entry,
                args=args,
                runtime_root=runtime_root,
            )
            if summary is not None:
                return ChildRunOutcome(
                    returncode=0,
                    salvaged=True,
                    salvage_note=(
                        "Recovered via eval-only salvage after the child runner "
                        "exited without writing summary artifacts."
                    ),
                )
            if salvage_error is not None:
                return ChildRunOutcome(
                    returncode=returncode if returncode != 0 else 1,
                    failure_note=salvage_error,
                )

    return ChildRunOutcome(returncode=returncode)


def build_child_command(
    *,
    entry: dict[str, Any],
    args: argparse.Namespace,
    runtime_root: Path,
) -> list[str]:
    categories = ",".join(parse_categories(args.categories))
    if entry["kind"] == "grok":
        command = [
            args.python_executable,
            str(GROK_RUNNER),
            "--bfcl-root",
            str(args.bfcl_root),
            "--runtime-root",
            str(runtime_root),
            "--model-name",
            entry["model_name"],
            "--categories",
            categories,
            "--cases-per-category",
            str(args.cases_per_category),
            "--temperature",
            str(args.temperature),
            "--num-threads",
            str(args.num_threads),
            "--max-step-limit",
            str(args.max_step_limit),
            "--report-markdown",
            "benchmark_report.md",
            "--error-report-json",
            "error_forensics.json",
        ]
        if args.include_input_log:
            command.append("--include-input-log")
        if args.allow_agentic_run_ids:
            command.append("--allow-agentic-run-ids")
        if args.preflight_only:
            command.append("--preflight-only")
        if args.skip_model_checks:
            command.append("--skip-key-check")
        return command

    if entry["kind"] == "kiro-cli":
        command = [
            args.python_executable,
            str(KIRO_CLI_RUNNER),
            "--bfcl-root",
            str(args.bfcl_root),
            "--runtime-root",
            str(runtime_root),
            "--provider-name",
            entry.get("provider_name", "Kiro CLI"),
            "--model-name",
            entry["model_name"],
            "--categories",
            categories,
            "--cases-per-category",
            str(args.cases_per_category),
            "--temperature",
            str(args.temperature),
            "--num-threads",
            str(args.num_threads),
            "--max-step-limit",
            str(args.max_step_limit),
            "--report-markdown",
            "benchmark_report.md",
            "--error-report-json",
            "error_forensics.json",
        ]

        provider_docs_url = entry.get("provider_docs_url")
        if isinstance(provider_docs_url, str) and provider_docs_url.strip():
            command.extend(["--provider-docs-url", provider_docs_url.strip()])

        provider_license = entry.get("provider_license")
        if isinstance(provider_license, str) and provider_license.strip():
            command.extend(["--provider-license", provider_license.strip()])

        cli_path = env_value_from_entry(entry, "cli_path")
        if cli_path:
            command.extend(["--cli-path", cli_path])

        kiro_model = env_value_from_entry(entry, "kiro_model")
        if kiro_model:
            command.extend(["--kiro-model", kiro_model])

        kiro_agent = env_value_from_entry(entry, "kiro_agent")
        if kiro_agent:
            command.extend(["--kiro-agent", kiro_agent])

        if "trust_tools" in entry:
            raw_trust_tools = entry.get("trust_tools")
            trust_tools = raw_trust_tools if isinstance(raw_trust_tools, str) else ""
            command.extend(["--trust-tools", trust_tools])
        else:
            trust_tools = env_value_from_entry(entry, "trust_tools")
            if trust_tools is not None:
                command.extend(["--trust-tools", trust_tools])

        request_timeout_sec = entry.get("request_timeout_sec")
        if isinstance(request_timeout_sec, (int, float)):
            command.extend(["--request-timeout-sec", str(request_timeout_sec)])

        if entry.get("skip_login_check", False):
            command.append("--skip-login-check")
        if args.include_input_log:
            command.append("--include-input-log")
        if args.allow_agentic_run_ids:
            command.append("--allow-agentic-run-ids")
        if args.preflight_only:
            command.append("--preflight-only")
        return command

    command = [
        args.python_executable,
        str(OPENAI_COMPATIBLE_RUNNER),
        "--bfcl-root",
        str(args.bfcl_root),
        "--runtime-root",
        str(runtime_root),
        "--provider-name",
        entry["provider_name"],
        "--model-name",
        entry["model_name"],
        "--base-url",
        env_value_from_entry(entry, "base_url") or "",
        "--categories",
        categories,
        "--cases-per-category",
        str(args.cases_per_category),
        "--temperature",
        str(args.temperature),
        "--num-threads",
        str(args.num_threads),
        "--max-step-limit",
        str(args.max_step_limit),
        "--report-markdown",
        "benchmark_report.md",
        "--error-report-json",
        "error_forensics.json",
    ]

    provider_docs_url = entry.get("provider_docs_url")
    if isinstance(provider_docs_url, str) and provider_docs_url.strip():
        command.extend(["--provider-docs-url", provider_docs_url.strip()])

    provider_license = entry.get("provider_license")
    if isinstance(provider_license, str) and provider_license.strip():
        command.extend(["--provider-license", provider_license.strip()])

    default_headers_json = env_value_from_entry(entry, "default_headers_json")
    if default_headers_json:
        command.extend(["--default-headers-json", default_headers_json])

    skip_model_check = args.skip_model_checks or entry.get("skip_model_check", False)
    if skip_model_check:
        command.append("--skip-model-check")
    if args.include_input_log:
        command.append("--include-input-log")
    if args.allow_agentic_run_ids:
        command.append("--allow-agentic-run-ids")
    if args.preflight_only:
        command.append("--preflight-only")

    return command


def extract_failure_message(stdout: str, stderr: str) -> str:
    def cleaned_lines(text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    stderr_lines = cleaned_lines(stderr)
    stdout_lines = cleaned_lines(stdout)
    if not stderr_lines and not stdout_lines:
        return "child process failed without output"

    priority_markers = [
        "missing required env vars",
        "incorrect api key",
        '"message":',
        "api precheck failed",
        "http ",
        "timed out",
        "unauthorized",
        "invalid",
        "traceback",
        "exception",
        "failed",
        "error",
    ]

    for marker in priority_markers:
        for lines in (list(reversed(stderr_lines)), list(reversed(stdout_lines))):
            for line in lines:
                if marker in line.lower():
                    return line[-400:]

    if stderr_lines:
        return stderr_lines[-1][-400:]
    return stdout_lines[-1][-400:]


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def compute_relative_delta_percent(
    baseline: float | None, delta_pp: float | None
) -> float | None:
    if baseline is None or delta_pp is None or baseline == 0:
        return None
    return round((delta_pp / baseline) * 100, 4)


def classify_outcome(delta_pp: float | None, threshold_pp: float) -> str:
    if delta_pp is None:
        return "unknown"
    if delta_pp > threshold_pp:
        return "improved"
    if delta_pp < -threshold_pp:
        return "regressed"
    return "flat"


def make_run_record(
    *,
    entry: dict[str, Any],
    runtime_root: Path,
    command: list[str],
    started_at: str,
    ended_at: str,
    duration_sec: float,
    status: str,
    error_message: str | None,
    summary: dict[str, Any] | None,
    salvaged: bool = False,
    salvage_note: str | None = None,
) -> dict[str, Any]:
    overall = {}
    if summary is not None:
        metrics = summary.get("metrics_percent_point")
        if isinstance(metrics, dict):
            overall = metrics.get("Overall Acc") or {}

    baseline = overall.get("baseline") if isinstance(overall, dict) else None
    ralph = overall.get("ralph") if isinstance(overall, dict) else None
    delta_pp = overall.get("delta") if isinstance(overall, dict) else None
    if isinstance(delta_pp, (int, float)):
        delta_pp = float(delta_pp)
    else:
        delta_pp = None
    if isinstance(baseline, (int, float)):
        baseline = float(baseline)
    else:
        baseline = None
    if isinstance(ralph, (int, float)):
        ralph = float(ralph)
    else:
        ralph = None

    return {
        "id": entry["id"],
        "label": entry.get("label", entry["id"]),
        "kind": entry["kind"],
        "provider_name": entry.get("provider_name", "xAI" if entry["kind"] == "grok" else None),
        "model_name": entry["model_name"],
        "runtime_root": str(runtime_root),
        "command": command,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "duration_sec": round(duration_sec, 4),
        "status": status,
        "error_message": error_message,
        "salvaged": salvaged,
        "salvage_note": salvage_note,
        "overall_baseline": baseline,
        "overall_ralph": ralph,
        "overall_delta_pp": delta_pp,
        "overall_relative_delta_percent": compute_relative_delta_percent(
            baseline, delta_pp
        ),
        "summary": summary,
    }


def run_single_model(
    *,
    entry: dict[str, Any],
    args: argparse.Namespace,
    matrix_runs_root: Path,
) -> dict[str, Any]:
    run_slug = slugify(entry["id"])
    runtime_root = matrix_runs_root / run_slug
    runtime_root.mkdir(parents=True, exist_ok=True)

    child_env, missing_env = build_child_env(entry)
    started_at = utc_now()
    if missing_env:
        ended_at = utc_now()
        return make_run_record(
            entry=entry,
            runtime_root=runtime_root,
            command=[],
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=0.0,
            status="failed",
            error_message=f"Missing required env vars: {', '.join(missing_env)}",
            summary=None,
        )

    command = build_child_command(entry=entry, args=args, runtime_root=runtime_root)
    print(f"[RUN] {entry['id']} -> {runtime_root}")
    outcome = run_child_process_monitored(
        entry=entry,
        command=command,
        child_env=child_env,
        args=args,
        runtime_root=runtime_root,
    )
    ended_at = utc_now()

    stdout_text = read_log_text(runtime_root / "stdout.log")
    stderr_text = read_log_text(runtime_root / "stderr.log")
    summary = None if args.preflight_only else load_summary(runtime_root / "summary.json")

    if outcome.returncode == 0:
        if not args.preflight_only and summary is None:
            return make_run_record(
                entry=entry,
                runtime_root=runtime_root,
                command=command,
                started_at=started_at,
                ended_at=ended_at,
                duration_sec=0.0,
                status="failed",
                error_message=(
                    outcome.failure_note
                    or "Child runner completed without producing summary.json."
                ),
                summary=None,
                salvaged=outcome.salvaged,
                salvage_note=outcome.salvage_note,
            )
        status = "preflight_ok" if args.preflight_only else "completed"
        return make_run_record(
            entry=entry,
            runtime_root=runtime_root,
            command=command,
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=0.0,
            status=status,
            error_message=None,
            summary=summary,
            salvaged=outcome.salvaged,
            salvage_note=outcome.salvage_note,
        )

    return make_run_record(
        entry=entry,
        runtime_root=runtime_root,
        command=command,
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=0.0,
        status="failed",
        error_message=outcome.failure_note
        or extract_failure_message(stdout_text, stderr_text),
        summary=summary,
        salvaged=outcome.salvaged,
        salvage_note=outcome.salvage_note,
    )


def attach_durations_from_timestamps(record: dict[str, Any]) -> None:
    try:
        started = datetime.fromisoformat(record["started_at_utc"])
        ended = datetime.fromisoformat(record["ended_at_utc"])
    except Exception:
        return
    record["duration_sec"] = round((ended - started).total_seconds(), 4)


def classify_matrix_records(
    records: list[dict[str, Any]], flat_threshold_pp: float
) -> list[dict[str, Any]]:
    for record in records:
        attach_durations_from_timestamps(record)
        if record["status"] != "completed":
            record["outcome"] = record["status"]
            continue
        record["outcome"] = classify_outcome(
            record.get("overall_delta_pp"),
            flat_threshold_pp,
        )
    return records


def build_matrix_summary(
    *,
    records: list[dict[str, Any]],
    models_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = {
        "completed": 0,
        "improved": 0,
        "flat": 0,
        "regressed": 0,
        "failed": 0,
        "preflight_ok": 0,
        "unknown": 0,
    }
    for record in records:
        status = record["status"]
        if status in counts:
            counts[status] += 1
        outcome = record.get("outcome")
        if outcome in counts and outcome != status:
            counts[outcome] += 1

    return {
        "generated_at_utc": utc_now(),
        "repo_root": str(REPO_ROOT),
        "models_file": str(models_file),
        "bfcl_root": str(args.bfcl_root),
        "runtime_root": str(args.runtime_root),
        "categories": parse_categories(args.categories),
        "cases_per_category": args.cases_per_category,
        "flat_threshold_pp": args.flat_threshold_pp,
        "preflight_only": args.preflight_only,
        "counts": counts,
        "records": records,
    }


def build_matrix_report(summary: dict[str, Any]) -> str:
    records = summary["records"]
    improved = sorted(
        [r for r in records if r.get("outcome") == "improved"],
        key=lambda r: r.get("overall_delta_pp") or 0.0,
        reverse=True,
    )
    regressed = sorted(
        [r for r in records if r.get("outcome") == "regressed"],
        key=lambda r: r.get("overall_delta_pp") or 0.0,
    )
    flat = sorted([r for r in records if r.get("outcome") == "flat"], key=lambda r: r["id"])
    failed = sorted(
        [r for r in records if r["status"] == "failed"],
        key=lambda r: r["id"],
    )
    salvaged_rows = sorted(
        [r for r in records if r.get("salvaged")],
        key=lambda r: r["id"],
    )

    lines = [
        "# Prompt-Mode BFCL RALPH Matrix Report",
        "",
        f"- Generated (UTC): {summary['generated_at_utc']}",
        f"- Models file: `{summary['models_file']}`",
        f"- BFCL root: `{summary['bfcl_root']}`",
        f"- Runtime root: `{summary['runtime_root']}`",
        f"- Categories: `{', '.join(summary['categories'])}`",
        f"- Cases per category: `{summary['cases_per_category']}`",
        f"- Preflight only: `{summary['preflight_only']}`",
        "",
        "## Counts",
        "",
        f"- Completed: `{summary['counts']['completed']}`",
        f"- Improved: `{summary['counts']['improved']}`",
        f"- Flat: `{summary['counts']['flat']}`",
        f"- Regressed: `{summary['counts']['regressed']}`",
        f"- Failed: `{summary['counts']['failed']}`",
        "",
        "## Leaderboard",
        "",
        "| Model | Kind | Outcome | Baseline | RALPH | Delta (pp) | Relative Delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for record in records:
        baseline = record.get("overall_baseline")
        ralph = record.get("overall_ralph")
        delta = record.get("overall_delta_pp")
        rel = record.get("overall_relative_delta_percent")
        baseline_txt = "N/A" if baseline is None else f"{baseline:.2f}"
        ralph_txt = "N/A" if ralph is None else f"{ralph:.2f}"
        delta_txt = "N/A" if delta is None else f"{delta:+.2f}"
        rel_txt = "N/A" if rel is None else f"{rel:+.2f}%"
        lines.append(
            f"| {record['id']} | {record['kind']} | {record['outcome']} | "
            f"{baseline_txt} | {ralph_txt} | {delta_txt} | {rel_txt} |"
        )

    def append_model_section(title: str, rows: list[dict[str, Any]], include_error: bool = False) -> None:
        if not rows:
            return
        lines.extend(["", f"## {title}", ""])
        for row in rows:
            detail = (
                "N/A"
                if row.get("overall_delta_pp") is None
                else f"{row['overall_delta_pp']:+.2f} pp"
            )
            line = f"- `{row['id']}`: {detail}"
            if include_error and row.get("error_message"):
                line += f" | {row['error_message']}"
            lines.append(line)

    append_model_section("Improved", improved)
    append_model_section("Regressed", regressed)
    append_model_section("Flat", flat)
    if salvaged_rows:
        lines.extend(["", "## Salvaged", ""])
        for row in salvaged_rows:
            note = row.get("salvage_note") or "eval-only salvage recovered this run"
            lines.append(f"- `{row['id']}`: {note}")
    append_model_section("Failed", failed, include_error=True)

    return "\n".join(lines) + "\n"


def write_matrix_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "label",
        "kind",
        "provider_name",
        "model_name",
        "status",
        "outcome",
        "overall_baseline",
        "overall_ralph",
        "overall_delta_pp",
        "overall_relative_delta_percent",
        "duration_sec",
        "runtime_root",
        "error_message",
        "salvaged",
        "salvage_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    categories = parse_categories(args.categories)
    if args.cases_per_category < 0:
        raise SystemExit("--cases-per-category must be >= 0.")
    if args.num_threads <= 0:
        raise SystemExit("--num-threads must be >= 1.")
    if args.max_step_limit <= 0:
        raise SystemExit("--max-step-limit must be >= 1.")
    if args.flat_threshold_pp < 0:
        raise SystemExit("--flat-threshold-pp must be >= 0.")
    if args.child_poll_interval_sec <= 0:
        raise SystemExit("--child-poll-interval-sec must be > 0.")
    if args.salvage_stall_sec < 0:
        raise SystemExit("--salvage-stall-sec must be >= 0.")
    if args.child_max_runtime_sec < 0:
        raise SystemExit("--child-max-runtime-sec must be >= 0.")

    requested_ids = None
    if args.model_ids:
        requested_ids = {item.strip() for item in args.model_ids.split(",") if item.strip()}
        if not requested_ids:
            raise SystemExit("--model-ids was provided but empty after parsing.")

    models = select_models(load_models_file(args.models_file), requested_ids)

    args.runtime_root.mkdir(parents=True, exist_ok=True)
    matrix_runs_root = args.runtime_root / "runs"
    matrix_runs_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[MATRIX] Starting {len(models)} model runs "
        f"({'preflight' if args.preflight_only else 'benchmark'})"
    )

    records: list[dict[str, Any]] = []
    for entry in models:
        record = run_single_model(
            entry=entry,
            args=args,
            matrix_runs_root=matrix_runs_root,
        )
        records.append(record)
        if record["status"] == "failed":
            print(f"[FAIL] {entry['id']}: {record['error_message']}")
            if args.fail_fast:
                break
        else:
            print(f"[OK] {entry['id']} -> {record['status']}")

    records = classify_matrix_records(records, args.flat_threshold_pp)
    summary = build_matrix_summary(
        records=records,
        models_file=args.models_file,
        args=args,
    )

    summary_path = args.runtime_root / "matrix_summary.json"
    results_csv_path = args.runtime_root / "matrix_results.csv"
    report_path = args.runtime_root / "matrix_report.md"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_matrix_csv(results_csv_path, records)
    report_path.write_text(build_matrix_report(summary), encoding="utf-8")

    print(f"[DONE] Summary: {summary_path}")
    print(f"[DONE] CSV: {results_csv_path}")
    print(f"[DONE] Report: {report_path}")

    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
