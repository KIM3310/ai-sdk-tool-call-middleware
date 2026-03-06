#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parent / "run_prompt_bfcl_ralph_matrix.py"
)
MOD = runpy.run_path(str(SCRIPT_PATH))

slugify = MOD["slugify"]
load_models_file = MOD["load_models_file"]
select_models = MOD["select_models"]
build_child_env = MOD["build_child_env"]
build_child_command = MOD["build_child_command"]
ChildRunOutcome = MOD["ChildRunOutcome"]
classify_outcome = MOD["classify_outcome"]
build_matrix_summary = MOD["build_matrix_summary"]
build_matrix_report = MOD["build_matrix_report"]
run_single_model = MOD["run_single_model"]
should_attempt_eval_only_salvage = MOD["should_attempt_eval_only_salvage"]
attempt_score_json_summary_salvage = MOD["attempt_score_json_summary_salvage"]


class TestRunPromptBfclRalphMatrix(unittest.TestCase):
    def test_slugify_normalizes_ids(self) -> None:
        self.assertEqual(slugify("OpenRouter / Qwen 3"), "openrouter-qwen-3")
        self.assertEqual(slugify("   "), "model")

    def test_load_and_select_models_filters_enabled_and_requested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "models.json"
            path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "grok-1",
                                "kind": "grok",
                                "enabled": True,
                                "model_name": "grok-1",
                            },
                            {
                                "id": "disabled",
                                "kind": "grok",
                                "enabled": False,
                                "model_name": "grok-2",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            models = load_models_file(path)
            selected = select_models(models, {"grok-1"})
            self.assertEqual([item["id"] for item in selected], ["grok-1"])

    def test_build_child_env_for_openai_compatible_checks_missing_envs(self) -> None:
        entry = {
            "id": "openrouter-qwen",
            "kind": "openai-compatible",
            "enabled": True,
            "provider_name": "OpenRouter",
            "model_name": "qwen/test",
            "base_url_env": "OPENROUTER_BASE_URL",
            "api_key_env": "OPENROUTER_API_KEY",
        }
        with patch.dict("os.environ", {}, clear=False):
            _env, missing = build_child_env(entry)
        self.assertEqual(missing, ["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"])

    def test_build_child_command_for_grok_and_openai_compatible(self) -> None:
        args = SimpleNamespace(
            python_executable="/usr/bin/python3",
            bfcl_root=Path("/tmp/bfcl"),
            categories="simple_python,multiple",
            cases_per_category=3,
            temperature=0.001,
            num_threads=1,
            max_step_limit=20,
            include_input_log=False,
            allow_agentic_run_ids=False,
            preflight_only=False,
            skip_model_checks=True,
        )
        grok_cmd = build_child_command(
            entry={
                "id": "grok-4",
                "kind": "grok",
                "model_name": "grok-4-latest",
            },
            args=args,
            runtime_root=Path("/tmp/runtime/grok"),
        )
        self.assertIn("run_grok_prompt_bfcl_ralph.py", grok_cmd[1])
        self.assertIn("--skip-key-check", grok_cmd)

        openai_cmd = build_child_command(
            entry={
                "id": "openrouter-qwen",
                "kind": "openai-compatible",
                "provider_name": "OpenRouter",
                "provider_license": "Proprietary",
                "model_name": "qwen/test",
                "base_url": "https://openrouter.ai/api/v1",
                "default_headers_json": '{"HTTP-Referer":"https://example.com"}',
            },
            args=args,
            runtime_root=Path("/tmp/runtime/openrouter"),
        )
        self.assertIn("run_openai_compatible_prompt_bfcl_ralph.py", openai_cmd[1])
        self.assertIn("--skip-model-check", openai_cmd)
        self.assertIn("https://openrouter.ai/api/v1", openai_cmd)

        kiro_cmd = build_child_command(
            entry={
                "id": "kiro-default",
                "kind": "kiro-cli",
                "provider_name": "Kiro CLI",
                "provider_license": "Proprietary",
                "model_name": "kiro-cli-default",
                "cli_path": "kiro-cli",
                "kiro_agent": "coder",
                "trust_tools": "",
            },
            args=args,
            runtime_root=Path("/tmp/runtime/kiro"),
        )
        self.assertIn("run_kiro_cli_prompt_bfcl_ralph.py", kiro_cmd[1])
        self.assertIn("--cli-path", kiro_cmd)
        self.assertIn("--kiro-agent", kiro_cmd)
        self.assertIn("--trust-tools", kiro_cmd)

    def test_classify_outcome_and_report(self) -> None:
        records = [
            {
                "id": "model-a",
                "label": "Model A",
                "kind": "grok",
                "provider_name": "xAI",
                "model_name": "grok-a",
                "runtime_root": "/tmp/a",
                "status": "completed",
                "outcome": "improved",
                "overall_baseline": 10.0,
                "overall_ralph": 12.0,
                "overall_delta_pp": 2.0,
                "overall_relative_delta_percent": 20.0,
                "salvaged": True,
                "salvage_note": "Recovered via eval-only salvage",
            },
            {
                "id": "model-b",
                "label": "Model B",
                "kind": "openai-compatible",
                "provider_name": "OpenRouter",
                "model_name": "model-b",
                "runtime_root": "/tmp/b",
                "status": "failed",
                "outcome": "failed",
                "overall_baseline": None,
                "overall_ralph": None,
                "overall_delta_pp": None,
                "overall_relative_delta_percent": None,
                "error_message": "missing key",
                "salvaged": False,
                "salvage_note": None,
            },
        ]
        args = SimpleNamespace(
            bfcl_root=Path("/tmp/bfcl"),
            runtime_root=Path("/tmp/runtime"),
            categories="simple_python,multiple",
            cases_per_category=3,
            flat_threshold_pp=0.0,
            preflight_only=False,
        )
        summary = build_matrix_summary(
            records=records,
            models_file=Path("/tmp/models.json"),
            args=args,
        )
        report = build_matrix_report(summary)
        self.assertEqual(classify_outcome(0.0, 0.0), "flat")
        self.assertEqual(classify_outcome(1.0, 0.0), "improved")
        self.assertIn("# Prompt-Mode BFCL RALPH Matrix Report", report)
        self.assertIn("| model-a | grok | improved | 10.00 | 12.00 | +2.00 | +20.00% |", report)
        self.assertIn("## Salvaged", report)
        self.assertIn("Recovered via eval-only salvage", report)
        self.assertIn("missing key", report)

    def test_should_attempt_eval_only_salvage_requires_complete_stale_results(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime_root = Path(td)
            summary_path = runtime_root / "summary.json"
            result_a = runtime_root / "a.json"
            result_b = runtime_root / "b.json"
            result_a.write_text("{}", encoding="utf-8")
            result_b.write_text("{}", encoding="utf-8")
            old_time = 1_700_000_000
            os.utime(result_a, (old_time, old_time))
            os.utime(result_b, (old_time, old_time))

            self.assertTrue(
                should_attempt_eval_only_salvage(
                    expected_result_paths=[result_a, result_b],
                    summary_path=summary_path,
                    stall_sec=60.0,
                    now=old_time + 61.0,
                )
            )
            self.assertFalse(
                should_attempt_eval_only_salvage(
                    expected_result_paths=[result_a, result_b],
                    summary_path=summary_path,
                    stall_sec=60.0,
                    now=old_time + 30.0,
                )
            )
            summary_path.write_text("{}", encoding="utf-8")
            self.assertFalse(
                should_attempt_eval_only_salvage(
                    expected_result_paths=[result_a, result_b],
                    summary_path=summary_path,
                    stall_sec=60.0,
                    now=old_time + 61.0,
                )
            )

    def test_run_single_model_marks_salvaged_completion(self) -> None:
        args = SimpleNamespace(
            python_executable="/usr/bin/python3",
            bfcl_root=Path("/tmp/bfcl"),
            categories="simple_python,multiple",
            cases_per_category=3,
            temperature=0.001,
            num_threads=1,
            max_step_limit=20,
            include_input_log=False,
            allow_agentic_run_ids=False,
            preflight_only=False,
            skip_model_checks=True,
            child_poll_interval_sec=0.01,
            salvage_stall_sec=60.0,
            child_max_runtime_sec=0.0,
        )
        entry = {
            "id": "ollama-phi3",
            "kind": "openai-compatible",
            "provider_name": "Ollama",
            "model_name": "phi3:latest",
            "base_url": "http://127.0.0.1:11434/v1",
        }

        def fake_run_child_process_monitored(*, runtime_root: Path, **_kwargs):
            summary = {
                "categories": ["simple_python", "multiple"],
                "cases_per_category": 3,
                "metrics_percent_point": {
                    "Overall Acc": {
                        "baseline": 7.0,
                        "ralph": 7.25,
                        "delta": 0.25,
                    }
                },
            }
            (runtime_root / "summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            (runtime_root / "stdout.log").write_text("", encoding="utf-8")
            (runtime_root / "stderr.log").write_text("", encoding="utf-8")
            return ChildRunOutcome(
                returncode=0,
                salvaged=True,
                salvage_note="Recovered via eval-only salvage",
            )

        with tempfile.TemporaryDirectory() as td:
            matrix_runs_root = Path(td)
            with patch.dict(
                run_single_model.__globals__,
                {
                    "build_child_env": lambda _entry: ({"OPENAI_API_KEY": "dummy"}, []),
                    "run_child_process_monitored": fake_run_child_process_monitored,
                },
            ):
                record = run_single_model(
                    entry=entry,
                    args=args,
                    matrix_runs_root=matrix_runs_root,
                )

        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["salvaged"])
        self.assertEqual(record["salvage_note"], "Recovered via eval-only salvage")
        self.assertEqual(record["overall_delta_pp"], 0.25)

    def test_attempt_score_json_summary_salvage_recovers_from_category_scores(self) -> None:
        def fake_get_category_score(score_dict, test_category):
            if test_category in score_dict:
                payload = dict(score_dict[test_category])
                payload["display_accuracy"] = payload["accuracy"]
                return payload
            return {"accuracy": 0.0, "total_count": 1, "display_accuracy": "N/A"}

        def fake_unweighted_accuracy(items, display_na_if_category_missing=True):
            has_na = any(item["display_accuracy"] == "N/A" for item in items)
            accuracy = sum(item["accuracy"] for item in items) / len(items)
            return {
                "accuracy": accuracy,
                "total_count": sum(item["total_count"] for item in items),
                "display_accuracy": (
                    "N/A" if has_na and display_na_if_category_missing else accuracy
                ),
            }

        def fake_weighted_accuracy(items, display_na_if_category_missing=True):
            has_na = any(item["display_accuracy"] == "N/A" for item in items)
            total_count = sum(item["total_count"] for item in items) or 1
            accuracy = (
                sum(item["accuracy"] * item["total_count"] for item in items)
                / total_count
            )
            return {
                "accuracy": accuracy,
                "total_count": total_count,
                "display_accuracy": (
                    "N/A" if has_na and display_na_if_category_missing else accuracy
                ),
            }

        def fake_percentage_weighted_accuracy(
            items, weights, display_na_if_category_missing=True
        ):
            has_na = any(item["display_accuracy"] == "N/A" for item in items)
            normalized = [weight / sum(weights) for weight in weights]
            accuracy = sum(
                item["accuracy"] * weight for item, weight in zip(items, normalized)
            )
            return {
                "accuracy": accuracy,
                "total_count": sum(item["total_count"] for item in items),
                "display_accuracy": (
                    "N/A" if has_na and display_na_if_category_missing else accuracy
                ),
            }

        with tempfile.TemporaryDirectory() as td:
            runtime_root = Path(td)
            baseline_dir = runtime_root / "score" / "baseline-registry" / "non_live"
            ralph_dir = runtime_root / "score" / "ralph-registry" / "non_live"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            ralph_dir.mkdir(parents=True, exist_ok=True)

            baseline_payload = {
                "accuracy": 0.5,
                "correct_count": 1,
                "total_count": 2,
            }
            ralph_payload = {
                "accuracy": 1.0,
                "correct_count": 2,
                "total_count": 2,
            }
            for category in [
                "simple_python",
                "multiple",
                "parallel",
                "parallel_multiple",
            ]:
                (baseline_dir / f"BFCL_v4_{category}_score.json").write_text(
                    json.dumps(baseline_payload),
                    encoding="utf-8",
                )
                (ralph_dir / f"BFCL_v4_{category}_score.json").write_text(
                    json.dumps(ralph_payload),
                    encoding="utf-8",
                )

            with patch.dict(
                attempt_score_json_summary_salvage.__globals__,
                {
                    "load_bfcl_scoring_helpers": lambda: {
                        "get_category_score": fake_get_category_score,
                        "calculate_unweighted_accuracy": fake_unweighted_accuracy,
                        "calculate_weighted_accuracy": fake_weighted_accuracy,
                        "calculate_percentage_weighted_accuracy": (
                            fake_percentage_weighted_accuracy
                        ),
                    }
                },
            ):
                summary = attempt_score_json_summary_salvage(
                    baseline_registry="baseline-registry",
                    ralph_registry="ralph-registry",
                    runtime_root=runtime_root,
                    categories=["simple_python", "multiple", "parallel", "parallel_multiple"],
                    cases_per_category=1,
                )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["cases_per_category"], 1)
        self.assertEqual(summary["categories"], ["simple_python", "multiple", "parallel", "parallel_multiple"])
        self.assertGreater(
            summary["metrics_percent_point"]["Overall Acc"]["delta"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
