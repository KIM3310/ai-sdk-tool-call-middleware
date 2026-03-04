#!/usr/bin/env python3
from __future__ import annotations

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parent / "run_grok_bfcl_ralph.py"
MOD = runpy.run_path(str(SCRIPT_PATH))

parse_categories = MOD["parse_categories"]
validate_args = MOD["validate_args"]
validate_run_ids_categories = MOD["validate_run_ids_categories"]
check_grok_api_key = MOD["check_grok_api_key"]
require_grok_api_key = MOD["require_grok_api_key"]
resolve_categories_for_run_ids = MOD["resolve_categories_for_run_ids"]
resolve_runtime_categories = MOD["resolve_runtime_categories"]
extract_result_rows = MOD["extract_result_rows"]
classify_inference_error = MOD["classify_inference_error"]
verify_generation_health = MOD["verify_generation_health"]
bootstrap_bfcl_imports = MOD["bootstrap_bfcl_imports"]
create_run_ids_map = MOD["create_run_ids_map"]
compute_metric_outcomes = MOD["compute_metric_outcomes"]
build_markdown_report = MOD["build_markdown_report"]
resolve_report_markdown_path = MOD["resolve_report_markdown_path"]
resolve_error_report_json_path = MOD["resolve_error_report_json_path"]
clear_stale_output_file = MOD["clear_stale_output_file"]
run_generation_and_eval = MOD["run_generation_and_eval"]


class TestRunGrokBfclRalph(unittest.TestCase):
    def _patch_bfcl_runtime_modules(self, generation_main, evaluation_main):
        bfcl_eval = ModuleType("bfcl_eval")
        bfcl_eval.__path__ = []

        llm_generation = ModuleType("bfcl_eval._llm_response_generation")
        llm_generation.main = generation_main

        eval_checker = ModuleType("bfcl_eval.eval_checker")
        eval_checker.__path__ = []
        eval_runner = ModuleType("bfcl_eval.eval_checker.eval_runner")
        eval_runner.main = evaluation_main
        eval_checker.eval_runner = eval_runner

        constants_pkg = ModuleType("bfcl_eval.constants")
        constants_pkg.__path__ = []
        eval_config = ModuleType("bfcl_eval.constants.eval_config")
        eval_config.DOTENV_PATH = Path("/tmp/.env")
        default_prompts = ModuleType("bfcl_eval.constants.default_prompts")
        default_prompts.MAXIMUM_STEP_LIMIT = 20
        constants_pkg.eval_config = eval_config
        constants_pkg.default_prompts = default_prompts

        model_handler = ModuleType("bfcl_eval.model_handler")
        model_handler.__path__ = []
        base_handler = ModuleType("bfcl_eval.model_handler.base_handler")
        base_handler.MAXIMUM_STEP_LIMIT = 20
        model_handler.base_handler = base_handler

        dotenv_mod = ModuleType("dotenv")
        dotenv_mod.load_dotenv = lambda **_kwargs: None

        bfcl_eval._llm_response_generation = llm_generation
        bfcl_eval.eval_checker = eval_checker
        bfcl_eval.constants = constants_pkg
        bfcl_eval.model_handler = model_handler

        return patch.dict(
            sys.modules,
            {
                "bfcl_eval": bfcl_eval,
                "bfcl_eval._llm_response_generation": llm_generation,
                "bfcl_eval.eval_checker": eval_checker,
                "bfcl_eval.eval_checker.eval_runner": eval_runner,
                "bfcl_eval.constants": constants_pkg,
                "bfcl_eval.constants.eval_config": eval_config,
                "bfcl_eval.constants.default_prompts": default_prompts,
                "bfcl_eval.model_handler": model_handler,
                "bfcl_eval.model_handler.base_handler": base_handler,
                "dotenv": dotenv_mod,
            },
        )

    def test_parse_categories_dedup_preserves_first_order(self) -> None:
        got = parse_categories("simple_python,multiple,simple_python,parallel,,multiple")
        self.assertEqual(got, ["simple_python", "multiple", "parallel"])

    def test_validate_args_rejects_bad_values(self) -> None:
        bad_cases = [
            SimpleNamespace(
                cases_per_category=-1,
                num_threads=1,
                temperature=0.0,
                model_name="grok-4-1-fast-reasoning",
                error_report_top_n=5,
            ),
            SimpleNamespace(
                cases_per_category=1,
                num_threads=0,
                temperature=0.0,
                model_name="grok-4-1-fast-reasoning",
                error_report_top_n=5,
            ),
            SimpleNamespace(
                cases_per_category=1,
                num_threads=1,
                temperature=-0.1,
                model_name="grok-4-1-fast-reasoning",
                error_report_top_n=5,
            ),
            SimpleNamespace(
                cases_per_category=1,
                num_threads=1,
                temperature=0.0,
                model_name="gpt-4o",
                error_report_top_n=5,
            ),
            SimpleNamespace(
                cases_per_category=1,
                num_threads=1,
                temperature=0.0,
                model_name="grok-4-1-fast-reasoning",
                error_report_top_n=0,
            ),
        ]
        for args in bad_cases:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit):
                    validate_args(args)

    def test_validate_run_ids_categories_blocks_agentic_by_default(self) -> None:
        with self.assertRaises(SystemExit):
            validate_run_ids_categories(["simple_python", "memory_kv"], False)
        # Explicit override should allow agentic categories.
        validate_run_ids_categories(["simple_python", "memory_kv"], True)

    def test_require_grok_api_key_rejects_blank(self) -> None:
        with self.assertRaises(SystemExit):
            require_grok_api_key("   ")

    def test_check_grok_api_key_model_visibility(self) -> None:
        payload = {
            "data": [
                {"id": "grok-4-1-fast-reasoning"},
                {"id": "grok-4-0709"},
            ]
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        class _FakeResp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return payload_bytes

        with patch("urllib.request.urlopen", lambda *args, **kwargs: _FakeResp()):
            # Visible model should pass.
            check_grok_api_key("xai-abc", model_name="grok-4-1-fast-reasoning")
            # Invisible model should be rejected.
            with self.assertRaises(SystemExit):
                check_grok_api_key("xai-abc", model_name="grok-4-1-fast-non-reasoning")

    def test_resolve_categories_for_run_ids_reports_missing_dependency(self) -> None:
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "bfcl_eval.utils":
                raise ModuleNotFoundError(
                    "No module named 'filelock'",
                    name="filelock",
                )
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(SystemExit) as ctx:
                resolve_categories_for_run_ids(["simple_python"])
        self.assertIn("Missing Python dependency 'filelock'", str(ctx.exception))

    def test_resolve_categories_for_run_ids_invalid_category_message(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            resolve_categories_for_run_ids(["not_a_real_category"])
        self.assertIn("Invalid --categories value", str(ctx.exception))

    def test_resolve_runtime_categories_validates_even_without_run_ids(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            resolve_runtime_categories(
                categories=["not_a_real_category"],
                run_ids_enabled=False,
                allow_agentic_run_ids=False,
            )
        self.assertIn("Invalid --categories value", str(ctx.exception))

    def test_extract_result_rows_supports_jsonl_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            jsonl_path = tmp / "jsonl.json"
            jsonl_path.write_text(
                '{"id":"a","result":"ok"}\n{"id":"b","result":"Error during inference: x"}\n',
                encoding="utf-8",
            )
            rows, parse_failed = extract_result_rows(jsonl_path)
            self.assertEqual([row["id"] for row in rows], ["a", "b"])
            self.assertFalse(parse_failed)

            dict_path = tmp / "dict.json"
            dict_path.write_text(
                json.dumps({"id": "single", "result": "ok"}),
                encoding="utf-8",
            )
            rows, parse_failed = extract_result_rows(dict_path)
            self.assertEqual([row["id"] for row in rows], ["single"])
            self.assertFalse(parse_failed)

            bad_path = tmp / "bad.json"
            bad_path.write_text("not-json", encoding="utf-8")
            rows, parse_failed = extract_result_rows(bad_path)
            self.assertEqual(rows, [])
            self.assertTrue(parse_failed)

    def test_compute_metric_outcomes_and_markdown_report(self) -> None:
        summary = {
            "categories": ["simple_python", "multiple"],
            "cases_per_category": 5,
            "metrics_percent_point": {
                "Overall Acc": {"baseline": 60.0, "ralph": 65.0, "delta": 5.0},
                "Live Acc": {"baseline": 40.0, "ralph": 35.0, "delta": -5.0},
                "Multi Turn Acc": {"baseline": 50.0, "ralph": 50.0, "delta": 0.0},
                "Relevance Detection": {"baseline": None, "ralph": None, "delta": None},
            },
        }
        outcomes = compute_metric_outcomes(summary["metrics_percent_point"])
        self.assertEqual(outcomes["wins"], [("Overall Acc", 5.0)])
        self.assertEqual(outcomes["losses"], [("Live Acc", -5.0)])
        self.assertEqual(outcomes["ties"], ["Multi Turn Acc"])
        self.assertEqual(outcomes["unknown"], ["Relevance Detection"])

        report = build_markdown_report(
            summary=summary,
            model_name="grok-4-1-fast-reasoning",
            baseline_display="baseline-name",
            ralph_display="ralph-name",
            runtime_root=Path("/tmp/runtime"),
            run_ids_enabled=True,
        )
        self.assertIn("# Grok BFCL Benchmark Report", report)
        self.assertIn("Verdict: `balanced`", report)
        self.assertIn("| Overall Acc | 60.00 | 65.00 | +5.00 |", report)
        self.assertIn("Best gain: `Overall Acc` (+5.00 pp)", report)
        self.assertIn("Biggest drop: `Live Acc` (-5.00 pp)", report)

    def test_resolve_report_markdown_path(self) -> None:
        runtime_root = Path("/tmp/runtime")
        self.assertIsNone(resolve_report_markdown_path(runtime_root, None))
        self.assertEqual(
            resolve_report_markdown_path(runtime_root, Path("reports/result.md")),
            Path("/tmp/runtime/reports/result.md"),
        )
        self.assertEqual(
            resolve_report_markdown_path(runtime_root, Path("/tmp/abs/report.md")),
            Path("/tmp/abs/report.md"),
        )

    def test_resolve_error_report_json_path(self) -> None:
        runtime_root = Path("/tmp/runtime")
        self.assertIsNone(resolve_error_report_json_path(runtime_root, None))
        self.assertEqual(
            resolve_error_report_json_path(runtime_root, Path("reports/error.json")),
            Path("/tmp/runtime/reports/error.json"),
        )
        self.assertEqual(
            resolve_error_report_json_path(runtime_root, Path("/tmp/abs/error.json")),
            Path("/tmp/abs/error.json"),
        )

    def test_clear_stale_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            stale_file = tmp / "stale.json"
            stale_file.write_text("{}", encoding="utf-8")
            clear_stale_output_file(stale_file)
            self.assertFalse(stale_file.exists())
            # Idempotent on missing files.
            clear_stale_output_file(stale_file)

            keep_dir = tmp / "keep_dir"
            keep_dir.mkdir()
            clear_stale_output_file(keep_dir)
            self.assertTrue(keep_dir.exists())

    def test_classify_inference_error(self) -> None:
        self.assertEqual(
            classify_inference_error(
                "Error during inference: Incorrect API key provided",
                "",
            ),
            "auth_incorrect_key",
        )
        self.assertEqual(
            classify_inference_error(
                "Error during inference: Error code: 429",
                "",
            ),
            "rate_limit",
        )
        self.assertEqual(
            classify_inference_error("Error during inference: something else", "ConnectionError"),
            "connection_error",
        )

    def test_verify_generation_health_writes_error_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime_root = Path(td)
            result_file = (
                runtime_root
                / "result"
                / "model-x"
                / "non_live"
                / "BFCL_v4_simple_python_result.json"
            )
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "a",
                                "result": "Error during inference: Error code: 429",
                                "traceback": "",
                            }
                        ),
                        json.dumps({"id": "b", "result": "ok"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report_path = runtime_root / "forensics" / "errors.json"
            verify_generation_health(
                runtime_root=runtime_root,
                registries=["model-x"],
                error_report_json=report_path,
                error_report_top_n=3,
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("registries", payload)
            self.assertIn("model-x", payload["registries"])
            reg = payload["registries"]["model-x"]
            self.assertEqual(reg["total_items"], 2)
            self.assertEqual(reg["error_items"], 1)
            reasons = reg["error_reasons"]
            self.assertEqual(reasons[0]["reason"], "rate_limit")
            self.assertEqual(reasons[0]["count"], 1)

    def test_verify_generation_health_all_fail_still_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime_root = Path(td)
            result_file = (
                runtime_root
                / "result"
                / "model-y"
                / "non_live"
                / "BFCL_v4_simple_python_result.json"
            )
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                json.dumps(
                    {
                        "id": "x",
                        "result": "Error during inference: Incorrect API key provided",
                        "traceback": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report_path = runtime_root / "forensics" / "errors.json"
            with self.assertRaises(SystemExit):
                verify_generation_health(
                    runtime_root=runtime_root,
                    registries=["model-y"],
                    error_report_json=report_path,
                    error_report_top_n=3,
                )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            reg = payload["registries"]["model-y"]
            self.assertEqual(reg["total_items"], 1)
            self.assertEqual(reg["error_items"], 1)
            self.assertEqual(reg["error_reasons"][0]["reason"], "auth_incorrect_key")

    def test_run_generation_and_eval_aborts_before_ralph_on_baseline_failure(self) -> None:
        baseline = "grok-4-1-fast-reasoning-baseline-prompt"
        ralph = "grok-4-1-fast-reasoning-ralph-loop-prompt"
        calls: list[str] = []

        def fake_generation_main(args):
            calls.extend(args.model)

        def fake_eval_main(*_args, **_kwargs):
            raise AssertionError("evaluation_main should not run on baseline hard failure")

        def fake_verify(
            *,
            runtime_root: Path,
            registries: list[str],
            error_report_json=None,
            error_report_top_n=5,
        ) -> None:
            if registries == [baseline]:
                raise SystemExit("baseline failed")

        original_verify = run_generation_and_eval.__globals__["verify_generation_health"]
        run_generation_and_eval.__globals__["verify_generation_health"] = fake_verify
        try:
            with self._patch_bfcl_runtime_modules(
                fake_generation_main,
                fake_eval_main,
            ):
                with self.assertRaises(SystemExit):
                    run_generation_and_eval(
                        baseline_registry=baseline,
                        ralph_registry=ralph,
                        categories=["simple_python"],
                        temperature=0.001,
                        num_threads=1,
                        include_input_log=False,
                        run_ids_enabled=True,
                        runtime_root=Path(__file__).resolve().parent / "runtime",
                        error_report_json=None,
                        error_report_top_n=5,
                    )
        finally:
            run_generation_and_eval.__globals__["verify_generation_health"] = original_verify

        self.assertEqual(
            calls,
            [baseline],
            "ralph generation should not start when baseline generation is fully broken",
        )

    def test_run_generation_and_eval_reports_generation_crash(self) -> None:
        baseline = "grok-4-1-fast-reasoning-baseline-prompt"
        ralph = "grok-4-1-fast-reasoning-ralph-loop-prompt"

        def fake_generation_main(_args):
            raise RuntimeError("boom")

        def fake_eval_main(*_args, **_kwargs):
            raise AssertionError("evaluation_main should not run when generation crashes")

        with self._patch_bfcl_runtime_modules(
            fake_generation_main,
            fake_eval_main,
        ):
            with self.assertRaises(SystemExit) as ctx:
                run_generation_and_eval(
                    baseline_registry=baseline,
                    ralph_registry=ralph,
                    categories=["simple_python"],
                    temperature=0.001,
                    num_threads=1,
                    include_input_log=False,
                    run_ids_enabled=True,
                    runtime_root=Path(__file__).resolve().parent / "runtime",
                    error_report_json=None,
                    error_report_top_n=5,
                )
        self.assertIn("Generation crashed", str(ctx.exception))

    def test_create_run_ids_map_expands_dependencies_for_memory(self) -> None:
        bfcl_root = Path("/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard")
        if not bfcl_root.exists():
            self.skipTest("BFCL root not found; skipping dependency-expansion integration test.")

        runtime_root = Path(__file__).resolve().parent / "runtime"
        bootstrap_bfcl_imports(bfcl_root, runtime_root)
        from bfcl_eval.utils import load_dataset_entry

        ids = create_run_ids_map(["memory_kv"], 1)["memory_kv"]
        self.assertIn("memory_kv_0-customer-0", ids)
        self.assertIn("memory_kv_prereq_0-customer-0", ids)
        self.assertIn("memory_kv_prereq_9-customer-9", ids)
        # Closure check: every selected id's dependencies must also be selected.
        all_entries = load_dataset_entry("memory_kv", include_prereq=True)
        by_id = {entry["id"]: entry for entry in all_entries}
        selected = set(ids)
        for selected_id in ids:
            entry = by_id[selected_id]
            for dep_id in entry.get("depends_on", []):
                if dep_id in by_id:
                    self.assertIn(dep_id, selected)


if __name__ == "__main__":
    unittest.main()
