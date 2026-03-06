#!/usr/bin/env python3
from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parent / "run_openai_compatible_prompt_bfcl_ralph.py"
)
MOD = runpy.run_path(str(SCRIPT_PATH))

validate_args = MOD["validate_args"]
validate_json_object = MOD["validate_json_object"]
check_model_access = MOD["check_model_access"]
ensure_runtime_layout = MOD["ensure_runtime_layout"]
build_markdown_report = MOD["build_markdown_report"]
require_api_key = MOD["require_api_key"]
require_base_url = MOD["require_base_url"]


class TestRunOpenAICompatibleBfclRalph(unittest.TestCase):
    def test_validate_args_rejects_missing_model_or_base_url(self) -> None:
        bad_cases = [
            SimpleNamespace(
                cases_per_category=1,
                num_threads=1,
                temperature=0.0,
                model_name="",
                base_url="https://example.com/v1",
                provider_name="Provider",
                error_report_top_n=5,
                request_timeout_sec=30.0,
                max_step_limit=20,
            ),
            SimpleNamespace(
                cases_per_category=1,
                num_threads=1,
                temperature=0.0,
                model_name="my-model",
                base_url=" ",
                provider_name="Provider",
                error_report_top_n=5,
                request_timeout_sec=30.0,
                max_step_limit=20,
            ),
        ]

        for args in bad_cases:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit):
                    validate_args(args)

    def test_validate_json_object_normalizes_compact_json(self) -> None:
        normalized = validate_json_object('{ "HTTP-Referer": "https://example.com" }')
        self.assertEqual(normalized, '{"HTTP-Referer":"https://example.com"}')
        self.assertIsNone(validate_json_object("   "))

        with self.assertRaises(SystemExit):
            validate_json_object('["not","an","object"]')

    def test_require_api_key_and_base_url_read_cli_or_env(self) -> None:
        with patch.dict("os.environ", {"OPENAI_COMPATIBLE_API_KEY": "test-key"}, clear=False):
            self.assertEqual(require_api_key(None), "test-key")
        self.assertEqual(require_api_key("cli-key"), "cli-key")
        self.assertEqual(require_base_url("https://example.com/v1/"), "https://example.com/v1")

    def test_check_model_access_rejects_invisible_model(self) -> None:
        payload = {"data": [{"id": "visible-model"}, {"id": "second-model"}]}
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
            check_model_access(
                api_key="key",
                base_url="https://example.com/v1",
                model_name="visible-model",
                default_headers_json='{"X-Test":"1"}',
            )
            with self.assertRaises(SystemExit):
                check_model_access(
                    api_key="key",
                    base_url="https://example.com/v1",
                    model_name="missing-model",
                    default_headers_json=None,
                )

    def test_ensure_runtime_layout_writes_openai_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime_root = Path(td)
            ensure_runtime_layout(
                runtime_root=runtime_root,
                api_key="sk-test",
                base_url="https://example.com/v1",
                default_headers_json='{"X-Test":"1"}',
            )
            env_text = (runtime_root / ".env").read_text(encoding="utf-8")
            self.assertIn("OPENAI_API_KEY=sk-test", env_text)
            self.assertIn("OPENAI_BASE_URL=https://example.com/v1", env_text)
            self.assertIn('OPENAI_DEFAULT_HEADERS={"X-Test":"1"}', env_text)

    def test_build_markdown_report_uses_generic_title(self) -> None:
        summary = {
            "categories": ["simple_python"],
            "cases_per_category": 3,
            "metrics_percent_point": {
                "Overall Acc": {"baseline": 10.0, "ralph": 12.0, "delta": 2.0},
            },
        }
        report = build_markdown_report(
            summary=summary,
            provider_name="OpenRouter",
            model_name="meta-llama/test",
            baseline_display="baseline",
            ralph_display="ralph",
            runtime_root=Path("/tmp/runtime"),
            run_ids_enabled=True,
        )
        self.assertIn("# OpenAI-Compatible Prompt-Mode BFCL Benchmark Report", report)
        self.assertIn("- Provider: `OpenRouter`", report)
        self.assertIn("| Overall Acc | 10.00 | 12.00 | +2.00 |", report)


if __name__ == "__main__":
    unittest.main()
