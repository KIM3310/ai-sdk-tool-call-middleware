#!/usr/bin/env python3
from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parent / "run_kiro_cli_prompt_bfcl_ralph.py"
)
MOD = runpy.run_path(str(SCRIPT_PATH))

validate_args = MOD["validate_args"]
resolve_cli_path = MOD["resolve_cli_path"]
strip_ansi = MOD["strip_ansi"]
sanitize_kiro_output = MOD["sanitize_kiro_output"]
build_kiro_chat_command = MOD["build_kiro_chat_command"]
ensure_runtime_layout = MOD["ensure_runtime_layout"]
check_kiro_login = MOD["check_kiro_login"]
build_markdown_report = MOD["build_markdown_report"]


class TestRunKiroCliPromptBfclRalph(unittest.TestCase):
    def test_validate_args_rejects_missing_model_or_cli_path(self) -> None:
        bad_cases = [
            SimpleNamespace(
                cases_per_category=1,
                num_threads=1,
                temperature=0.0,
                model_name="",
                provider_name="Kiro CLI",
                cli_path="kiro-cli",
                error_report_top_n=5,
                request_timeout_sec=30.0,
                max_step_limit=20,
            ),
            SimpleNamespace(
                cases_per_category=1,
                num_threads=1,
                temperature=0.0,
                model_name="kiro-cli-default",
                provider_name="Kiro CLI",
                cli_path=" ",
                error_report_top_n=5,
                request_timeout_sec=30.0,
                max_step_limit=20,
            ),
        ]

        for args in bad_cases:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit):
                    validate_args(args)

    def test_resolve_cli_path_and_command_building(self) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/kiro-cli"):
            self.assertEqual(resolve_cli_path("kiro-cli"), "/usr/local/bin/kiro-cli")

        command = build_kiro_chat_command(
            cli_path="/usr/local/bin/kiro-cli",
            prompt="Reply with OK",
            kiro_model="auto",
            kiro_agent="coder",
            trust_tools="",
        )
        self.assertEqual(command[:6], [
            "/usr/local/bin/kiro-cli",
            "chat",
            "--no-interactive",
            "--wrap",
            "never",
            "--trust-tools=",
        ])
        self.assertIn("--agent", command)
        self.assertIn("--model", command)
        self.assertEqual(command[-1], "Reply with OK")

    def test_sanitize_kiro_output_removes_ansi_and_spinner_lines(self) -> None:
        raw = "\x1b[?25l\r▰▱▱ Opening browser... | Press (^) + C to cancel\nAnswer line 1\n\nAnswer line 2\n"
        self.assertEqual(strip_ansi(raw).splitlines()[-2:], ["Answer line 1", "", "Answer line 2"][-2:])
        self.assertEqual(sanitize_kiro_output(raw), "Answer line 1\nAnswer line 2")

    def test_ensure_runtime_layout_writes_metadata_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime_root = Path(td)
            ensure_runtime_layout(
                runtime_root=runtime_root,
                cli_path="/usr/local/bin/kiro-cli",
                kiro_model="auto",
                kiro_agent="coder",
                trust_tools="",
            )
            env_text = (runtime_root / ".env").read_text(encoding="utf-8")
            self.assertIn("KIRO_CLI_PATH=/usr/local/bin/kiro-cli", env_text)
            self.assertIn("KIRO_MODEL=auto", env_text)
            self.assertIn("KIRO_AGENT=coder", env_text)
            self.assertIn("KIRO_TRUST_TOOLS=", env_text)

    def test_check_kiro_login_surfaces_not_logged_in(self) -> None:
        class _Proc:
            def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        with patch("subprocess.run", return_value=_Proc(0, stdout="kim@example.com\n")):
            self.assertEqual(check_kiro_login("/usr/local/bin/kiro-cli"), "kim@example.com")

        with patch("subprocess.run", return_value=_Proc(1, stdout="Not logged in\n")):
            with self.assertRaises(SystemExit):
                check_kiro_login("/usr/local/bin/kiro-cli")

    def test_build_markdown_report_uses_kiro_title(self) -> None:
        summary = {
            "categories": ["simple_python"],
            "cases_per_category": 3,
            "metrics_percent_point": {
                "Overall Acc": {"baseline": 10.0, "ralph": 12.0, "delta": 2.0},
            },
        }
        report = build_markdown_report(
            summary=summary,
            provider_name="Kiro CLI",
            model_name="kiro-cli-default",
            baseline_display="baseline",
            ralph_display="ralph",
            runtime_root=Path("/tmp/runtime"),
            run_ids_enabled=True,
        )
        self.assertIn("# Kiro CLI Prompt-Mode BFCL Benchmark Report", report)
        self.assertIn("- Provider: `Kiro CLI`", report)
        self.assertIn("| Overall Acc | 10.00 | 12.00 | +2.00 |", report)


if __name__ == "__main__":
    unittest.main()
