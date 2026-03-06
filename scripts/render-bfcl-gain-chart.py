#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a simple SVG chart from a BFCL prompt-mode summary.json file."
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--subtitle", type=str, required=True)
    parser.add_argument("--source-label", type=str, required=True)
    parser.add_argument("--baseline-label", type=str, default="Baseline")
    parser.add_argument("--ralph-label", type=str, default="RALPH Loop")
    return parser.parse_args()


def load_overall_metrics(summary_path: Path) -> tuple[float, float, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    overall = payload["metrics_percent_point"]["Overall Acc"]
    baseline = float(overall["baseline"])
    ralph = float(overall["ralph"])
    delta = float(overall["delta"])
    return baseline, ralph, delta


def render_svg(
    *,
    baseline: float,
    ralph: float,
    delta: float,
    title: str,
    subtitle: str,
    source_label: str,
    baseline_label: str,
    ralph_label: str,
) -> str:
    chart_max = max(10.0, baseline, ralph)
    axis_top = 180.0
    axis_bottom = 590.0
    axis_height = axis_bottom - axis_top

    def bar_y(value: float) -> float:
        return axis_bottom - ((value / chart_max) * axis_height)

    def bar_height(value: float) -> float:
        return axis_bottom - bar_y(value)

    baseline_y = bar_y(baseline)
    ralph_y = bar_y(ralph)
    baseline_height = bar_height(baseline)
    ralph_height = bar_height(ralph)

    relative = 0.0 if baseline == 0 else (delta / baseline) * 100.0
    badge_fill = "#d9f2e3" if delta > 0 else "#eef2f7" if delta == 0 else "#fde8e8"
    badge_stroke = "#74c69d" if delta > 0 else "#9fb3c8" if delta == 0 else "#f08c8c"
    badge_text = "#1b4332" if delta > 0 else "#334e68" if delta == 0 else "#9b2226"
    baseline_fill = "#6c7a89"
    ralph_fill = "#2f9e44" if delta > 0 else "#7f8c8d" if delta == 0 else "#c92a2a"

    ticks = []
    for tick in range(0, int(chart_max) + 1, 2):
        if tick > chart_max:
            break
        y = axis_bottom - ((tick / chart_max) * axis_height)
        ticks.append(
            f'<text x="76" y="{y + 5:.0f}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="16" fill="#486581">{tick}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720" role="img" aria-label="{title}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f8fbff"/>
      <stop offset="100%" stop-color="#edf4ff"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="1200" height="720" fill="url(#bg)"/>

  <text x="80" y="88" font-family="Arial, Helvetica, sans-serif" font-size="42" font-weight="700" fill="#102a43">
    {title}
  </text>
  <text x="80" y="128" font-family="Arial, Helvetica, sans-serif" font-size="24" fill="#334e68">
    {subtitle}
  </text>

  <line x1="120" y1="{axis_bottom:.0f}" x2="1080" y2="{axis_bottom:.0f}" stroke="#9fb3c8" stroke-width="2"/>
  <line x1="120" y1="{axis_top:.0f}" x2="120" y2="{axis_bottom:.0f}" stroke="#9fb3c8" stroke-width="2"/>

  {''.join(ticks)}

  <rect x="300" y="{baseline_y:.2f}" width="220" height="{baseline_height:.2f}" rx="12" fill="{baseline_fill}"/>
  <text x="410" y="{baseline_y - 10:.0f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="30" font-weight="700" fill="#334e68">{baseline:.2f}</text>
  <text x="410" y="630" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#334e68">{baseline_label}</text>

  <rect x="700" y="{ralph_y:.2f}" width="220" height="{ralph_height:.2f}" rx="12" fill="{ralph_fill}"/>
  <text x="810" y="{ralph_y - 10:.0f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="30" font-weight="700" fill="{badge_text}">{ralph:.2f}</text>
  <text x="810" y="630" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="{badge_text}">{ralph_label}</text>

  <rect x="430" y="150" width="380" height="58" rx="29" fill="{badge_fill}" stroke="{badge_stroke}"/>
  <text x="620" y="188" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="26" font-weight="700" fill="{badge_text}">
    {delta:+.2f}pp | {relative:+.1f}% relative
  </text>

  <text x="80" y="682" font-family="Arial, Helvetica, sans-serif" font-size="18" fill="#627d98">
    Source: {source_label}
  </text>
</svg>
"""


def main() -> int:
    args = parse_args()
    baseline, ralph, delta = load_overall_metrics(args.summary_json)
    svg = render_svg(
        baseline=baseline,
        ralph=ralph,
        delta=delta,
        title=args.title,
        subtitle=args.subtitle,
        source_label=args.source_label,
        baseline_label=args.baseline_label,
        ralph_label=args.ralph_label,
    )
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(svg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
