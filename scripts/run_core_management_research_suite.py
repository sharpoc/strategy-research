from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from research_backtest_utils import repo_root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stage1-first research suite for core-management accumulation."
    )
    parser.add_argument("--start-date", default="20250925", help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", default="20260324", help="End date in YYYYMMDD.")
    parser.add_argument("--hold-days", default="3,5,10", help="Holding windows.")
    parser.add_argument("--export-root", default="", help="Optional export root.")
    parser.add_argument("--api-sleep-sec", type=float, default=0.12, help="Sleep between API calls.")
    return parser.parse_args()


def export_root_dir(path_str: str = "") -> Path:
    if path_str.strip():
        path = Path(path_str).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(args: list[str], env: dict[str, str]) -> tuple[dict[str, Any], str]:
    result = subprocess.run(args, check=True, capture_output=True, text=True, env=env)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] | None = None
    export_dir = ""
    json_lines: list[str] = []
    collecting_json = False
    for line in reversed(lines):
        if line.startswith("export_dir="):
            export_dir = line.split("=", 1)[1].strip()
            continue
        if payload is None and line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                pass
        if line == "}":
            collecting_json = True
        if collecting_json:
            json_lines.append(line)
            if line.startswith("{"):
                try:
                    payload = json.loads("\n".join(reversed(json_lines)))
                    break
                except Exception:
                    collecting_json = False
                    json_lines = []
    if payload is None:
        raise SystemExit(f"Failed to parse JSON payload from command: {' '.join(args)}")
    return payload, export_dir


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_suite_report(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# 核心高管连增臻选研究套件")
    lines.append("")
    lines.append("| 场景 | Stage1样本 | 唯一股票 | 3日 | 5日 | 10日 | Final信号 | Final 5日 | Final 10日 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        stage1 = row.get("stage1_summary", {})
        final = row.get("final_summary", {})
        lines.append(
            f"| {row.get('label')} | {stage1.get('rows')} | {row.get('stage1_unique_stock_count')} | "
            f"{stage1.get('avg_3d_pct')} | {stage1.get('avg_5d_pct')} | {stage1.get('avg_10d_pct')} | "
            f"{final.get('rows')} | {final.get('avg_5d_pct')} | {final.get('avg_10d_pct')} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = repo_root_dir()
    export_dir = export_root_dir(args.export_root) / f"core_management_research_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    scenarios = [
        {"label": "baseline", "config_file": ""},
        {"label": "stage1_relaxed_v1", "config_file": str(repo_root / "configs" / "core_management_stage1_relaxed_v1.json")},
        {"label": "stage1_relaxed_v2_cost", "config_file": str(repo_root / "configs" / "core_management_stage1_relaxed_v2_cost.json")},
    ]

    suite_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        stage1_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_core_management_stage1_research.py"),
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--hold-days",
            args.hold_days,
            "--export-root",
            str(export_dir / scenario["label"]),
            "--api-sleep-sec",
            str(args.api_sleep_sec),
        ]
        if scenario["config_file"]:
            stage1_cmd.extend(["--config-file", scenario["config_file"]])
        stage1_payload, stage1_export_dir = run_command(stage1_cmd, env)

        final_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_core_management_final_review.py"),
            "--stats-json",
            str(Path(stage1_export_dir) / "stage1_summary.json"),
            "--hold-days",
            args.hold_days,
            "--export-root",
            str(export_dir / scenario["label"]),
            "--api-sleep-sec",
            str(args.api_sleep_sec),
        ]
        if scenario["config_file"]:
            final_cmd.extend(["--config-file", scenario["config_file"]])
        final_payload, final_export_dir = run_command(final_cmd, env)

        final_summary = final_payload.get("optimized_final_summary", {})
        if not final_summary and final_export_dir:
            review_summary_path = Path(final_export_dir) / "review_summary.json"
            if review_summary_path.exists():
                final_summary = load_json(review_summary_path).get("optimized_final_summary", {})

        suite_rows.append(
            {
                "label": scenario["label"],
                "config_file": scenario["config_file"],
                "stage1_summary": stage1_payload.get("stage1_summary", {}),
                "stage1_unique_stock_count": stage1_payload.get("stage1_unique_stock_count", 0),
                "stage1_export_dir": stage1_export_dir,
                "final_summary": final_summary,
                "final_export_dir": final_export_dir,
            }
        )

    summary_payload = {
        "strategy_id": "core_management_accumulation",
        "strategy_name": "核心高管连增臻选",
        "range": {"start_date": args.start_date, "end_date": args.end_date},
        "hold_days": args.hold_days,
        "scenarios": suite_rows,
        "export_dir": str(export_dir),
    }
    (export_dir / "suite_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (export_dir / "suite_report.md").write_text(build_suite_report(suite_rows), encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
