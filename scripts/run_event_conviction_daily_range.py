from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from holder_strategy_core import configure_tushare_client, ensure_token, safe_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run event_conviction_signal one trade date at a time using the single-day live path."
    )
    parser.add_argument("--start-date", required=True, help="Start trade date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End trade date YYYYMMDD.")
    parser.add_argument("--config-file", required=True, help="JSON config for EventConvictionConfig overrides.")
    parser.add_argument("--export-root", required=True, help="Directory for event_conviction_screen_<date> exports.")
    parser.add_argument("--report-root", default="", help="Directory for batch reports.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip dates with existing export dirs.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch after first failure.")
    parser.add_argument("--max-trade-days", type=int, default=0, help="Limit trade days for smoke tests.")
    parser.add_argument("--between-day-sleep-sec", type=float, default=1.0, help="Pause between trade dates.")
    parser.add_argument("--api-sleep-sec", type=float, default=0.12, help="Sleep between API calls.")
    return parser.parse_args()


def report_root_dir(path_str: str = "") -> Path:
    if path_str.strip():
        path = Path(path_str).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parent.parent / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_open_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    cal = safe_call(
        "trade_cal_event_daily_range",
        getattr(pro, "trade_cal", None),
        start_date=start_date,
        end_date=end_date,
        is_open="1",
    )
    if cal.empty:
        raise RuntimeError(f"trade_cal returned empty for {start_date}~{end_date}")
    date_col = "cal_date" if "cal_date" in cal.columns else "trade_date"
    return sorted(cal[date_col].dropna().astype(str).unique().tolist())


def export_dir_for_trade_date(base_dir: Path, trade_date: str) -> Path:
    return base_dir / f"event_conviction_screen_{trade_date}"


def export_is_complete(base_dir: Path, trade_date: str) -> bool:
    export_dir = export_dir_for_trade_date(base_dir, trade_date)
    required = [
        export_dir / "event_candidates.csv",
        export_dir / "scored_candidates.csv",
        export_dir / "best_pick_candidate.csv",
        export_dir / "screen_summary.json",
    ]
    return export_dir.exists() and all(path.exists() for path in required)


def run_single_trade_date(
    trade_date: str,
    export_root: Path,
    config_file: Path,
    api_sleep_sec: float,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve().parent / "run_tushare_event_conviction_strategy.py"
    command = [
        sys.executable,
        str(script_path),
        "--end-date",
        trade_date,
        "--config-file",
        str(config_file),
        "--export-root",
        str(export_root),
        "--api-sleep-sec",
        str(api_sleep_sec),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    export_dir = export_dir_for_trade_date(export_root, trade_date)
    if completed.returncode != 0:
        return {
            "ok": False,
            "trade_date": trade_date,
            "error": completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}",
        }

    summary_path = export_dir / "screen_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "trade_date": trade_date,
        "export_dir": str(export_dir),
        "best_pick_ts_code": summary.get("best_pick_ts_code"),
        "best_pick_name": summary.get("best_pick_name"),
        "latest_trade_date": summary.get("latest_trade_date"),
    }


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    ensure_token(token)

    export_root = Path(args.export_root).expanduser().resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    report_root = report_root_dir(args.report_root)
    config_file = Path(args.config_file).expanduser().resolve()
    if not config_file.exists():
        raise SystemExit(f"config file not found: {config_file}")

    pro = configure_tushare_client(token, custom_http_url=custom_http_url)
    trade_dates = get_open_trade_dates(pro, args.start_date, args.end_date)
    if args.max_trade_days > 0:
        trade_dates = trade_dates[: args.max_trade_days]
    if not trade_dates:
        raise SystemExit("No trade dates found in the requested range.")

    run_tag = f"event_conviction_daily_range_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    batch_dir = report_root / run_tag
    batch_dir.mkdir(parents=True, exist_ok=True)

    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for idx, trade_date in enumerate(trade_dates, start=1):
        if args.skip_existing and export_is_complete(export_root, trade_date):
            skipped_rows.append({"trade_date": trade_date, "status": "skipped_existing"})
            print(f"[event-daily-range] {idx}/{len(trade_dates)} trade_date={trade_date} skipped existing", flush=True)
            continue
        print(f"[event-daily-range] {idx}/{len(trade_dates)} trade_date={trade_date} start", flush=True)
        row = run_single_trade_date(
            trade_date=trade_date,
            export_root=export_root,
            config_file=config_file,
            api_sleep_sec=args.api_sleep_sec,
        )
        if row["ok"]:
            success_rows.append(row)
            print(f"[event-daily-range] trade_date={trade_date} ok best_pick={row.get('best_pick_ts_code') or '(empty)'}", flush=True)
        else:
            failure_rows.append(row)
            print(f"[event-daily-range] trade_date={trade_date} failed error={row.get('error')}", flush=True)
            if args.stop_on_error:
                break
        if args.between_day_sleep_sec > 0 and idx < len(trade_dates):
            time.sleep(args.between_day_sleep_sec)

    pd.DataFrame(success_rows).to_csv(batch_dir / "success_rows.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(batch_dir / "failure_rows.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(batch_dir / "skipped_rows.csv", index=False)

    summary = {
        "run_tag": run_tag,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "config_file": str(config_file),
        "export_root": str(export_root),
        "report_dir": str(batch_dir),
        "total_trade_dates": len(trade_dates),
        "success_count": len(success_rows),
        "failure_count": len(failure_rows),
        "skipped_count": len(skipped_rows),
        "failed_trade_dates": [row["trade_date"] for row in failure_rows],
        "skipped_trade_dates": [row["trade_date"] for row in skipped_rows],
    }
    (batch_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
