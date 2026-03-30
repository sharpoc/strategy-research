from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from holder_strategy_core import (
    HolderStrategyConfig,
    configure_tushare_client,
    ensure_token,
    json_safe,
    log_step,
    output_root_dir,
    retry_sleep_seconds,
    run_holder_strategy_screening,
    safe_call,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the holder strategy one trade date at a time, using the same single-day path as the live screening flow."
    )
    parser.add_argument("--start-date", required=True, help="Start trade date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End trade date YYYYMMDD.")
    parser.add_argument("--ann-start-date", default="", help="Optional fixed announcement window start date YYYYMMDD.")
    parser.add_argument("--config-file", default="", help="Optional JSON file with HolderStrategyConfig overrides.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON object with HolderStrategyConfig overrides.")
    parser.add_argument("--export-root", default="", help="Directory for holder_increase_screen_<date> exports.")
    parser.add_argument("--report-root", default="", help="Directory for batch reports.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip dates whose export directory already looks complete.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the batch after the first failed trade date.")
    parser.add_argument("--max-trade-days", type=int, default=0, help="Limit trade days for smoke tests.")
    parser.add_argument("--per-day-retries", type=int, default=1, help="Extra retries for an entire trade-date run.")
    parser.add_argument("--no-resume-existing", action="store_true", help="Disable resuming from partial per-day export files.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow a trade date to finish with partial deep-dive data.")
    parser.add_argument("--between-day-sleep-sec", type=float, default=1.0, help="Pause between trade dates.")
    parser.add_argument(
        "--api-sleep-sec",
        type=float,
        default=0.25,
        help="Execution pacing between API calls. Does not change scoring logic.",
    )
    parser.add_argument(
        "--cyq-sleep-sec",
        type=float,
        default=0.25,
        help="Execution pacing for stage2 cyq calls. Does not change scoring logic.",
    )
    return parser.parse_args()


def load_json_file(path_str: str) -> dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a JSON object: {path}")
    return data


def load_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    file_overrides = load_json_file(args.config_file)
    if file_overrides:
        config.update(file_overrides)
    if args.config_json.strip():
        try:
            inline = json.loads(args.config_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --config-json: {exc}") from exc
        if not isinstance(inline, dict):
            raise SystemExit("--config-json must be a JSON object.")
        config.update(inline)
    config["api_sleep_sec"] = float(args.api_sleep_sec)
    config["cyq_sleep_sec"] = float(args.cyq_sleep_sec)
    return config


def report_root_dir(path_str: str = "") -> Path:
    if path_str.strip():
        path = Path(path_str).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parent.parent / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_root_dir(path_str: str = "") -> Path:
    if path_str.strip():
        path = Path(path_str).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return output_root_dir()


def export_dir_for_trade_date(base_dir: Path, trade_date: str) -> Path:
    return base_dir / f"holder_increase_screen_{trade_date}"


def export_is_complete(base_dir: Path, trade_date: str) -> bool:
    export_dir = export_dir_for_trade_date(base_dir, trade_date)
    required = [
        export_dir / "candidate_base.csv",
        export_dir / "deep_metrics_stage1.csv",
        export_dir / "reranked_candidates_stage2.csv",
        export_dir / "best_pick_candidate.csv",
        export_dir / "screen_summary.json",
    ]
    return export_dir.exists() and all(path.exists() for path in required)


def get_open_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    cal = safe_call(
        "trade_cal_daily_range",
        getattr(pro, "trade_cal", None),
        start_date=start_date,
        end_date=end_date,
        is_open="1",
    )
    if cal.empty:
        raise RuntimeError(f"trade_cal returned empty for {start_date}~{end_date}")
    date_col = "cal_date" if "cal_date" in cal.columns else "trade_date"
    return sorted(cal[date_col].dropna().astype(str).unique().tolist())


def run_single_trade_date(
    pro,
    trade_date: str,
    export_root: Path,
    ann_start_date: str,
    config_overrides: dict[str, Any],
    retries: int,
    resume_existing: bool,
    require_complete: bool,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            config = HolderStrategyConfig.for_end_date(trade_date, ann_start_date=ann_start_date, **config_overrides)
            result = run_holder_strategy_screening(
                config=config,
                pro=pro,
                custom_http_url=os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip(),
                export_results=True,
                export_root=export_root,
                resume_existing=resume_existing,
                require_complete=require_complete,
            )
            best_pick = result["best_pick_candidate"]
            return {
                "ok": True,
                "trade_date": trade_date,
                "market_regime": result["screen_summary"].get("market_regime"),
                "today_direction": result["screen_summary"].get("today_direction"),
                "best_pick_ts_code": None if best_pick.empty else str(best_pick.iloc[0].get("ts_code", "")),
                "best_pick_name": None if best_pick.empty else str(best_pick.iloc[0].get("name", "")),
                "export_dir": str(result["export_dir"]),
            }
        except Exception as exc:
            last_error = repr(exc)
            log_step(f"[daily-range] trade_date={trade_date} attempt={attempt + 1} failed error={exc}")
            if attempt < retries:
                time.sleep(retry_sleep_seconds(str(exc), attempt))
    return {
        "ok": False,
        "trade_date": trade_date,
        "error": last_error or "unknown error",
    }


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    ensure_token(token)
    config_overrides = load_config_overrides(args)
    export_root = export_root_dir(args.export_root)
    report_root = report_root_dir(args.report_root)
    pro = configure_tushare_client(token, custom_http_url=custom_http_url)
    trade_dates = get_open_trade_dates(pro, args.start_date, args.end_date)
    if args.max_trade_days > 0:
        trade_dates = trade_dates[: args.max_trade_days]
    if not trade_dates:
        raise SystemExit("No trade dates found in the requested range.")

    run_tag = f"holder_daily_range_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    batch_dir = report_root / run_tag
    batch_dir.mkdir(parents=True, exist_ok=True)

    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for idx, trade_date in enumerate(trade_dates, start=1):
        if args.skip_existing and export_is_complete(export_root, trade_date):
            row = {"trade_date": trade_date, "status": "skipped_existing"}
            skipped_rows.append(row)
            log_step(f"[daily-range] {idx}/{len(trade_dates)} trade_date={trade_date} skipped existing export")
            continue

        log_step(f"[daily-range] {idx}/{len(trade_dates)} trade_date={trade_date} start")
        row = run_single_trade_date(
            pro=pro,
            trade_date=trade_date,
            export_root=export_root,
            ann_start_date=args.ann_start_date,
            config_overrides=config_overrides,
            retries=max(0, args.per_day_retries),
            resume_existing=not args.no_resume_existing,
            require_complete=not args.allow_partial,
        )
        if row["ok"]:
            success_rows.append(row)
            log_step(
                f"[daily-range] trade_date={trade_date} ok best_pick={row.get('best_pick_ts_code') or '(empty)'}"
            )
        else:
            failure_rows.append(row)
            log_step(f"[daily-range] trade_date={trade_date} failed error={row.get('error')}")
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
        "export_root": str(export_root),
        "report_dir": str(batch_dir),
        "total_trade_dates": len(trade_dates),
        "success_count": len(success_rows),
        "failure_count": len(failure_rows),
        "skipped_count": len(skipped_rows),
        "failed_trade_dates": [row["trade_date"] for row in failure_rows],
        "skipped_trade_dates": [row["trade_date"] for row in skipped_rows],
        "config_overrides": json_safe(config_overrides),
    }
    with (batch_dir / "batch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
