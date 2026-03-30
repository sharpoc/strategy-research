from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from holder_replay_utils import (
    build_holder_daily_results,
    export_root_dir,
    load_export_snapshots,
    load_price_context,
    write_replay_summary,
)
from optimize_price_strategy_params import evaluate_backtest_frame
from research_config_presets import exit_config_preset_help, load_exit_config_with_preset
from research_backtest_utils import json_safe, log_step
from run_price_strategy_regime_backtest import summarize_results
from strategy_exit_rules import summarize_exit_reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay holder-strategy daily snapshots with the same T+1-open and exit-rule semantics used in local research."
    )
    parser.add_argument("--start-date", required=True, help="Replay start date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Replay end date YYYYMMDD.")
    parser.add_argument("--hold-days", default="1,3,5,10", help="Comma-separated forward horizons for reporting.")
    parser.add_argument("--config-file", default="", help="Optional JSON file with HolderStrategyConfig override keys.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON object with HolderStrategyConfig override keys.")
    parser.add_argument(
        "--exit-config-preset",
        default="",
        help=f"Named exit preset. Available: {exit_config_preset_help()}",
    )
    parser.add_argument("--exit-config-file", default="", help="Optional JSON sell-rule override file.")
    parser.add_argument("--snapshot-root", default="", help="Optional root directory containing holder_increase_screen_<date> snapshots.")
    parser.add_argument("--max-trade-days", type=int, default=0, help="Limit replay to the most recent N snapshot trade days.")
    parser.add_argument("--min-trades", type=int, default=3, help="Soft minimum trade count for summary scoring.")
    parser.add_argument("--show-top", type=int, default=20, help="Rows to print from daily replay results.")
    return parser.parse_args()


def load_json_file(path_str: str) -> dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a JSON object: {path}")
    return data


def load_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config_file:
        config.update(load_json_file(args.config_file))
    if args.config_json.strip():
        try:
            inline = json.loads(args.config_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --config-json: {exc}") from exc
        if not isinstance(inline, dict):
            raise SystemExit("--config-json must be a JSON object.")
        config.update(inline)
    return config


def main() -> None:
    args = parse_args()
    hold_days = sorted({int(item.strip()) for item in args.hold_days.split(",") if item.strip()})
    config_overrides = load_config_overrides(args)
    exit_config, exit_config_meta = load_exit_config_with_preset(
        preset_name=args.exit_config_preset,
        config_file=args.exit_config_file,
    )

    snapshots = load_export_snapshots(
        start_date=args.start_date,
        end_date=args.end_date,
        snapshot_root=args.snapshot_root,
        max_trade_days=args.max_trade_days,
    )
    if not snapshots:
        raise SystemExit("No holder snapshots found in the requested range.")

    signal_trade_dates = [snapshot["trade_date"] for snapshot in snapshots]
    log_step(f"holder replay snapshots={len(snapshots)} range={signal_trade_dates[0]}~{signal_trade_dates[-1]}")

    market_daily_history, forward_table, price_path_map, history_trade_dates = load_price_context(
        signal_start_date=signal_trade_dates[0],
        hold_days=hold_days,
    )
    if forward_table.empty or not price_path_map:
        raise SystemExit("No cached market daily history available for holder replay.")

    daily_results = build_holder_daily_results(
        snapshots=snapshots,
        forward_table=forward_table,
        price_path_map=price_path_map,
        config_overrides=config_overrides,
        exit_config=exit_config,
        apply_exit=True,
    )
    strategy_summary, regime_summary, monthly_summary = summarize_results(daily_results, hold_days)
    exit_reason_summary = summarize_exit_reasons(daily_results)
    overall_metrics = evaluate_backtest_frame(daily_results, regime_filter="", min_trades=args.min_trades)

    run_tag = f"holder_replay_backtest_{signal_trade_dates[0]}_{signal_trade_dates[-1]}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = export_root_dir() / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    daily_results.to_csv(export_dir / "daily_results.csv", index=False)
    strategy_summary.to_csv(export_dir / "strategy_summary.csv", index=False)
    regime_summary.to_csv(export_dir / "regime_summary.csv", index=False)
    monthly_summary.to_csv(export_dir / "monthly_summary.csv", index=False)
    exit_reason_summary.to_csv(export_dir / "exit_reason_summary.csv", index=False)

    summary = {
        "mode": "holder_snapshot_replay_backtest",
        "range": [signal_trade_dates[0], signal_trade_dates[-1]],
        "requested_range": [args.start_date, args.end_date],
        "snapshot_count": int(len(snapshots)),
        "history_trade_dates": [history_trade_dates[0], history_trade_dates[-1]] if history_trade_dates else None,
        "hold_days": hold_days,
        "min_trades": args.min_trades,
        "config_overrides": json_safe(config_overrides),
        "exit_config_preset": exit_config_meta["exit_config_preset"],
        "exit_config_file": exit_config_meta["exit_config_file"],
        "preset_exit_config_file": exit_config_meta["preset_exit_config_file"],
        "overall_metrics": json_safe(overall_metrics),
        "export_dir": str(export_dir.resolve()),
    }
    write_replay_summary(export_dir / "summary.json", summary)

    signal_rows = daily_results[daily_results["has_signal"]].copy()
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print("===== strategy summary =====")
    print(strategy_summary.to_string(index=False) if not strategy_summary.empty else "(empty)")
    print("===== daily results =====")
    print(signal_rows.head(args.show_top).to_string(index=False) if not signal_rows.empty else "(empty)")
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
