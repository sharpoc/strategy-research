from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest_data_catalog import build_strategy_data_inventory, build_strategy_inventory_json_safe
from research_backtest_utils import repo_root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local backtest data readiness without touching test/live environments.")
    parser.add_argument("--show-top", type=int, default=20, help="Rows to print for dataset / strategy tables.")
    parser.add_argument("--skip-export", action="store_true", help="Skip writing JSON / CSV outputs under output/research_backtests.")
    return parser.parse_args()


def export_root_dir() -> Path:
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    payload = build_strategy_data_inventory()
    dataset_table: pd.DataFrame = payload["dataset_table"]
    strategy_table: pd.DataFrame = payload["strategy_table"]

    print("===== dataset availability =====")
    if dataset_table.empty:
        print("(empty)")
    else:
        show_cols = [
            "dataset_id",
            "available",
            "coverage_start_date",
            "coverage_end_date",
            "coverage_trade_days",
            "coverage_years",
            "expected_trade_days_in_range",
            "missing_trade_days_in_range",
            "coverage_ratio_pct",
            "file_count",
            "dir_count",
        ]
        print(dataset_table[show_cols].head(args.show_top).to_string(index=False))

    print("===== strategy readiness =====")
    if strategy_table.empty:
        print("(empty)")
    else:
        show_cols = [
            "strategy_id",
            "strategy_name",
            "strategy_kind",
            "replay_ready",
            "api_replay_ready",
            "history_window_ready",
            "recommended_history_years",
            "market_daily_coverage_years",
            "market_daily_coverage_ratio_pct",
            "market_daily_missing_trade_days",
            "holder_snapshot_trade_days",
            "missing_required_dataset_ids",
        ]
        print(strategy_table[show_cols].head(args.show_top).to_string(index=False))

    export_dir = None
    if not args.skip_export:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = export_root_dir() / f"backtest_data_inventory_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)
        dataset_table.to_csv(export_dir / "dataset_table.csv", index=False)
        strategy_table.to_csv(export_dir / "strategy_table.csv", index=False)
        with (export_dir / "inventory.json").open("w", encoding="utf-8") as f:
            json.dump(build_strategy_inventory_json_safe(), f, ensure_ascii=False, indent=2)

    summary = {
        "repo_root": payload["repo_root"],
        "dataset_count": int(len(dataset_table)),
        "strategy_count": int(len(strategy_table)),
        "replay_ready_strategies": strategy_table[strategy_table["replay_ready"]]["strategy_id"].tolist() if not strategy_table.empty else [],
        "history_window_ready_strategies": strategy_table[strategy_table["history_window_ready"]]["strategy_id"].tolist() if not strategy_table.empty else [],
        "export_dir": str(export_dir) if export_dir is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
