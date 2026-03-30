from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from backtest_data_catalog import build_strategy_data_inventory
from backtest_strategy_registry import build_price_strategy_registry, supported_price_strategy_ids
from prepare_backtest_market_cache import prepare_market_cache
from run_price_strategy_regime_backtest import HISTORY_LOOKBACK_CALENDAR_MULTIPLIER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local backtest data for one or more strategies.")
    parser.add_argument("--strategy-ids", required=True, help="Comma-separated strategy ids, e.g. limitup_l1l2,double_bottom,holder_increase")
    parser.add_argument("--backtest-start-date", default="", help="Optional actual backtest start date YYYYMMDD; warmup history will be prepared before this date.")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"), help="End date YYYYMMDD.")
    parser.add_argument("--only-missing", action="store_true", help="Only fetch market daily dates not already cached.")
    parser.add_argument("--passes", type=int, default=3, help="How many missing-only passes to run for flaky interfaces.")
    parser.add_argument("--smoke-limit", type=int, default=0, help="Limit fetched trade dates for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "").strip()
    if not token:
        raise SystemExit("Missing TUSHARE_TOKEN in environment.")

    inventory = build_strategy_data_inventory()
    strategy_map = inventory["strategies"]
    requested_ids = [item.strip() for item in args.strategy_ids.split(",") if item.strip()]
    supported_ids = set(supported_price_strategy_ids()) | {"holder_increase"}
    invalid_ids = [strategy_id for strategy_id in requested_ids if strategy_id not in supported_ids]
    if invalid_ids:
        raise SystemExit(f"Unsupported strategy ids: {','.join(invalid_ids)}")

    selected_rows = [strategy_map[strategy_id] for strategy_id in requested_ids if strategy_id in strategy_map]
    if not selected_rows:
        raise SystemExit("No valid strategies selected.")

    target_years = max(float(row["recommended_history_years"]) for row in selected_rows)
    registry = build_price_strategy_registry()
    selected_price_specs = [registry[strategy_id] for strategy_id in requested_ids if strategy_id in registry]
    max_history_bars = max((spec.history_bars for spec in selected_price_specs), default=0)
    warmup_calendar_days = max(90, int(max_history_bars * HISTORY_LOOKBACK_CALENDAR_MULTIPLIER)) if max_history_bars > 0 else 0

    history_window_start_date = (pd.Timestamp(args.end_date) - pd.Timedelta(days=int(round(target_years * 365.25)))).strftime("%Y%m%d")
    if args.backtest_start_date:
        warmup_start_date = (pd.Timestamp(args.backtest_start_date) - pd.Timedelta(days=warmup_calendar_days)).strftime("%Y%m%d")
        start_date = min(history_window_start_date, warmup_start_date)
    else:
        warmup_start_date = ""
        start_date = history_window_start_date
    market_summary = prepare_market_cache(
        token=token,
        custom_http_url=custom_http_url,
        start_date=start_date,
        end_date=args.end_date,
        only_missing=args.only_missing,
        passes=args.passes,
        smoke_limit=args.smoke_limit,
    )
    refreshed_inventory = build_strategy_data_inventory()

    summary = {
        "requested_strategy_ids": requested_ids,
        "target_years": target_years,
        "history_window_start_date": history_window_start_date,
        "backtest_start_date": args.backtest_start_date or None,
        "warmup_calendar_days": warmup_calendar_days,
        "warmup_start_date": warmup_start_date or None,
        "start_date": start_date,
        "end_date": args.end_date,
        "market_cache_summary": market_summary,
        "strategy_readiness": {
            strategy_id: refreshed_inventory["strategies"].get(strategy_id)
            for strategy_id in requested_ids
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
