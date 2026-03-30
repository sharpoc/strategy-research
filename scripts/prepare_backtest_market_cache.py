from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from backtest_data_catalog import scan_dataset_availability
from research_backtest_utils import (
    configure_tushare_client,
    discover_cached_trade_dates,
    fetch_market_daily_history,
    fetch_stock_basic_all,
    get_open_trade_dates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local market cache for backtests without touching test/live environments.")
    parser.add_argument("--start-date", default="", help="Start date YYYYMMDD. Omit to use --years-back from end date.")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"), help="End date YYYYMMDD.")
    parser.add_argument("--years-back", type=float, default=2.5, help="Used only when --start-date is omitted.")
    parser.add_argument("--only-missing", action="store_true", help="Only fetch trade dates not already cached.")
    parser.add_argument("--passes", type=int, default=3, help="How many missing-only passes to run for flaky interfaces.")
    parser.add_argument("--smoke-limit", type=int, default=0, help="Limit to the first N trade dates for smoke testing.")
    return parser.parse_args()


def _resolve_start_date(start_date: str, end_date: str, years_back: float) -> str:
    if start_date:
        return start_date
    end_ts = pd.Timestamp(end_date)
    start_ts = end_ts - pd.Timedelta(days=int(round(years_back * 365.25)))
    return start_ts.strftime("%Y%m%d")


def _build_weekday_fallback_dates(start_date: str, end_date: str) -> list[str]:
    return pd.date_range(start=start_date, end=end_date, freq="B").strftime("%Y%m%d").tolist()


def prepare_market_cache(
    token: str,
    custom_http_url: str,
    start_date: str,
    end_date: str,
    only_missing: bool = True,
    passes: int = 1,
    smoke_limit: int = 0,
) -> dict:
    pro = configure_tushare_client(token=token, custom_http_url=custom_http_url)
    stock_basic_all = fetch_stock_basic_all(pro)
    cached_before = discover_cached_trade_dates(start_date=start_date, end_date=end_date)
    requested_trade_dates = 0
    pass_rows: list[dict] = []
    for pass_index in range(max(int(passes), 1)):
        trade_dates = get_open_trade_dates(pro, start_date=start_date, end_date=end_date)
        if not trade_dates or (len(trade_dates) <= len(cached_before) and (start_date not in trade_dates or end_date not in trade_dates)):
            weekday_fallback = _build_weekday_fallback_dates(start_date, end_date)
            trade_dates = sorted({*trade_dates, *weekday_fallback})
        requested_trade_dates = len(trade_dates)
        trade_dates_to_fetch = list(trade_dates)
        if only_missing:
            cached_date_set = set(cached_before)
            trade_dates_to_fetch = [trade_date for trade_date in trade_dates if trade_date not in cached_date_set]
        if smoke_limit > 0:
            trade_dates_to_fetch = trade_dates_to_fetch[: smoke_limit]
        pass_rows.append(
            {
                "pass_index": pass_index + 1,
                "cached_before": len(cached_before),
                "requested_fetch_dates": len(trade_dates_to_fetch),
            }
        )
        if not trade_dates_to_fetch:
            break
        fetch_market_daily_history(pro, trade_dates=trade_dates_to_fetch, sleep_sec=0.0)
        cached_after_pass = discover_cached_trade_dates(start_date=start_date, end_date=end_date)
        pass_rows[-1]["cached_after"] = len(cached_after_pass)
        if len(cached_after_pass) <= len(cached_before):
            cached_before = cached_after_pass
            break
        cached_before = cached_after_pass

    cached_after = discover_cached_trade_dates(start_date=start_date, end_date=end_date)
    daily_inventory = scan_dataset_availability().get("market_daily_all", {})

    return {
        "start_date": start_date,
        "end_date": end_date,
        "requested_trade_dates": requested_trade_dates,
        "cached_before": pass_rows[0]["cached_before"] if pass_rows else len(cached_before),
        "requested_fetch_dates": int(sum(int(row.get("requested_fetch_dates", 0) or 0) for row in pass_rows)),
        "cached_after": len(cached_after),
        "passes_requested": int(max(int(passes), 1)),
        "passes_executed": int(len(pass_rows)),
        "pass_rows": pass_rows,
        "stock_basic_rows": int(len(stock_basic_all)),
        "market_daily_coverage_start_date": daily_inventory.get("coverage_start_date"),
        "market_daily_coverage_end_date": daily_inventory.get("coverage_end_date"),
        "market_daily_coverage_years": daily_inventory.get("coverage_years"),
        "market_daily_coverage_trade_days": daily_inventory.get("coverage_trade_days"),
        "market_daily_missing_trade_days_in_range": daily_inventory.get("missing_trade_days_in_range"),
        "market_daily_coverage_ratio_pct": daily_inventory.get("coverage_ratio_pct"),
    }


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "").strip()
    if not token:
        raise SystemExit("Missing TUSHARE_TOKEN in environment.")

    start_date = _resolve_start_date(args.start_date, args.end_date, args.years_back)
    summary = prepare_market_cache(
        token=token,
        custom_http_url=custom_http_url,
        start_date=start_date,
        end_date=args.end_date,
        only_missing=args.only_missing,
        passes=args.passes,
        smoke_limit=args.smoke_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
