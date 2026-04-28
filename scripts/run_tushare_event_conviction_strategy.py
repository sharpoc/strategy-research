from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from event_conviction_strategy import (
    EVENT_STRATEGY_NAME,
    EventConvictionConfig,
    apply_bottom_filters,
    build_buyback_event_candidates,
    build_holdertrade_event_candidates,
    build_screen_summary,
    display_columns,
    enrich_market_features,
    score_candidates,
    select_top1_candidate,
)
from holder_strategy_core import (
    build_market_snapshot,
    configure_tushare_client,
    ensure_columns,
    ensure_token,
    fetch_holdertrade_events,
    fetch_latest_complete_market_inputs,
    fetch_stock_basic_all,
    get_recent_open_trade_dates,
    output_root_dir,
    safe_call,
    write_csv_checkpoint,
    write_json_checkpoint,
)
from run_core_management_final_review import fetch_full_price_bundle
from run_tushare_core_management_accumulation_strategy import fetch_margin_detail_summary


def log_step(message: str) -> None:
    print(f"[event_conviction] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the event-conviction strategy and keep only the Top-1 stock.")
    parser.add_argument("--end-date", default="", help="Requested end date in YYYYMMDD. Default: today with 20:00 cutoff.")
    parser.add_argument("--ann-start-date", default="", help="Optional fixed announcement window start date in YYYYMMDD.")
    parser.add_argument("--show-top", type=int, default=20, help="Rows to print from the scored candidates.")
    parser.add_argument("--config-file", default="", help="Optional JSON file with EventConvictionConfig overrides.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON overrides.")
    parser.add_argument("--export-root", default="", help="Optional export directory.")
    parser.add_argument("--api-sleep-sec", type=float, default=0.15, help="Sleep between API calls.")
    return parser.parse_args()


def load_json_file(path_str: str) -> dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a JSON object: {path}")
    return data


def load_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    overrides.update(load_json_file(args.config_file))
    if args.config_json.strip():
        inline = json.loads(args.config_json)
        if not isinstance(inline, dict):
            raise SystemExit("--config-json must be a JSON object.")
        overrides.update(inline)
    return overrides


def export_root_dir(path_str: str = "") -> Path:
    if path_str.strip():
        path = Path(path_str).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return output_root_dir()


def fetch_buyback_events(
    pro,
    start_date: str,
    end_date: str,
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    return safe_call(
        f"repurchase_{start_date}_{end_date}",
        getattr(pro, "repurchase", None),
        sleep_sec=sleep_sec,
        start_date=start_date,
        end_date=end_date,
    )


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    ensure_token(token)

    overrides = load_config_overrides(args)
    now_ts = pd.Timestamp.now()
    today_str = now_ts.strftime("%Y%m%d")
    requested_end_date = str(args.end_date or "").strip() or today_str
    screen_end_date = requested_end_date if args.end_date else (today_str if now_ts.hour >= 20 else (now_ts.normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d"))
    config = EventConvictionConfig.for_end_date(screen_end_date, ann_start_date=args.ann_start_date, **overrides)

    export_root = export_root_dir(args.export_root)
    export_dir = export_root / f"event_conviction_screen_{config.end_date}"
    export_dir.mkdir(parents=True, exist_ok=True)

    pro = configure_tushare_client(token, custom_http_url=custom_http_url)
    recent_trade_dates = get_recent_open_trade_dates(
        pro,
        config.end_date,
        count=max(config.recent_wave_trade_days + 5, config.moneyflow_lookback_days + 5, 25),
    )
    latest_trade_date, market_moneyflow_dates, daily_basic_latest, tech_latest, moneyflow_summary = fetch_latest_complete_market_inputs(
        pro,
        recent_trade_dates,
        moneyflow_lookback_days=config.moneyflow_lookback_days,
        sleep_sec=args.api_sleep_sec,
    )
    margin_trade_dates = [value for value in recent_trade_dates if value <= latest_trade_date][-3:]
    stock_basic_all = fetch_stock_basic_all(pro)
    market_snapshot = build_market_snapshot(stock_basic_all, daily_basic_latest, tech_latest, moneyflow_summary)
    margin_summary = fetch_margin_detail_summary(pro, margin_trade_dates, sleep_sec=args.api_sleep_sec)
    if not margin_summary.empty:
        market_snapshot = market_snapshot.merge(margin_summary, on="ts_code", how="left")

    holdertrade_raw = fetch_holdertrade_events(
        pro,
        config.ann_start_date,
        latest_trade_date,
        chunk_days=config.event_chunk_days,
        sleep_sec=args.api_sleep_sec,
    )
    buyback_raw = fetch_buyback_events(
        pro,
        config.ann_start_date,
        latest_trade_date,
        sleep_sec=args.api_sleep_sec,
    )
    write_csv_checkpoint(holdertrade_raw, export_dir / "holdertrade_raw.csv")
    write_csv_checkpoint(buyback_raw, export_dir / "buyback_raw.csv")

    holder_candidates = build_holdertrade_event_candidates(holdertrade_raw, stock_basic_all, market_snapshot, latest_trade_date, config)
    buyback_candidates = build_buyback_event_candidates(buyback_raw, stock_basic_all, market_snapshot, latest_trade_date, config)
    candidate_df = pd.concat([holder_candidates, buyback_candidates], ignore_index=True) if not holder_candidates.empty or not buyback_candidates.empty else pd.DataFrame()
    candidate_df = ensure_columns(candidate_df, display_columns())
    candidate_df = apply_bottom_filters(candidate_df, config)
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values(["event_date", "event_amount"], ascending=[False, False]).drop_duplicates(subset=["ts_code", "event_type", "event_date"]).reset_index(drop=True)
        candidate_df = candidate_df.head(config.max_candidates).reset_index(drop=True)

    price_bundle_map: dict[str, dict[str, pd.DataFrame]] = {}
    for ts_code in candidate_df["ts_code"].astype(str).tolist() if not candidate_df.empty else []:
        if ts_code in price_bundle_map:
            continue
        price_bundle_map[ts_code] = fetch_full_price_bundle(
            pro,
            ts_code=ts_code,
            start_date=(pd.Timestamp(config.end_date) - pd.Timedelta(days=max(420, config.price_lookback_days * 2))).strftime("%Y%m%d"),
            end_date=config.end_date,
            sleep_sec=args.api_sleep_sec,
        )
    candidate_df = enrich_market_features(candidate_df, price_bundle_map, latest_trade_date, config)
    scored_df = score_candidates(candidate_df, config)
    top1_df = select_top1_candidate(scored_df, config)

    candidate_export = ensure_columns(candidate_df, display_columns())
    scored_export = ensure_columns(scored_df, display_columns())
    top1_export = ensure_columns(top1_df, display_columns()) if not top1_df.empty else pd.DataFrame(columns=display_columns())
    write_csv_checkpoint(candidate_export, export_dir / "event_candidates.csv")
    write_csv_checkpoint(scored_export, export_dir / "scored_candidates.csv")
    write_csv_checkpoint(top1_export, export_dir / "best_pick_candidate.csv")
    summary = build_screen_summary(config, str(export_dir.resolve()), latest_trade_date, candidate_df, scored_df, top1_df)
    write_json_checkpoint(summary, export_dir / "screen_summary.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if scored_df.empty:
        print("(empty)")
        return
    print(scored_export[display_columns()].head(args.show_top).to_string(index=False))
    if not top1_df.empty:
        print("")
        print("Top-1:")
        print(top1_export[display_columns()].to_string(index=False))


if __name__ == "__main__":
    main()
