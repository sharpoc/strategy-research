from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from event_conviction_strategy import (
    EventConvictionConfig,
    apply_bottom_filters,
    build_buyback_event_candidates,
    build_holdertrade_event_candidates,
    enrich_market_features,
    score_candidates,
    select_top1_candidate,
)
from holder_strategy_core import (
    build_market_snapshot,
    configure_tushare_client,
    ensure_token,
    fetch_holdertrade_events,
    fetch_latest_complete_market_inputs,
    fetch_stock_basic_all,
    get_recent_open_trade_dates,
)
from research_backtest_utils import json_safe, repo_root_dir
from run_core_management_final_review import compute_forward_returns, summarize_returns, fetch_full_price_bundle
from run_tushare_core_management_accumulation_strategy import fetch_margin_detail_summary
from run_tushare_event_conviction_strategy import fetch_buyback_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the event-conviction strategy and evaluate daily Top-1 picks.")
    parser.add_argument("--start-date", default="20260317", help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", default="20260324", help="End date in YYYYMMDD.")
    parser.add_argument("--hold-days", default="3,5,10", help="Comma-separated holding windows.")
    parser.add_argument("--config-file", default="", help="Optional JSON config overrides.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON config overrides.")
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


def parse_hold_days(raw: str) -> list[int]:
    values = sorted({int(token.strip()) for token in str(raw or "").split(",") if token.strip()})
    if not values:
        raise SystemExit("At least one positive hold day is required.")
    return [value for value in values if value > 0]


def load_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.config_file:
        path = Path(args.config_file).expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit("Config file must contain a JSON object.")
        overrides.update(data)
    if args.config_json.strip():
        inline = json.loads(args.config_json)
        if not isinstance(inline, dict):
            raise SystemExit("--config-json must be a JSON object.")
        overrides.update(inline)
    return overrides


def build_report_markdown(summary_payload: dict[str, Any], top1_df: pd.DataFrame, daily_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# 事件信念臻选 Top-1 回测")
    lines.append("")
    range_info = summary_payload.get("range", {})
    lines.append(f"- 区间: {range_info.get('start_date')} ~ {range_info.get('end_date')}")
    lines.append(f"- 扫描交易日: {range_info.get('trade_days_scanned')}")
    lines.append(f"- 候选日均: {summary_payload.get('candidate_day_avg')}")
    lines.append(f"- 候选日峰值: {summary_payload.get('candidate_day_max')}")
    lines.append(f"- Top-1 出信号天数: {summary_payload.get('top1_signal_days')}")
    lines.append("")

    top1_summary = summary_payload.get("top1_summary", {})
    lines.append("## Top-1 收益摘要")
    lines.append("")
    lines.append(f"- 3日均值: {top1_summary.get('avg_3d_pct')}")
    lines.append(f"- 3日胜率: {top1_summary.get('win_rate_3d_pct')}")
    lines.append(f"- 5日均值: {top1_summary.get('avg_5d_pct')}")
    lines.append(f"- 5日胜率: {top1_summary.get('win_rate_5d_pct')}")
    lines.append(f"- 10日均值: {top1_summary.get('avg_10d_pct')}")
    lines.append(f"- 10日胜率: {top1_summary.get('win_rate_10d_pct')}")
    lines.append("")

    duplicates = summary_payload.get("top1_duplicates", {})
    lines.append("## Top-1 重复分布")
    lines.append("")
    if duplicates:
        for ts_code, count in duplicates.items():
            lines.append(f"- {ts_code}: {count}")
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 每日摘要")
    lines.append("")
    if daily_df.empty:
        lines.append("- 无")
    else:
        for _, row in daily_df.iterrows():
            lines.append(
                f"- {row.get('trade_date')}: 候选 {int(row.get('candidate_rows') or 0)} / 唯一股票 {int(row.get('unique_stocks') or 0)} / Top-1 {int(row.get('top1_rows') or 0)}"
            )
    lines.append("")

    lines.append("## Top-1 明细")
    lines.append("")
    if top1_df.empty:
        lines.append("- 无")
    else:
        display = top1_df.copy()
        keep_cols = [
            "signal_date",
            "ts_code",
            "name",
            "event_type",
            "total_score",
            "return_open_to_close_3d_pct",
            "return_open_to_close_5d_pct",
            "return_open_to_close_10d_pct",
        ]
        display = display[[column for column in keep_cols if column in display.columns]]
        lines.append(dataframe_to_markdown_table(display))
    lines.append("")
    return "\n".join(lines)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for column in df.columns:
            value = row.get(column)
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    ensure_token(token)
    hold_days = parse_hold_days(args.hold_days)
    config_overrides = load_config_overrides(args)

    export_dir = export_root_dir(args.export_root) / f"event_conviction_top1_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    pro = configure_tushare_client(token, custom_http_url=custom_http_url)
    trade_dates = get_recent_open_trade_dates(pro, args.end_date, count=2000)
    trade_dates = [value for value in trade_dates if args.start_date <= value <= args.end_date]
    if not trade_dates:
        raise SystemExit("No trade dates found in requested range.")

    stock_basic_all = fetch_stock_basic_all(pro)
    price_bundle_map: dict[str, dict[str, pd.DataFrame]] = {}
    top1_rows: list[dict[str, Any]] = []
    daily_summary_rows: list[dict[str, Any]] = []
    prior_top1_history: list[dict[str, str]] = []

    for idx, trade_date in enumerate(trade_dates, start=1):
        print(f"[event_conviction_research] {idx}/{len(trade_dates)} trade_date={trade_date}", flush=True)
        config = EventConvictionConfig.for_end_date(trade_date, **config_overrides)
        recent_trade_dates = get_recent_open_trade_dates(
            pro,
            trade_date,
            count=max(config.recent_wave_trade_days + 5, config.moneyflow_lookback_days + 5, 25),
        )
        latest_trade_date, _, daily_basic_latest, tech_latest, moneyflow_summary = fetch_latest_complete_market_inputs(
            pro,
            recent_trade_dates,
            moneyflow_lookback_days=config.moneyflow_lookback_days,
            sleep_sec=args.api_sleep_sec,
        )
        market_snapshot = build_market_snapshot(stock_basic_all, daily_basic_latest, tech_latest, moneyflow_summary)
        margin_trade_dates = [value for value in recent_trade_dates if value <= latest_trade_date][-3:]
        margin_summary = fetch_margin_detail_summary(pro, margin_trade_dates, sleep_sec=args.api_sleep_sec)
        if not margin_summary.empty:
            market_snapshot = market_snapshot.merge(margin_summary, on="ts_code", how="left")

        holdertrade_raw = fetch_holdertrade_events(pro, config.ann_start_date, latest_trade_date, chunk_days=config.event_chunk_days, sleep_sec=args.api_sleep_sec)
        buyback_raw = fetch_buyback_events(pro, config.ann_start_date, latest_trade_date, sleep_sec=args.api_sleep_sec)
        holder_candidates = build_holdertrade_event_candidates(holdertrade_raw, stock_basic_all, market_snapshot, latest_trade_date, config)
        buyback_candidates = build_buyback_event_candidates(buyback_raw, stock_basic_all, market_snapshot, latest_trade_date, config)
        candidate_df = pd.concat([holder_candidates, buyback_candidates], ignore_index=True) if not holder_candidates.empty or not buyback_candidates.empty else pd.DataFrame()
        candidate_df = apply_bottom_filters(candidate_df, config)
        if not candidate_df.empty:
            candidate_df = candidate_df.sort_values(["event_date", "event_amount"], ascending=[False, False]).drop_duplicates(subset=["ts_code", "event_type", "event_date"]).reset_index(drop=True)
            candidate_df = candidate_df.head(config.max_candidates).reset_index(drop=True)

        for ts_code in candidate_df["ts_code"].astype(str).tolist() if not candidate_df.empty else []:
            if ts_code in price_bundle_map:
                continue
            price_bundle_map[ts_code] = fetch_full_price_bundle(
                pro,
                ts_code=ts_code,
                start_date=(pd.Timestamp(trade_date) - pd.Timedelta(days=max(420, config.price_lookback_days * 2))).strftime("%Y%m%d"),
                end_date=args.end_date,
                sleep_sec=args.api_sleep_sec,
            )
        candidate_df = enrich_market_features(candidate_df, price_bundle_map, latest_trade_date, config)
        if not candidate_df.empty:
            recent_counts: dict[str, int] = {}
            recent_window = prior_top1_history[-5:]
            for item in recent_window:
                ts_code = str(item.get("ts_code") or "")
                if not ts_code:
                    continue
                recent_counts[ts_code] = recent_counts.get(ts_code, 0) + 1
            candidate_df["recent_top1_count"] = candidate_df["ts_code"].astype(str).map(recent_counts).fillna(0).astype(int)
        scored_df = score_candidates(candidate_df, config)
        top1_df = select_top1_candidate(scored_df, config)

        daily_summary_rows.append(
            {
                "trade_date": trade_date,
                "candidate_rows": int(len(candidate_df)),
                "unique_stocks": int(candidate_df["ts_code"].astype(str).nunique()) if not candidate_df.empty else 0,
                "top1_rows": int(len(top1_df)),
            }
        )
        if top1_df.empty:
            continue
        row = top1_df.iloc[0].to_dict()
        ts_code = str(row.get("ts_code") or "")
        forward_payload = compute_forward_returns(price_bundle_map.get(ts_code, {}).get("daily_df", pd.DataFrame()), latest_trade_date, hold_days)
        top1_rows.append(
            {
                "signal_date": latest_trade_date,
                "screen_trade_date": trade_date,
                **row,
                **forward_payload,
            }
        )
        prior_top1_history.append({"trade_date": latest_trade_date, "ts_code": ts_code})

    top1_df = pd.DataFrame(top1_rows)
    daily_df = pd.DataFrame(daily_summary_rows)
    summary_payload = {
        "strategy_id": "event_conviction_signal",
        "strategy_name": "事件信念臻选",
        "range": {"start_date": args.start_date, "end_date": args.end_date, "trade_days_scanned": len(trade_dates)},
        "hold_days": hold_days,
        "config_overrides": config_overrides,
        "top1_summary": summarize_returns("top1", top1_df, hold_days),
        "candidate_day_avg": round(float(daily_df["candidate_rows"].mean()), 2) if not daily_df.empty else 0.0,
        "candidate_day_max": int(daily_df["candidate_rows"].max()) if not daily_df.empty else 0,
        "top1_signal_days": int(top1_df["signal_date"].astype(str).nunique()) if not top1_df.empty else 0,
        "top1_duplicates": top1_df["ts_code"].astype(str).value_counts().head(10).to_dict() if not top1_df.empty else {},
        "export_dir": str(export_dir),
    }
    top1_df.to_csv(export_dir / "top1_signals.csv", index=False)
    daily_df.to_csv(export_dir / "daily_summary.csv", index=False)
    (export_dir / "top1_summary.json").write_text(json.dumps(json_safe(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (export_dir / "top1_report.md").write_text(build_report_markdown(summary_payload, top1_df, daily_df), encoding="utf-8")
    print(json.dumps(json_safe(summary_payload), ensure_ascii=False, indent=2))
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
