from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from core_management_accumulation_strategy import (
    CoreManagementAccumulationConfig,
    STRATEGY_NAME,
    build_event_wave_details,
    build_final_candidate_flags,
    build_margin_summary,
    build_post_wave_structure_metrics,
    build_preliminary_candidate_flags,
    build_screen_summary,
    display_columns,
    score_final_candidates,
    select_best_wave_per_stock,
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


def log_step(message: str) -> None:
    print(f"[core_mgmt] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the core-management repeated buying strategy and keep the strongest stock for one day.")
    parser.add_argument("--end-date", default="", help="Requested end date in YYYYMMDD. Default: today with 20:00 cutoff.")
    parser.add_argument("--ann-start-date", default="", help="Optional fixed announcement window start date in YYYYMMDD.")
    parser.add_argument("--show-top", type=int, default=10, help="Rows to print from the final ranking.")
    parser.add_argument("--config-file", default="", help="Optional JSON file with CoreManagementAccumulationConfig overrides.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON object with CoreManagementAccumulationConfig override keys.")
    parser.add_argument("--export-root", default="", help="Optional export directory. Defaults to tushare_screen_exports.")
    parser.add_argument("--api-sleep-sec", type=float, default=0.15, help="Sleep between API calls. Does not change scoring logic.")
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
    return config


def export_root_dir(path_str: str = "") -> Path:
    if path_str.strip():
        path = Path(path_str).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return output_root_dir()


def fetch_margin_detail_summary(
    pro,
    trade_dates: list[str],
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        log_step(f"margin detail trade_date={trade_date}")
        df = safe_call(
            f"margin_detail_{trade_date}",
            getattr(pro, "margin_detail", None),
            sleep_sec=sleep_sec,
            trade_date=trade_date,
        )
        if df.empty:
            continue
        keep_cols = [c for c in ["ts_code", "trade_date", "rzye", "rzmre", "rzche"] if c in df.columns]
        if keep_cols:
            frames.append(df[keep_cols].copy())
    if not frames:
        return pd.DataFrame()
    return build_margin_summary(pd.concat(frames, ignore_index=True), trade_dates=trade_dates)


def fetch_single_stock_price_bundle(
    pro,
    ts_code: str,
    end_date: str,
    price_lookback_days: int,
    sleep_sec: float = 0.0,
) -> dict[str, Any]:
    price_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(420, price_lookback_days * 2))).strftime("%Y%m%d")
    daily_label = f"core_mgmt_daily_{ts_code}"
    adj_label = f"core_mgmt_adj_factor_{ts_code}"
    daily_df = safe_call(
        daily_label,
        getattr(pro, "daily", None),
        sleep_sec=sleep_sec,
        ts_code=ts_code,
        start_date=price_start_date,
        end_date=end_date,
    )
    adj_df = safe_call(
        adj_label,
        getattr(pro, "adj_factor", None),
        sleep_sec=sleep_sec,
        ts_code=ts_code,
        start_date=price_start_date,
        end_date=end_date,
    )
    return {
        "ts_code": ts_code,
        "daily_df": daily_df,
        "adj_df": adj_df,
    }


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    ensure_token(token)

    config_overrides = load_config_overrides(args)
    now_ts = pd.Timestamp.now()
    today_str = now_ts.strftime("%Y%m%d")
    requested_end_date = str(args.end_date or "").strip() or today_str
    screen_end_date = requested_end_date if args.end_date else (today_str if now_ts.hour >= 20 else (now_ts.normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d"))
    config = CoreManagementAccumulationConfig.for_end_date(screen_end_date, ann_start_date=args.ann_start_date, **config_overrides)
    export_root = export_root_dir(args.export_root)
    export_dir = export_root / f"core_management_accumulation_screen_{config.end_date}"
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
    recent_wave_trade_dates = [value for value in recent_trade_dates if value <= latest_trade_date][-config.recent_wave_trade_days :]
    margin_trade_dates = [value for value in recent_trade_dates if value <= latest_trade_date][-3:]

    stock_basic_all = fetch_stock_basic_all(pro)
    holdertrade_raw = fetch_holdertrade_events(
        pro,
        config.ann_start_date,
        latest_trade_date,
        chunk_days=config.event_chunk_days,
        sleep_sec=args.api_sleep_sec,
    )
    market_snapshot = build_market_snapshot(stock_basic_all, daily_basic_latest, tech_latest, moneyflow_summary)
    margin_summary = fetch_margin_detail_summary(pro, margin_trade_dates, sleep_sec=args.api_sleep_sec)
    if not margin_summary.empty:
        market_snapshot = market_snapshot.merge(margin_summary, on="ts_code", how="left")

    wave_details = build_event_wave_details(
        holdertrade_df=holdertrade_raw,
        stock_basic_df=stock_basic_all,
        market_snapshot=market_snapshot,
        trade_dates=recent_wave_trade_dates,
        config=config,
        latest_trade_date=latest_trade_date,
    )
    write_csv_checkpoint(holdertrade_raw, export_dir / "holdertrade_raw.csv")
    write_csv_checkpoint(wave_details, export_dir / "event_wave_details.csv")

    best_wave_per_stock = select_best_wave_per_stock(wave_details)
    if best_wave_per_stock.empty:
        empty_df = pd.DataFrame(columns=display_columns())
        write_csv_checkpoint(empty_df, export_dir / "stage1_candidates.csv")
        write_csv_checkpoint(empty_df, export_dir / "final_candidates.csv")
        write_csv_checkpoint(empty_df.head(0), export_dir / "best_pick_candidate.csv")
        summary = build_screen_summary(
            config=config,
            export_dir=str(export_dir.resolve()),
            latest_trade_date=latest_trade_date,
            market_moneyflow_dates=market_moneyflow_dates,
            wave_details=wave_details,
            stage1_candidates=empty_df,
            final_candidates=empty_df,
            best_pick_candidate=empty_df,
        )
        write_json_checkpoint(summary, export_dir / "screen_summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("(empty)")
        return

    preliminary_flags = best_wave_per_stock.apply(lambda row: build_preliminary_candidate_flags(row.to_dict(), config), axis=1, result_type="expand")
    stage1_candidates = pd.concat([best_wave_per_stock, preliminary_flags], axis=1)
    stage1_candidates = stage1_candidates[
        stage1_candidates["board_ok"].fillna(False)
        & stage1_candidates["st_ok"].fillna(False)
        & stage1_candidates["listing_ok"].fillna(False)
        & stage1_candidates["price_ok"].fillna(False)
        & stage1_candidates["wave_days_ok"].fillna(False)
        & stage1_candidates["core_event_ok"].fillna(False)
        & stage1_candidates["wave_amount_ok"].fillna(False)
        & stage1_candidates["cost_zone_ok"].fillna(False)
    ].copy()
    stage1_candidates = stage1_candidates.sort_values(
        ["preliminary_score", "wave_last_date", "wave_total_amount", "wave_trade_days"],
        ascending=[False, False, False, False],
    ).head(config.max_deep_dive_stocks).reset_index(drop=True)
    write_csv_checkpoint(stage1_candidates, export_dir / "stage1_candidates.csv")

    if stage1_candidates.empty:
        empty_df = pd.DataFrame(columns=display_columns())
        write_csv_checkpoint(empty_df, export_dir / "final_candidates.csv")
        write_csv_checkpoint(empty_df.head(0), export_dir / "best_pick_candidate.csv")
        summary = build_screen_summary(
            config=config,
            export_dir=str(export_dir.resolve()),
            latest_trade_date=latest_trade_date,
            market_moneyflow_dates=market_moneyflow_dates,
            wave_details=wave_details,
            stage1_candidates=stage1_candidates,
            final_candidates=empty_df,
            best_pick_candidate=empty_df,
        )
        write_json_checkpoint(summary, export_dir / "screen_summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("===== final candidates =====")
        print("(empty)")
        print("===== best pick =====")
        print("(empty)")
        print(f"export_dir={export_dir}")
        return

    deep_rows: list[dict[str, Any]] = []
    for idx, row in stage1_candidates.iterrows():
        ts_code = str(row["ts_code"])
        log_step(f"[stage1 {idx + 1}/{len(stage1_candidates)}] price bundle {ts_code}")
        bundle = fetch_single_stock_price_bundle(
            pro,
            ts_code=ts_code,
            end_date=latest_trade_date,
            price_lookback_days=config.price_lookback_days,
            sleep_sec=args.api_sleep_sec,
        )
        deep_rows.append(
            {
                "ts_code": ts_code,
                **build_post_wave_structure_metrics(
                    daily_df=bundle["daily_df"],
                    adj_df=bundle["adj_df"],
                    end_date=latest_trade_date,
                    wave_first_date=str(row.get("wave_first_date", "")),
                    wave_last_date=str(row.get("wave_last_date", "")),
                    weighted_cost=row.get("wave_buy_avg_price_weighted"),
                    config=config,
                ),
            }
        )
    deep_metrics_df = pd.DataFrame(deep_rows)
    if not deep_metrics_df.empty:
        write_csv_checkpoint(deep_metrics_df, export_dir / "deep_metrics_stage1.csv")

    merged = stage1_candidates.merge(deep_metrics_df, on="ts_code", how="left")
    merged = ensure_columns(
        merged,
        [
            "avg_amount_20d_yuan",
            "latest_close_raw",
            "latest_trade_date",
            "above_ma5",
            "above_ma10",
            "above_ma20",
            "ma10_slope_up",
            "recent_restrengthen_flag",
            "post_wave_breakdown_flag",
            "post_wave_low_to_cost_pct",
            "post_wave_structure_score",
        ],
    )
    final_candidates = score_final_candidates(merged, config)
    best_pick_candidate = final_candidates.head(1).copy()

    write_csv_checkpoint(final_candidates, export_dir / "final_candidates.csv")
    write_csv_checkpoint(best_pick_candidate, export_dir / "best_pick_candidate.csv")
    summary = build_screen_summary(
        config=config,
        export_dir=str(export_dir.resolve()),
        latest_trade_date=latest_trade_date,
        market_moneyflow_dates=market_moneyflow_dates,
        wave_details=wave_details,
        stage1_candidates=stage1_candidates,
        final_candidates=final_candidates,
        best_pick_candidate=best_pick_candidate,
    )
    write_json_checkpoint(summary, export_dir / "screen_summary.json")

    cols = [c for c in display_columns() if c in final_candidates.columns]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("===== final candidates =====")
    if final_candidates.empty:
        print("(empty)")
    else:
        print(ensure_columns(final_candidates, cols)[cols].head(args.show_top).to_string(index=False))
    print("===== best pick =====")
    if best_pick_candidate.empty:
        print("(empty)")
    else:
        print(ensure_columns(best_pick_candidate, cols)[cols].head(1).to_string(index=False))
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
