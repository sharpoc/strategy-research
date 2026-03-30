from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_strategy_registry import (
    StrategyPluginSpec,
    apply_plugin_entry_gate,
    build_price_strategy_registry,
    supported_price_strategy_ids,
)
from market_regime import build_market_regime_snapshot
from research_config_presets import (
    exit_config_preset_help,
    load_exit_config_with_preset,
    load_json_file,
    load_strategy_overrides_with_preset,
    strategy_config_preset_help,
)
from research_backtest_utils import (
    build_daily_frame_map,
    build_forward_return_table,
    configure_tushare_client,
    ensure_columns,
    fetch_market_daily_history,
    fetch_stock_basic_all,
    get_open_trade_dates,
    json_safe,
    log_step,
    repo_root_dir,
)
from research_universe_filters import apply_research_candidate_filters, apply_research_universe_filters
from strategy_exit_rules import apply_exit_rules, build_price_path_map, summarize_exit_reasons


@dataclass(frozen=True)
class BacktestDataset:
    start_date: str
    end_date: str
    trade_dates: list[str]
    history_trade_dates: list[str]
    stock_basic_all: pd.DataFrame
    market_daily_history: pd.DataFrame
    daily_frame_map: dict[str, pd.DataFrame]
    price_path_map: dict[str, pd.DataFrame]
    regime_snapshot: pd.DataFrame
    regime_map: pd.DataFrame
    forward_table: pd.DataFrame


STRATEGY_REGISTRY: dict[str, StrategyPluginSpec] = build_price_strategy_registry()
SUPPORTED_STRATEGY_IDS = supported_price_strategy_ids()


def slice_backtest_dataset(dataset: BacktestDataset, start_date: str, end_date: str) -> BacktestDataset:
    trade_dates = [trade_date for trade_date in dataset.trade_dates if start_date <= trade_date <= end_date]
    if not trade_dates:
        raise ValueError(f"No trade dates inside dataset for range {start_date} - {end_date}")

    allowed_history_dates = {trade_date for trade_date in dataset.history_trade_dates if trade_date <= trade_dates[-1]}
    history_trade_dates = [trade_date for trade_date in dataset.history_trade_dates if trade_date in allowed_history_dates]
    history_filter = dataset.market_daily_history["trade_date"].astype(str).isin(allowed_history_dates)
    regime_snapshot = (
        dataset.regime_snapshot[dataset.regime_snapshot["trade_date"].astype(str).isin(trade_dates)].copy()
        if not dataset.regime_snapshot.empty and "trade_date" in dataset.regime_snapshot.columns
        else pd.DataFrame()
    )
    regime_map = regime_snapshot.set_index("trade_date") if not regime_snapshot.empty else pd.DataFrame()
    forward_table = (
        dataset.forward_table[dataset.forward_table["trade_date"].astype(str).isin(trade_dates)].copy()
        if not dataset.forward_table.empty and "trade_date" in dataset.forward_table.columns
        else pd.DataFrame()
    )
    daily_frame_map = {trade_date: dataset.daily_frame_map[trade_date] for trade_date in history_trade_dates if trade_date in dataset.daily_frame_map}
    return BacktestDataset(
        start_date=start_date,
        end_date=end_date,
        trade_dates=trade_dates,
        history_trade_dates=history_trade_dates,
        stock_basic_all=dataset.stock_basic_all,
        market_daily_history=dataset.market_daily_history[history_filter].copy(),
        daily_frame_map=daily_frame_map,
        price_path_map=dataset.price_path_map,
        regime_snapshot=regime_snapshot,
        regime_map=regime_map,
        forward_table=forward_table,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest price-based strategies by market regime without touching the live/test environment.")
    parser.add_argument("--start-date", required=True, help="Backtest start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Backtest end date in YYYYMMDD.")
    parser.add_argument(
        "--strategies",
        default="limitup_l1l2,platform_breakout,double_bottom",
        help=f"Comma-separated strategy ids. Supported: {','.join(SUPPORTED_STRATEGY_IDS)}",
    )
    parser.add_argument("--hold-days", default="1,3,5,10", help="Comma-separated holding days, based on T+1 open entry.")
    parser.add_argument(
        "--strategy-config-preset",
        default="",
        help=f"Named research preset for strategy overrides. Available: {strategy_config_preset_help()}",
    )
    parser.add_argument("--strategy-config-file", default="", help="JSON file with per-strategy config overrides.")
    parser.add_argument("--regime-config-file", default="", help="JSON file with market regime config overrides.")
    parser.add_argument(
        "--exit-config-preset",
        default="",
        help=f"Named exit preset. Available: {exit_config_preset_help()}",
    )
    parser.add_argument("--exit-config-file", default="", help="JSON file with sell-rule config overrides.")
    parser.add_argument("--max-trade-days", type=int, default=0, help="Only use the most recent N trade days for smoke test.")
    parser.add_argument("--show-top", type=int, default=20, help="Rows to print from daily picks.")
    return parser.parse_args()


def hash_config(config: dict[str, Any]) -> str:
    normalized = json.dumps(json_safe(config), ensure_ascii=False, sort_keys=True)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]

HISTORY_LOOKBACK_CALENDAR_MULTIPLIER = 2.2


def signal_cache_dir() -> Path:
    path = repo_root_dir() / "output" / "research_backtests" / "signal_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_root_dir() -> Path:
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def signal_cache_path(strategy_id: str, trade_date: str, config_hash: str) -> Path:
    return signal_cache_dir() / f"{strategy_id}_{trade_date}_{config_hash}.csv"


def evaluate_strategy_day(
    adapter: StrategyPluginSpec,
    trade_date: str,
    market_regime: str,
    window_history: pd.DataFrame,
    stock_basic_df: pd.DataFrame,
    config: dict[str, Any],
    use_cache: bool = True,
) -> dict[str, Any]:
    config_hash = hash_config(config)
    cache_path = signal_cache_path(adapter.strategy_id, trade_date, config_hash)
    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if not cached.empty:
            row = cached.iloc[0].to_dict()
            if "trade_date" in row and row["trade_date"] is not None:
                row["trade_date"] = str(row["trade_date"])
            if "ts_code" in row and row["ts_code"] is not None and not pd.isna(row["ts_code"]):
                row["ts_code"] = str(row["ts_code"])
            return row

    filtered_stock_basic, filtered_window_history, universe_meta = apply_research_universe_filters(
        stock_basic_df=stock_basic_df,
        window_history=window_history,
        filter_config=config.get("_research_filters"),
    )
    candidates = adapter.build_candidates(filtered_window_history, filtered_stock_basic, config)
    raw_signal_count = int(len(candidates))
    candidates, filter_meta = apply_research_candidate_filters(
        candidates=candidates,
        window_history=filtered_window_history,
        filter_config=config.get("_research_filters"),
    )
    row: dict[str, Any] = {
        "trade_date": trade_date,
        "strategy_id": adapter.strategy_id,
        "strategy_name": adapter.strategy_name,
        "signal_count": int(len(candidates)),
        "signal_count_before_filter": raw_signal_count,
        "has_signal": bool(not candidates.empty),
        "config_hash": config_hash,
        "research_filter_enabled": bool(filter_meta.get("enabled", False)),
        "research_filter_drop_count": int(filter_meta.get("dropped_count", 0) or 0),
        "research_filter_drop_reasons": json.dumps(filter_meta.get("drop_reasons", {}), ensure_ascii=False, sort_keys=True),
        "research_universe_before_count": int(universe_meta.get("before_count", 0) or 0),
        "research_universe_after_count": int(universe_meta.get("after_count", 0) or 0),
        "research_universe_drop_count": int(universe_meta.get("dropped_count", 0) or 0),
        "research_universe_drop_reasons": json.dumps(universe_meta.get("drop_reasons", {}), ensure_ascii=False, sort_keys=True),
    }
    if not candidates.empty:
        best = candidates.iloc[0].to_dict()
        gate_passed, gate_failures, gate_enabled = apply_plugin_entry_gate(
            plugin=adapter,
            row=best,
            market_regime=market_regime,
            config=config,
        )
        row["research_entry_gate_enabled"] = gate_enabled
        row["research_entry_gate_passed"] = (bool(gate_passed) if gate_enabled else None)
        row["research_entry_gate_reason"] = "|".join(gate_failures) if gate_enabled else ""
        if gate_passed or not gate_enabled:
            row.update(best)
        else:
            row["has_signal"] = False
            row["research_entry_gate_blocked_ts_code"] = best.get("ts_code")
            row["research_entry_gate_blocked_name"] = best.get("name")
            row["research_entry_gate_blocked_rank_score"] = best.get("strategy_rank_score")
    else:
        row["research_entry_gate_enabled"] = bool((config.get("_research_entry_gate") or {}).get("enabled", False))
        row["research_entry_gate_passed"] = None
        row["research_entry_gate_reason"] = ""
    if use_cache:
        pd.DataFrame([row]).to_csv(cache_path, index=False)
    return row


def summarize_results(daily_results: pd.DataFrame, hold_days: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_results = daily_results[daily_results["has_signal"]].copy()
    strategy_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []

    def aggregate_frame(group_df: pd.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
        row = dict(extra)
        row["calendar_days"] = int(group_df["trade_date"].nunique())
        row["signal_days"] = int(group_df["has_signal"].sum())
        entry_series = pd.to_numeric(group_df["entry_open"], errors="coerce") if "entry_open" in group_df.columns else pd.Series(dtype=float)
        row["filled_trades"] = int(entry_series.notna().sum())
        for day in hold_days:
            ret_col = f"return_open_to_close_{day}d_pct"
            runup_col = f"max_runup_{day}d_pct"
            dd_col = f"max_drawdown_{day}d_pct"
            valid_returns = group_df[ret_col].dropna() if ret_col in group_df.columns else pd.Series(dtype=float)
            valid_runups = group_df[runup_col].dropna() if runup_col in group_df.columns else pd.Series(dtype=float)
            valid_drawdowns = group_df[dd_col].dropna() if dd_col in group_df.columns else pd.Series(dtype=float)
            row[f"avg_return_{day}d_pct"] = round(float(valid_returns.mean()), 4) if not valid_returns.empty else None
            row[f"win_rate_{day}d_pct"] = round(float((valid_returns > 0).mean() * 100.0), 2) if not valid_returns.empty else None
            row[f"avg_runup_{day}d_pct"] = round(float(valid_runups.mean()), 4) if not valid_runups.empty else None
            row[f"avg_drawdown_{day}d_pct"] = round(float(valid_drawdowns.mean()), 4) if not valid_drawdowns.empty else None
        if "exit_return_pct" in group_df.columns:
            valid_exit_returns = pd.to_numeric(group_df["exit_return_pct"], errors="coerce").dropna()
            valid_exit_hold = pd.to_numeric(group_df.get("exit_hold_days"), errors="coerce").dropna()
            valid_exit_mfe = pd.to_numeric(group_df.get("exit_mfe_pct"), errors="coerce").dropna()
            valid_exit_mae = pd.to_numeric(group_df.get("exit_mae_pct"), errors="coerce").dropna()
            row["avg_exit_return_pct"] = round(float(valid_exit_returns.mean()), 4) if not valid_exit_returns.empty else None
            row["exit_win_rate_pct"] = round(float((valid_exit_returns > 0).mean() * 100.0), 2) if not valid_exit_returns.empty else None
            row["avg_exit_hold_days"] = round(float(valid_exit_hold.mean()), 2) if not valid_exit_hold.empty else None
            row["avg_exit_mfe_pct"] = round(float(valid_exit_mfe.mean()), 4) if not valid_exit_mfe.empty else None
            row["avg_exit_mae_pct"] = round(float(valid_exit_mae.mean()), 4) if not valid_exit_mae.empty else None
            if "exit_reason" in group_df.columns:
                reason_series = group_df["exit_reason"].fillna("").astype(str)
                reason_counts = reason_series[reason_series != ""].value_counts()
                row["top_exit_reason"] = reason_counts.index[0] if not reason_counts.empty else None
        row["avg_signal_score"] = round(float(group_df["strategy_rank_score"].mean()), 4) if "strategy_rank_score" in group_df.columns else None
        return row

    for strategy_id, sub in daily_results.groupby("strategy_id", sort=False):
        strategy_rows.append(aggregate_frame(sub, {"strategy_id": strategy_id, "strategy_name": sub["strategy_name"].iloc[0]}))

    for (strategy_id, market_regime), sub in daily_results.groupby(["strategy_id", "market_regime"], sort=False):
        regime_rows.append(
            aggregate_frame(
                sub,
                {
                    "strategy_id": strategy_id,
                    "strategy_name": sub["strategy_name"].iloc[0],
                    "market_regime": market_regime,
                },
            )
        )

    if not signal_results.empty:
        signal_results["trade_month"] = signal_results["trade_date"].astype(str).str.slice(0, 6)
        for (strategy_id, trade_month), sub in signal_results.groupby(["strategy_id", "trade_month"], sort=False):
            month_rows.append(
                aggregate_frame(
                    sub,
                    {
                        "strategy_id": strategy_id,
                        "strategy_name": sub["strategy_name"].iloc[0],
                        "trade_month": trade_month,
                    },
                )
            )

    return pd.DataFrame(strategy_rows), pd.DataFrame(regime_rows), pd.DataFrame(month_rows)


def run_backtest(
    start_date: str,
    end_date: str,
    strategy_ids: list[str],
    hold_days: list[int],
    strategy_overrides: dict[str, Any] | None = None,
    regime_config: dict[str, Any] | None = None,
    exit_config: dict[str, Any] | None = None,
    max_trade_days: int = 0,
    use_signal_cache: bool = True,
) -> dict[str, Any]:
    dataset = prepare_backtest_dataset(
        start_date=start_date,
        end_date=end_date,
        strategy_ids=strategy_ids,
        hold_days=hold_days,
        regime_config=regime_config,
        max_trade_days=max_trade_days,
    )
    return run_backtest_on_dataset(
        dataset=dataset,
        strategy_ids=strategy_ids,
        hold_days=hold_days,
        strategy_overrides=strategy_overrides,
        exit_config=exit_config,
        use_signal_cache=use_signal_cache,
    )


def prepare_backtest_dataset(
    start_date: str,
    end_date: str,
    strategy_ids: list[str],
    hold_days: list[int],
    regime_config: dict[str, Any] | None = None,
    max_trade_days: int = 0,
) -> BacktestDataset:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "").strip()
    enable_official_fallback = os.getenv("RESEARCH_ENABLE_OFFICIAL_FALLBACK", "").strip().lower() in {"1", "true", "yes", "y"}
    if not token:
        raise SystemExit("Missing TUSHARE_TOKEN in environment.")

    today_str = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
    effective_end_date = min(end_date, today_str)

    pro = configure_tushare_client(token=token, custom_http_url=custom_http_url)
    fallback_pro = configure_tushare_client(token=token, custom_http_url="") if custom_http_url and enable_official_fallback else None
    trade_dates = get_open_trade_dates(pro, start_date=start_date, end_date=effective_end_date)
    if not trade_dates:
        raise SystemExit("No trade dates found for the requested range.")
    if max_trade_days > 0:
        trade_dates = trade_dates[-max_trade_days:]

    max_history_bars = max(STRATEGY_REGISTRY[strategy_id].history_bars for strategy_id in strategy_ids)
    history_lookback_days = max(90, int(max_history_bars * HISTORY_LOOKBACK_CALENDAR_MULTIPLIER))
    history_trade_dates = get_open_trade_dates(
        pro,
        start_date=(pd.Timestamp(trade_dates[0]) - pd.Timedelta(days=history_lookback_days)).strftime("%Y%m%d"),
        end_date=min(trade_dates[-1], effective_end_date),
    )
    stock_basic_all = fetch_stock_basic_all(pro)
    if stock_basic_all.empty:
        raise SystemExit("stock_basic returned no data.")

    market_daily_history = fetch_market_daily_history(pro, history_trade_dates, sleep_sec=0.0, fallback_pro=fallback_pro)
    if market_daily_history.empty:
        raise SystemExit("daily history returned no data.")
    available_trade_dates = sorted(market_daily_history["trade_date"].astype(str).unique().tolist())
    available_trade_date_set = set(available_trade_dates)
    trade_dates = [trade_date for trade_date in trade_dates if trade_date in available_trade_date_set]
    history_trade_dates = [trade_date for trade_date in history_trade_dates if trade_date in available_trade_date_set]
    if not trade_dates:
        raise SystemExit("No overlapping trade dates with available daily history.")
    daily_frame_map = build_daily_frame_map(market_daily_history)
    price_path_map = build_price_path_map(market_daily_history)
    regime_snapshot = build_market_regime_snapshot(market_daily_history, stock_basic_all, config=regime_config)
    regime_map = regime_snapshot.set_index("trade_date") if not regime_snapshot.empty else pd.DataFrame()
    forward_table = build_forward_return_table(market_daily_history, hold_days=hold_days)

    return BacktestDataset(
        start_date=start_date,
        end_date=effective_end_date,
        trade_dates=trade_dates,
        history_trade_dates=history_trade_dates,
        stock_basic_all=stock_basic_all,
        market_daily_history=market_daily_history,
        daily_frame_map=daily_frame_map,
        price_path_map=price_path_map,
        regime_snapshot=regime_snapshot,
        regime_map=regime_map,
        forward_table=forward_table,
    )


def run_backtest_on_dataset(
    dataset: BacktestDataset,
    strategy_ids: list[str],
    hold_days: list[int],
    strategy_overrides: dict[str, Any] | None = None,
    exit_config: dict[str, Any] | None = None,
    export_results: bool = True,
    use_signal_cache: bool = True,
) -> dict[str, Any]:
    strategy_overrides = strategy_overrides or {}

    rows: list[dict[str, Any]] = []
    history_index_map = {trade_date: idx for idx, trade_date in enumerate(dataset.history_trade_dates)}
    for date_idx, trade_date in enumerate(dataset.trade_dates, start=1):
        log_step(f"backtest trade_date {date_idx}/{len(dataset.trade_dates)} {trade_date}")
        date_position = history_index_map[trade_date]
        for strategy_id in strategy_ids:
            adapter = STRATEGY_REGISTRY[strategy_id]
            window_dates = dataset.history_trade_dates[max(0, date_position - adapter.history_bars + 1) : date_position + 1]
            window_frames = [dataset.daily_frame_map[d] for d in window_dates if d in dataset.daily_frame_map]
            if not window_frames:
                continue
            window_history = pd.concat(window_frames, ignore_index=True)
            config = dict(strategy_overrides.get(strategy_id, {}))
            market_regime = (
                dataset.regime_map.at[trade_date, "market_regime"]
                if not dataset.regime_map.empty and trade_date in dataset.regime_map.index
                else ""
            )
            row = evaluate_strategy_day(
                adapter,
                trade_date,
                market_regime,
                window_history,
                dataset.stock_basic_all,
                config,
                use_cache=use_signal_cache,
            )
            row["market_regime"] = market_regime
            row["market_regime_score"] = (
                dataset.regime_map.at[trade_date, "market_regime_score"]
                if not dataset.regime_map.empty and trade_date in dataset.regime_map.index
                else np.nan
            )
            row["market_regime_reason"] = (
                dataset.regime_map.at[trade_date, "market_regime_reason"]
                if not dataset.regime_map.empty and trade_date in dataset.regime_map.index
                else ""
            )
            rows.append(row)

    daily_results = pd.DataFrame(rows)
    if daily_results.empty:
        raise SystemExit("Backtest produced no daily rows.")
    if "trade_date" in daily_results.columns:
        daily_results["trade_date"] = daily_results["trade_date"].astype(str)
    if "ts_code" in daily_results.columns:
        daily_results["ts_code"] = daily_results["ts_code"].astype(str)

    if not dataset.forward_table.empty and "ts_code" in daily_results.columns:
        forward_table = dataset.forward_table.copy()
        forward_table["trade_date"] = forward_table["trade_date"].astype(str)
        forward_table["ts_code"] = forward_table["ts_code"].astype(str)
        daily_results = daily_results.merge(forward_table, on=["ts_code", "trade_date"], how="left")

    daily_results = apply_exit_rules(daily_results, dataset.price_path_map, config=exit_config)
    strategy_summary, regime_summary, monthly_summary = summarize_results(daily_results, hold_days=hold_days)
    exit_reason_summary = summarize_exit_reasons(daily_results)

    export_dir: Path | None = None
    if export_results:
        run_tag = f"price_regime_backtest_{dataset.start_date}_{dataset.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
        export_dir = export_root_dir() / run_tag
        export_dir.mkdir(parents=True, exist_ok=True)
        dataset.regime_snapshot.to_csv(export_dir / "market_regime_snapshot.csv", index=False)
        daily_results.to_csv(export_dir / "daily_results.csv", index=False)
        strategy_summary.to_csv(export_dir / "strategy_summary.csv", index=False)
        regime_summary.to_csv(export_dir / "regime_summary.csv", index=False)
        monthly_summary.to_csv(export_dir / "monthly_summary.csv", index=False)
        exit_reason_summary.to_csv(export_dir / "exit_reason_summary.csv", index=False)

    summary = {
        "start_date": dataset.start_date,
        "end_date": dataset.end_date,
        "trade_days": len(dataset.trade_dates),
        "strategies": strategy_ids,
        "hold_days": hold_days,
        "market_regime_rows": int(len(dataset.regime_snapshot)),
        "daily_rows": int(len(daily_results)),
        "export_dir": str(export_dir.resolve()) if export_dir is not None else None,
    }
    if export_dir is not None:
        with (export_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    return {
        "summary": summary,
        "daily_results": daily_results,
        "strategy_summary": strategy_summary,
        "regime_summary": regime_summary,
        "monthly_summary": monthly_summary,
        "exit_reason_summary": exit_reason_summary,
        "regime_snapshot": dataset.regime_snapshot,
        "export_dir": export_dir,
    }


def main() -> None:
    args = parse_args()
    strategy_ids = [item.strip() for item in args.strategies.split(",") if item.strip()]
    unknown = [item for item in strategy_ids if item not in STRATEGY_REGISTRY]
    if unknown:
        raise SystemExit(f"Unsupported strategies: {unknown}")
    hold_days = sorted({int(item.strip()) for item in args.hold_days.split(",") if item.strip()})
    strategy_overrides, strategy_config_meta = load_strategy_overrides_with_preset(
        preset_name=args.strategy_config_preset,
        config_file=args.strategy_config_file,
    )
    regime_config = load_json_file(args.regime_config_file)
    exit_config, _exit_config_meta = load_exit_config_with_preset(
        preset_name=args.exit_config_preset,
        config_file=args.exit_config_file,
    )

    result = run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_ids=strategy_ids,
        hold_days=hold_days,
        strategy_overrides=strategy_overrides,
        regime_config=regime_config,
        exit_config=exit_config,
        max_trade_days=args.max_trade_days,
    )
    result["summary"]["strategy_config_preset"] = strategy_config_meta["strategy_config_preset"]
    result["summary"]["strategy_config_file"] = strategy_config_meta["strategy_config_file"]
    result["summary"]["preset_strategy_config_file"] = strategy_config_meta["preset_strategy_config_file"]
    export_dir = result.get("export_dir")
    if export_dir is not None:
        with (export_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(json_safe(result["summary"]), f, ensure_ascii=False, indent=2)

    print(json.dumps(json_safe(result["summary"]), ensure_ascii=False, indent=2))
    print("===== strategy summary =====")
    print(result["strategy_summary"].to_string(index=False))
    print("===== regime summary =====")
    print(result["regime_summary"].to_string(index=False))
    print("===== exit reason summary =====")
    if result["exit_reason_summary"].empty:
        print("(empty)")
    else:
        print(result["exit_reason_summary"].to_string(index=False))
    print("===== daily sample =====")
    print(result["daily_results"].head(args.show_top).to_string(index=False))
    print(f"export_dir={result['export_dir']}")


if __name__ == "__main__":
    main()
