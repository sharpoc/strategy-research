from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

from core_management_accumulation_strategy import (
    CoreManagementAccumulationConfig,
    STRATEGY_NAME,
    build_event_wave_details,
    build_final_candidate_flags,
    build_margin_summary,
    build_post_wave_structure_metrics,
    build_screen_summary,
    build_wave_signature,
    display_columns,
    score_final_candidates,
    select_best_wave_per_stock,
)
from holder_strategy_core import (
    build_market_technical_snapshot_from_cached_history,
    build_market_snapshot,
    chunk_date_ranges,
    compute_main_net_amount,
    configure_tushare_client,
    ensure_token,
    fetch_holdertrade_events,
    fetch_latest_complete_market_inputs,
    fetch_stock_basic_all,
    get_recent_open_trade_dates,
    safe_call,
)
from research_backtest_utils import json_safe, repo_root_dir
from run_tushare_core_management_accumulation_strategy import fetch_margin_detail_summary


def log_step(message: str) -> None:
    print(f"[core_mgmt_review] {message}", flush=True)


def holder_cache_dir() -> Path:
    return repo_root_dir() / "output" / "cache" / "holder_increase_api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay core-management accumulation candidate days and compare the lighter final layer against stage1."
    )
    parser.add_argument(
        "--stats-json",
        default="/tmp/core_management_6m_stats.json",
        help="Path to the baseline 6m stats JSON that provides candidate dates and baseline summaries.",
    )
    parser.add_argument(
        "--export-root",
        default="",
        help="Optional export root. Defaults to output/research_backtests.",
    )
    parser.add_argument(
        "--hold-days",
        default="3,5,10",
        help="Comma-separated holding windows measured from T+1 open to T+N close.",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=0,
        help="Optional cap for candidate trade dates, useful for smoke runs.",
    )
    parser.add_argument(
        "--config-file",
        default="",
        help="Optional JSON file with CoreManagementAccumulationConfig overrides.",
    )
    parser.add_argument(
        "--config-json",
        default="",
        help="Optional inline JSON object with CoreManagementAccumulationConfig override keys.",
    )
    parser.add_argument(
        "--api-sleep-sec",
        type=float,
        default=0.12,
        help="Sleep between API calls. Keep low when cache is warm.",
    )
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"JSON object expected: {path}")
    return data


def load_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.config_file:
        path = Path(args.config_file).expanduser().resolve()
        overrides.update(load_json_object(path))
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
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_hold_days(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    cleaned = sorted({value for value in values if value > 0})
    if not cleaned:
        raise SystemExit("At least one positive hold day is required.")
    return cleaned


def build_candidate_dates(stats_payload: dict[str, Any], max_dates: int = 0) -> list[str]:
    progress_rows = stats_payload.get("progress", [])
    dates: list[str] = []
    for row in progress_rows:
        if isinstance(row, list) and row:
            trade_date = str(row[0])
            if len(trade_date) == 8 and trade_date.isdigit():
                dates.append(trade_date)
    dates = sorted(dict.fromkeys(dates))
    if max_dates > 0:
        dates = dates[:max_dates]
    return dates


def load_latest_cached_frame_by_prefix(prefix: str) -> pd.DataFrame:
    cache_dir = holder_cache_dir()
    if not cache_dir.exists():
        return pd.DataFrame()
    candidates = sorted(cache_dir.glob(f"{prefix}_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            return pd.read_csv(path)
        except Exception:
            continue
    return pd.DataFrame()


def load_cached_trade_date_frame(prefix: str, trade_date: str) -> pd.DataFrame:
    return load_latest_cached_frame_by_prefix(f"{prefix}_{trade_date}")


def normalize_cache_code(ts_code: str) -> str:
    return str(ts_code).replace(".", "_")


def load_cached_stock_history_by_prefixes(
    ts_code: str,
    prefixes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    cache_dir = holder_cache_dir()
    if not cache_dir.exists():
        return pd.DataFrame()
    code_token = normalize_cache_code(ts_code)
    candidates: list[tuple[int, int, str, str, int, int, Path]] = []
    for prefix in prefixes:
        for path in cache_dir.glob(f"{prefix}_{code_token}_*.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or "trade_date" not in df.columns:
                continue
            values = df["trade_date"].astype(str)
            min_date = values.min()
            max_date = values.max()
            cover_end = int(max_date >= end_date)
            cover_start = int(min_date <= start_date)
            cover_full = int(cover_start and cover_end)
            span = int(len(df))
            mtime = int(path.stat().st_mtime)
            candidates.append((cover_full, cover_end, max_date, min_date, span, mtime, path))
    if not candidates:
        return pd.DataFrame()
    candidates.sort(reverse=True)
    best_path = candidates[0][6]
    try:
        df = pd.read_csv(best_path)
        log_step(f"stock-history cache hit {best_path.name} rows={len(df)}")
        return df
    except Exception:
        return pd.DataFrame()


def load_cached_holdertrade_range(start_date: str, end_date: str) -> pd.DataFrame:
    cache_dir = holder_cache_dir()
    if not cache_dir.exists():
        return pd.DataFrame()
    pattern = re.compile(r"^stk_holdertrade_(\d{8})_(\d{8})_[0-9a-f]{12}\.csv$")
    frames: list[pd.DataFrame] = []
    for path in cache_dir.glob("stk_holdertrade_*.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        left, right = match.group(1), match.group(2)
        if right < start_date or left > end_date:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    if "ann_date" in combined.columns:
        ann_dates = combined["ann_date"].fillna("").astype(str).str.replace("-", "", regex=False)
        combined = combined[(ann_dates >= start_date) & (ann_dates <= end_date)].copy()
        combined["ann_date"] = ann_dates[(ann_dates >= start_date) & (ann_dates <= end_date)]
    return combined


def load_or_fetch_trade_date_frame(pro, prefix: str, method_name: str, trade_date: str, sleep_sec: float = 0.0) -> pd.DataFrame:
    cached = load_cached_trade_date_frame(prefix, trade_date)
    if not cached.empty:
        return cached
    return safe_call(
        f"{prefix}_{trade_date}",
        getattr(pro, method_name, None),
        sleep_sec=sleep_sec,
        trade_date=trade_date,
    )


def build_cached_moneyflow_summary(
    pro,
    trade_dates: list[str],
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        df = load_or_fetch_trade_date_frame(pro, "moneyflow", "moneyflow", trade_date, sleep_sec=sleep_sec)
        if df.empty:
            continue
        work = df.copy()
        work["trade_date"] = str(trade_date)
        work["main_net_amount"] = compute_main_net_amount(work)
        frames.append(work)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["main_net_amount"] = pd.to_numeric(combined["main_net_amount"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    latest_3 = set(trade_dates[-3:])
    latest_5 = set(trade_dates[-5:])
    for ts_code, sub in combined.groupby("ts_code", dropna=False):
        ordered = sub.sort_values("trade_date").reset_index(drop=True)
        positive_mask = pd.to_numeric(ordered["main_net_amount"], errors="coerce").fillna(0.0) > 0
        consecutive_positive_days = 0
        for value in positive_mask.iloc[::-1].tolist():
            if value:
                consecutive_positive_days += 1
            else:
                break
        rows.append(
            {
                "ts_code": ts_code,
                "main_net_amount_3d": round(float(sub[sub["trade_date"].isin(latest_3)]["main_net_amount"].sum()), 0),
                "main_net_amount_5d": round(float(sub[sub["trade_date"].isin(latest_5)]["main_net_amount"].sum()), 0),
                "main_net_positive_days_3d": int((sub[sub["trade_date"].isin(latest_3)]["main_net_amount"] > 0).sum()),
                "main_net_positive_days_5d": int((sub[sub["trade_date"].isin(latest_5)]["main_net_amount"] > 0).sum()),
                "main_net_consecutive_days": int(consecutive_positive_days),
                "moneyflow_days": int(sub["trade_date"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_cached_margin_summary(
    pro,
    trade_dates: list[str],
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        df = load_or_fetch_trade_date_frame(pro, "margin_detail", "margin_detail", trade_date, sleep_sec=sleep_sec)
        if df.empty:
            continue
        keep_cols = [c for c in ["ts_code", "trade_date", "rzye", "rzmre", "rzche"] if c in df.columns]
        if keep_cols:
            frames.append(df[keep_cols].copy())
    if not frames:
        return pd.DataFrame()
    return build_margin_summary(pd.concat(frames, ignore_index=True), trade_dates=trade_dates)


def fetch_market_inputs_cached_first(
    pro,
    trade_dates: list[str],
    moneyflow_lookback_days: int,
    sleep_sec: float = 0.0,
) -> tuple[str, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not trade_dates:
        return "", [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    fallback_trade_date = trade_dates[-1]
    fallback_moneyflow_dates = trade_dates[-moneyflow_lookback_days:]
    fallback_daily_basic = pd.DataFrame()
    fallback_tech = pd.DataFrame()
    fallback_moneyflow = pd.DataFrame()
    for idx in range(len(trade_dates) - 1, -1, -1):
        trade_date = trade_dates[idx]
        moneyflow_dates = trade_dates[max(0, idx - moneyflow_lookback_days + 1) : idx + 1]
        daily_basic_df = load_or_fetch_trade_date_frame(pro, "daily_basic", "daily_basic", trade_date, sleep_sec=0.0)
        tech_df = load_or_fetch_trade_date_frame(pro, "stk_factor_pro", "stk_factor_pro", trade_date, sleep_sec=0.0)
        if tech_df.empty:
            tech_df = build_market_technical_snapshot_from_cached_history(trade_date)
        moneyflow_df = build_cached_moneyflow_summary(pro, moneyflow_dates, sleep_sec=sleep_sec)

        fallback_trade_date = trade_date
        fallback_moneyflow_dates = moneyflow_dates
        fallback_daily_basic = daily_basic_df
        fallback_tech = tech_df
        fallback_moneyflow = moneyflow_df

        if not daily_basic_df.empty and not tech_df.empty and not moneyflow_df.empty:
            return trade_date, moneyflow_dates, daily_basic_df, tech_df, moneyflow_df

    return (
        fallback_trade_date,
        fallback_moneyflow_dates,
        fallback_daily_basic,
        fallback_tech,
        fallback_moneyflow,
    )


def compute_forward_returns(daily_df: pd.DataFrame, signal_date: str, hold_days: list[int]) -> dict[str, Any]:
    if daily_df.empty:
        return {}
    work = daily_df.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work = work[work["trade_date"] >= signal_date].copy()
    if work.empty:
        return {}
    work = work.sort_values("trade_date").reset_index(drop=True)
    for column in ["open", "close"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    signal_rows = work.index[work["trade_date"] == signal_date].tolist()
    if not signal_rows:
        return {}
    signal_idx = signal_rows[-1]
    entry_idx = signal_idx + 1
    if entry_idx >= len(work):
        return {}
    entry_trade_date = str(work.iloc[entry_idx]["trade_date"])
    entry_open = work.iloc[entry_idx]["open"]
    payload: dict[str, Any] = {
        "entry_trade_date": entry_trade_date,
        "entry_open": float(entry_open) if pd.notna(entry_open) else None,
    }
    for day in hold_days:
        target_idx = signal_idx + day
        key = f"return_open_to_close_{day}d_pct"
        if pd.isna(entry_open) or float(entry_open) <= 0 or target_idx >= len(work):
            payload[key] = None
            continue
        close_value = work.iloc[target_idx]["close"]
        if pd.isna(close_value):
            payload[key] = None
            continue
        payload[key] = round((float(close_value) / float(entry_open) - 1.0) * 100.0, 4)
    return payload


def summarize_returns(label: str, df: pd.DataFrame, hold_days: list[int]) -> dict[str, Any]:
    if df.empty:
        payload: dict[str, Any] = {"label": label, "rows": 0}
        for day in hold_days:
            payload[f"avg_{day}d_pct"] = None
            payload[f"median_{day}d_pct"] = None
            payload[f"win_rate_{day}d_pct"] = None
            payload[f"count_{day}d"] = 0
        return payload

    summary: dict[str, Any] = {"label": label, "rows": int(len(df))}
    if "signal_date" in df.columns:
        dates = df["signal_date"].dropna().astype(str)
        summary["signal_days"] = int(dates.nunique())
        summary["first_signal_date"] = dates.min() if not dates.empty else None
        summary["last_signal_date"] = dates.max() if not dates.empty else None
    for day in hold_days:
        column = f"return_open_to_close_{day}d_pct"
        values = pd.to_numeric(df.get(column), errors="coerce").dropna()
        summary[f"avg_{day}d_pct"] = round(float(values.mean()), 4) if not values.empty else None
        summary[f"median_{day}d_pct"] = round(float(values.median()), 4) if not values.empty else None
        summary[f"win_rate_{day}d_pct"] = round(float((values > 0).mean() * 100.0), 2) if not values.empty else None
        summary[f"count_{day}d"] = int(values.count())
    return summary


def duplicate_summary(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if df.empty or column not in df.columns:
        return {
            "column": column,
            "rows": 0,
            "unique_values": 0,
            "duplicate_row_count": 0,
            "duplicate_value_count": 0,
            "top_duplicates": [],
        }
    counts = df[column].fillna("").astype(str).value_counts()
    duplicate_counts = counts[counts > 1]
    top_duplicates = [
        {"value": value, "count": int(count)}
        for value, count in duplicate_counts.head(10).items()
    ]
    return {
        "column": column,
        "rows": int(len(df)),
        "unique_values": int(counts.shape[0]),
        "duplicate_row_count": int((duplicate_counts - 1).sum()) if not duplicate_counts.empty else 0,
        "duplicate_value_count": int(len(duplicate_counts)),
        "top_duplicates": top_duplicates,
    }


def build_review_report(
    baseline_stats: dict[str, Any],
    optimized_final_summary: dict[str, Any],
    final_signals_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    hold_days: list[int],
) -> str:
    stage1_summary = baseline_stats.get("stage1_summary", {})
    lines: list[str] = []
    lines.append("# 核心高管连增臻选 final 轻确认对拍")
    lines.append("")
    lines.append("## 样本概览")
    scan_range = baseline_stats.get("range", {})
    lines.append(f"- 扫描区间：`{scan_range.get('start_date')} ~ {scan_range.get('end_date')}`")
    lines.append(f"- 扫描交易日：`{scan_range.get('trade_days_scanned')}`")
    lines.append(f"- 候选交易日：`{baseline_stats.get('candidate_trade_day_count')}`")
    lines.append(f"- 新版 final 真信号：`{optimized_final_summary.get('rows', 0)}`")
    lines.append(f"- 基线 stage1 样本：`{stage1_summary.get('rows', 0)}`")
    lines.append("")
    lines.append("## 绩效总表")
    header = "| 口径 | 样本数 | 3日均值 | 5日均值 | 10日均值 | 3日胜率 | 5日胜率 | 10日胜率 |"
    divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines.extend([header, divider])
    lines.append(
        f"| 新版 final | {optimized_final_summary.get('rows', 0)} | "
        f"{optimized_final_summary.get('avg_3d_pct')} | {optimized_final_summary.get('avg_5d_pct')} | {optimized_final_summary.get('avg_10d_pct')} | "
        f"{optimized_final_summary.get('win_rate_3d_pct')}% | {optimized_final_summary.get('win_rate_5d_pct')}% | {optimized_final_summary.get('win_rate_10d_pct')}% |"
    )
    lines.append(
        f"| 基线 stage1 | {stage1_summary.get('rows', 0)} | "
        f"{stage1_summary.get('avg_3d_pct')} | {stage1_summary.get('avg_5d_pct')} | {stage1_summary.get('avg_10d_pct')} | "
        f"{stage1_summary.get('win_rate_3d_pct')}% | {stage1_summary.get('win_rate_5d_pct')}% | {stage1_summary.get('win_rate_10d_pct')}% |"
    )
    lines.append("")
    lines.append("## 新版 final 明细")
    if final_signals_df.empty:
        lines.append("- 当前对拍结果没有最终信号。")
    else:
        lines.append("| 日期 | 股票 | 分数 | 3日 | 5日 | 10日 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for _, row in final_signals_df.iterrows():
            lines.append(
                f"| {row.get('signal_date')} | {row.get('ts_code')} {row.get('name', '')} | "
                f"{row.get('total_score')} | {row.get('return_open_to_close_3d_pct')} | "
                f"{row.get('return_open_to_close_5d_pct')} | {row.get('return_open_to_close_10d_pct')} |"
            )
    lines.append("")
    lines.append("## 重复信号观察")
    if "repeat_recent_signal_hit" in final_signals_df.columns:
        repeat_hits = int(pd.to_numeric(final_signals_df["repeat_recent_signal_hit"], errors="coerce").fillna(False).astype(bool).sum())
        lines.append(f"- 新版 final 中带重复信号痕迹的条目：`{repeat_hits}`")
    if not progress_df.empty:
        skipped = int((progress_df["status"] != "ok").sum()) if "status" in progress_df.columns else 0
        if skipped:
            lines.append(f"- 对拍过程中有 `skip/error` 日期：`{skipped}`，需查看 `progress.csv` 明细。")
    lines.append("")
    lines.append("## 诊断")
    lines.append("- 这轮对拍只改 `final` 层，`stage1` 入口和事件识别逻辑保持不变。")
    lines.append("- 如果新版 final 的 5/10 日仍明显弱于基线 stage1，说明下一步应继续优化最终排序和重复放行条件，而不是先放宽事件入口。")
    return "\n".join(lines)


def fetch_full_price_bundle(
    pro,
    ts_code: str,
    start_date: str,
    end_date: str,
    sleep_sec: float,
) -> dict[str, pd.DataFrame]:
    daily_df = load_cached_stock_history_by_prefixes(
        ts_code,
        prefixes=["core_mgmt_daily", "daily", "perf_daily", "target_daily", "eval_forward_daily"],
        start_date=start_date,
        end_date=end_date,
    )
    if daily_df.empty:
        daily_df = safe_call(
            f"core_mgmt_daily_{ts_code}",
            getattr(pro, "daily", None),
            sleep_sec=sleep_sec,
            retries=1,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    adj_df = load_cached_stock_history_by_prefixes(
        ts_code,
        prefixes=["core_mgmt_adj_factor", "adj_factor", "target_adj"],
        start_date=start_date,
        end_date=end_date,
    )
    if adj_df.empty:
        adj_df = safe_call(
            f"core_mgmt_adj_factor_{ts_code}",
            getattr(pro, "adj_factor", None),
            sleep_sec=sleep_sec,
            retries=1,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
    if adj_df.empty and not daily_df.empty and "trade_date" in daily_df.columns:
        adj_df = daily_df[["ts_code", "trade_date"]].copy()
        adj_df["adj_factor"] = 1.0
        log_step(f"adj-factor fallback ts_code={ts_code} rows={len(adj_df)}")
    return {"daily_df": daily_df, "adj_df": adj_df}


def evaluate_trade_date(
    pro,
    trade_date: str,
    config_overrides: dict[str, Any],
    api_sleep_sec: float,
    stock_basic_all: pd.DataFrame,
    price_bundle_cache: dict[str, dict[str, pd.DataFrame]],
    price_history_start_date: str,
    global_end_date: str,
    recent_final_signals_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = CoreManagementAccumulationConfig.for_end_date(trade_date, **config_overrides)
    recent_trade_dates = get_recent_open_trade_dates(
        pro,
        config.end_date,
        count=max(config.recent_wave_trade_days + 5, config.moneyflow_lookback_days + 5, 25),
    )
    latest_trade_date, market_moneyflow_dates, daily_basic_latest, tech_latest, moneyflow_summary = fetch_market_inputs_cached_first(
        pro,
        recent_trade_dates,
        moneyflow_lookback_days=config.moneyflow_lookback_days,
        sleep_sec=api_sleep_sec,
    )
    recent_wave_trade_dates = [value for value in recent_trade_dates if value <= latest_trade_date][-config.recent_wave_trade_days :]
    margin_trade_dates = [value for value in recent_trade_dates if value <= latest_trade_date][-3:]

    holdertrade_raw = load_cached_holdertrade_range(config.ann_start_date, latest_trade_date)
    if holdertrade_raw.empty:
        holdertrade_raw = fetch_holdertrade_events(
            pro,
            config.ann_start_date,
            latest_trade_date,
            chunk_days=config.event_chunk_days,
            sleep_sec=api_sleep_sec,
        )
    market_snapshot = build_market_snapshot(stock_basic_all, daily_basic_latest, tech_latest, moneyflow_summary)
    margin_summary = build_cached_margin_summary(pro, margin_trade_dates, sleep_sec=api_sleep_sec)
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
    best_wave_per_stock = select_best_wave_per_stock(wave_details)
    if best_wave_per_stock.empty:
        progress = {
            "trade_date": trade_date,
            "latest_trade_date": latest_trade_date,
            "stage1_rows": 0,
            "final_rows": 0,
            "status": "ok",
        }
        return progress, pd.DataFrame(), pd.DataFrame()

    from core_management_accumulation_strategy import build_preliminary_candidate_flags

    preliminary_flags = best_wave_per_stock.apply(
        lambda row: build_preliminary_candidate_flags(row.to_dict(), config),
        axis=1,
        result_type="expand",
    )
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
    if stage1_candidates.empty:
        progress = {
            "trade_date": trade_date,
            "latest_trade_date": latest_trade_date,
            "stage1_rows": 0,
            "final_rows": 0,
            "status": "ok",
        }
        return progress, pd.DataFrame(), pd.DataFrame()

    deep_rows: list[dict[str, Any]] = []
    for _, row in stage1_candidates.iterrows():
        ts_code = str(row["ts_code"])
        if ts_code not in price_bundle_cache:
            price_bundle_cache[ts_code] = fetch_full_price_bundle(
                pro,
                ts_code=ts_code,
                start_date=price_history_start_date,
                end_date=global_end_date,
                sleep_sec=api_sleep_sec,
            )
        bundle = price_bundle_cache[ts_code]
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
    merged = stage1_candidates.merge(deep_metrics_df, on="ts_code", how="left")
    final_candidates = score_final_candidates(merged, config, recent_final_signals=recent_final_signals_df)
    progress = {
        "trade_date": trade_date,
        "latest_trade_date": latest_trade_date,
        "stage1_rows": int(len(stage1_candidates)),
        "final_rows": int(len(final_candidates)),
        "status": "ok",
    }
    return progress, stage1_candidates, final_candidates


def persist_outputs(
    export_dir: Path,
    summary_payload: dict[str, Any],
    final_signals_df: pd.DataFrame,
    progress_df: pd.DataFrame,
) -> None:
    final_export = final_signals_df.copy()
    if final_export.empty:
        final_export = pd.DataFrame(
            columns=[
                "signal_date",
                "screen_trade_date",
                "ts_code",
                "name",
                "total_score",
                "adjusted_total_score",
                "base_total_score",
                "wave_signature",
                "wave_first_date",
                "wave_last_date",
                "repeat_recent_signal_hit",
                "repeat_signal_distance_trade_days",
                "repeat_allowed_override",
                "freshness_penalty_score",
                "repeat_penalty_score",
                "retrigger_quality_score",
                "final_confirmation_score",
                "entry_trade_date",
                "entry_open",
                "return_open_to_close_3d_pct",
                "return_open_to_close_5d_pct",
                "return_open_to_close_10d_pct",
            ]
        )
    (export_dir / "review_summary.json").write_text(
        json.dumps(json_safe(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_export.to_csv(export_dir / "optimized_final_signals.csv", index=False)
    progress_df.to_csv(export_dir / "progress.csv", index=False)
    report_text = build_review_report(
        baseline_stats=summary_payload["baseline_stats"],
        optimized_final_summary=summary_payload["optimized_final_summary"],
        final_signals_df=final_signals_df,
        progress_df=progress_df,
        hold_days=summary_payload["hold_days"],
    )
    (export_dir / "review_report.md").write_text(report_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    stats_path = Path(args.stats_json).expanduser().resolve()
    if not stats_path.exists():
        raise SystemExit(f"Stats JSON not found: {stats_path}")

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    ensure_token(token)
    baseline_stats = load_json_object(stats_path)
    config_overrides = load_config_overrides(args)
    hold_days = parse_hold_days(args.hold_days)
    candidate_dates = build_candidate_dates(baseline_stats, max_dates=args.max_dates)
    if not candidate_dates:
        raise SystemExit("No candidate dates found in baseline stats.")

    export_dir = export_root_dir(args.export_root) / f"core_management_final_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"candidate_dates={len(candidate_dates)} export_dir={export_dir}")

    pro = configure_tushare_client(token, custom_http_url=custom_http_url)
    stock_basic_all = fetch_stock_basic_all(pro)
    global_end_date = str(baseline_stats.get("range", {}).get("end_date") or candidate_dates[-1])
    price_history_start_date = (pd.Timestamp(candidate_dates[0]) - pd.Timedelta(days=max(420, 250 * 2))).strftime("%Y%m%d")

    price_bundle_cache: dict[str, dict[str, pd.DataFrame]] = {}
    recent_final_signals_rows: list[dict[str, Any]] = []
    final_signal_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []

    for idx, trade_date in enumerate(candidate_dates, start=1):
        log_step(f"evaluate {idx}/{len(candidate_dates)} trade_date={trade_date}")
        recent_final_df = pd.DataFrame(recent_final_signals_rows)
        try:
            progress_row, _, final_candidates = evaluate_trade_date(
                pro=pro,
                trade_date=trade_date,
                config_overrides=config_overrides,
                api_sleep_sec=args.api_sleep_sec,
                stock_basic_all=stock_basic_all,
                price_bundle_cache=price_bundle_cache,
                price_history_start_date=price_history_start_date,
                global_end_date=global_end_date,
                recent_final_signals_df=recent_final_df,
            )
        except Exception as exc:
            progress_row = {
                "trade_date": trade_date,
                "latest_trade_date": None,
                "stage1_rows": None,
                "final_rows": None,
                "status": "error",
                "error": str(exc),
            }
            progress_rows.append(progress_row)
            summary_payload = {
                "strategy_id": "core_management_accumulation",
                "strategy_name": STRATEGY_NAME,
                "baseline_stats": baseline_stats,
                "hold_days": hold_days,
                "optimized_final_summary": summarize_returns("optimized_final", pd.DataFrame(final_signal_rows), hold_days),
                "ts_code_duplicates": duplicate_summary(pd.DataFrame(final_signal_rows), "ts_code"),
                "wave_signature_duplicates": duplicate_summary(pd.DataFrame(final_signal_rows), "wave_signature"),
                "candidate_dates": candidate_dates,
                "export_dir": str(export_dir),
            }
            persist_outputs(export_dir, summary_payload, pd.DataFrame(final_signal_rows), pd.DataFrame(progress_rows))
            log_step(f"trade_date={trade_date} error={exc}")
            continue

        progress_rows.append(progress_row)
        if final_candidates.empty:
            summary_payload = {
                "strategy_id": "core_management_accumulation",
                "strategy_name": STRATEGY_NAME,
                "baseline_stats": baseline_stats,
                "hold_days": hold_days,
                "optimized_final_summary": summarize_returns("optimized_final", pd.DataFrame(final_signal_rows), hold_days),
                "ts_code_duplicates": duplicate_summary(pd.DataFrame(final_signal_rows), "ts_code"),
                "wave_signature_duplicates": duplicate_summary(pd.DataFrame(final_signal_rows), "wave_signature"),
                "candidate_dates": candidate_dates,
                "export_dir": str(export_dir),
            }
            persist_outputs(export_dir, summary_payload, pd.DataFrame(final_signal_rows), pd.DataFrame(progress_rows))
            continue

        best_pick = final_candidates.head(1).copy()
        row = best_pick.iloc[0].to_dict()
        ts_code = str(row.get("ts_code") or "")
        bundle = price_bundle_cache.get(ts_code, {})
        forward_payload = compute_forward_returns(bundle.get("daily_df", pd.DataFrame()), progress_row["latest_trade_date"], hold_days)
        signal_row = {
            "signal_date": progress_row["latest_trade_date"],
            "screen_trade_date": trade_date,
            "ts_code": ts_code,
            "name": row.get("name"),
            "total_score": row.get("total_score"),
            "adjusted_total_score": row.get("adjusted_total_score"),
            "base_total_score": row.get("base_total_score"),
            "wave_signature": row.get("wave_signature") or build_wave_signature(row),
            "wave_first_date": row.get("wave_first_date"),
            "wave_last_date": row.get("wave_last_date"),
            "repeat_recent_signal_hit": row.get("repeat_recent_signal_hit"),
            "repeat_signal_distance_trade_days": row.get("repeat_signal_distance_trade_days"),
            "repeat_allowed_override": row.get("repeat_allowed_override"),
            "freshness_penalty_score": row.get("freshness_penalty_score"),
            "repeat_penalty_score": row.get("repeat_penalty_score"),
            "retrigger_quality_score": row.get("retrigger_quality_score"),
            "final_confirmation_score": row.get("final_confirmation_score"),
            **forward_payload,
        }
        final_signal_rows.append(signal_row)
        recent_final_signals_rows.append(
            {
                "signal_date": signal_row["signal_date"],
                "ts_code": signal_row["ts_code"],
                "wave_signature": signal_row["wave_signature"],
                "wave_last_date": signal_row["wave_last_date"],
            }
        )

        summary_payload = {
            "strategy_id": "core_management_accumulation",
            "strategy_name": STRATEGY_NAME,
            "baseline_stats": baseline_stats,
            "hold_days": hold_days,
            "optimized_final_summary": summarize_returns("optimized_final", pd.DataFrame(final_signal_rows), hold_days),
            "ts_code_duplicates": duplicate_summary(pd.DataFrame(final_signal_rows), "ts_code"),
            "wave_signature_duplicates": duplicate_summary(pd.DataFrame(final_signal_rows), "wave_signature"),
            "candidate_dates": candidate_dates,
            "export_dir": str(export_dir),
        }
        persist_outputs(export_dir, summary_payload, pd.DataFrame(final_signal_rows), pd.DataFrame(progress_rows))

    final_signals_df = pd.DataFrame(final_signal_rows)
    progress_df = pd.DataFrame(progress_rows)
    summary_payload = {
        "strategy_id": "core_management_accumulation",
        "strategy_name": STRATEGY_NAME,
        "baseline_stats": baseline_stats,
        "hold_days": hold_days,
        "optimized_final_summary": summarize_returns("optimized_final", final_signals_df, hold_days),
        "ts_code_duplicates": duplicate_summary(final_signals_df, "ts_code"),
        "wave_signature_duplicates": duplicate_summary(final_signals_df, "wave_signature"),
        "candidate_dates": candidate_dates,
        "export_dir": str(export_dir),
    }
    persist_outputs(export_dir, summary_payload, final_signals_df, progress_df)
    print(json.dumps(json_safe(summary_payload), ensure_ascii=False, indent=2))
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
