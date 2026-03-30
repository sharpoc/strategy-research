from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

import numpy as np
import pandas as pd
import tushare as ts


API_ERROR_LOG: dict[str, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a user-provided watchlist and pick the best stock.")
    parser.add_argument("ts_codes", nargs="*", help="Stock codes such as 603766.SH 002286.SZ or comma-separated values.")
    parser.add_argument("--end-date", default="", help="Requested end date in YYYYMMDD. Default: today with 20:00 cutoff.")
    parser.add_argument("--ann-lookback-days", type=int, default=45, help="Holdertrade lookback window in days.")
    parser.add_argument("--moneyflow-lookback-days", type=int, default=5, help="Moneyflow lookback days.")
    parser.add_argument("--cyq-lookback-days", type=int, default=20, help="Chip data lookback days.")
    parser.add_argument("--cutoff-hour", type=int, default=20, help="Use previous trading day before this hour.")
    parser.add_argument("--show-top", type=int, default=10, help="Rows to print from the watchlist ranking.")
    return parser.parse_args()


def split_codes(values: list[str]) -> list[str]:
    parts: list[str] = []
    for value in values:
        for piece in re.split(r"[\s,，]+", str(value).strip()):
            if piece:
                parts.append(piece)
    return parts


def normalize_ts_code(value: str) -> str:
    code = value.strip().upper()
    if not code:
        return code
    if "." in code:
        return code
    if re.fullmatch(r"\d{6}", code):
        return f"{code}.SH" if code.startswith(("5", "6", "9")) else f"{code}.SZ"
    return code


def ensure_token(token: str) -> None:
    if not token or token.startswith("PASTE_"):
        raise SystemExit("Missing TUSHARE_TOKEN in environment.")


def configure_proxy_bypass(custom_http_url: str) -> None:
    if not custom_http_url:
        return
    parsed = urlparse(custom_http_url)
    host = (parsed.hostname or "").strip()
    no_proxy_tokens: list[str] = []
    for value in [os.getenv("NO_PROXY", ""), os.getenv("no_proxy", ""), "127.0.0.1", "localhost", host]:
        for token in str(value).split(","):
            token = token.strip()
            if token and token not in no_proxy_tokens:
                no_proxy_tokens.append(token)
    if no_proxy_tokens:
        merged = ",".join(no_proxy_tokens)
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)


def configure_tushare_client(pro, token: str, use_custom_http_endpoint: bool, custom_http_url: str):
    if use_custom_http_endpoint and custom_http_url:
        configure_proxy_bypass(custom_http_url)
        pro._DataApi__token = token
        pro._DataApi__http_url = custom_http_url
    return pro


def to_float(value):
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    try:
        return float(value)
    except Exception:
        return None


def to_number(value, digits: int = 2):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return round(float(value), digits)
    except Exception:
        return value


def to_bool(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def clip_score(value, low: float, high: float) -> float:
    if value is None:
        return low
    return float(min(max(value, low), high))


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def safe_call(label: str, fn, sleep_sec: float = 0.0, retries: int = 2, **kwargs) -> pd.DataFrame:
    if fn is None:
        return pd.DataFrame()
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            df = fn(**kwargs)
            API_ERROR_LOG.pop(label, None)
            if sleep_sec:
                time.sleep(sleep_sec)
            if df is None:
                return pd.DataFrame()
            return df.copy()
        except Exception as exc:
            last_exc = exc
            API_ERROR_LOG[label] = str(exc)
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
    print(f"[{label}] 调用失败: {last_exc}")
    return pd.DataFrame()


def sort_desc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["trade_date", "ann_date", "end_date", "f_ann_date"]:
        if col in df.columns:
            return df.sort_values(col, ascending=False).reset_index(drop=True)
    return df.reset_index(drop=True)


def latest_row(df: pd.DataFrame) -> dict:
    ordered = sort_desc(df)
    if ordered.empty:
        return {}
    row = ordered.iloc[0]
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def rank_pct(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranked = numeric.rank(pct=True, method="average")
    return ranked.fillna(0.5)


def ensure_columns(df: pd.DataFrame, columns: list[str], fill_value=np.nan) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column not in work.columns:
            work[column] = fill_value
    return work


def choose_screen_end_date(now_ts: pd.Timestamp, end_date: str, today_str: str, cutoff_hour: int = 20) -> str:
    requested = end_date or today_str
    if requested == today_str and int(now_ts.hour) < int(cutoff_hour):
        return (now_ts.normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    return requested


def get_recent_open_trade_dates(pro, end_date: str, count: int = 10) -> list[str]:
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(20, count * 4))).strftime("%Y%m%d")
    cal = safe_call(
        "trade_cal",
        getattr(pro, "trade_cal", None),
        start_date=start_date,
        end_date=end_date,
        is_open="1",
    )
    if cal.empty:
        return [end_date]
    date_col = "cal_date" if "cal_date" in cal.columns else "trade_date"
    dates = sorted(cal[date_col].dropna().astype(str).unique().tolist())
    return dates[-count:] if dates else [end_date]


def compute_main_net_amount(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if "net_mf_amount" in df.columns:
        return pd.to_numeric(df["net_mf_amount"], errors="coerce").fillna(0.0)
    buys = pd.Series(0.0, index=df.index)
    sells = pd.Series(0.0, index=df.index)
    for col in ["buy_lg_amount", "buy_elg_amount"]:
        if col in df.columns:
            buys = buys.add(pd.to_numeric(df[col], errors="coerce").fillna(0.0), fill_value=0.0)
    for col in ["sell_lg_amount", "sell_elg_amount"]:
        if col in df.columns:
            sells = sells.add(pd.to_numeric(df[col], errors="coerce").fillna(0.0), fill_value=0.0)
    return buys - sells


def fetch_recent_moneyflow_summary(pro, trade_dates: list[str], sleep_sec: float = 0.0) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        df = safe_call(
            f"moneyflow_{trade_date}",
            getattr(pro, "moneyflow", None),
            sleep_sec=sleep_sec,
            trade_date=trade_date,
        )
        if not df.empty:
            work = df.copy()
            work["trade_date"] = trade_date
            work["main_net_amount"] = compute_main_net_amount(work)
            frames.append(work)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["main_net_amount"] = pd.to_numeric(combined["main_net_amount"], errors="coerce").fillna(0.0)
    rows: list[dict] = []
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
                "main_net_amount_3d": to_number(sub[sub["trade_date"].isin(latest_3)]["main_net_amount"].sum(), 0),
                "main_net_amount_5d": to_number(sub[sub["trade_date"].isin(latest_5)]["main_net_amount"].sum(), 0),
                "main_net_positive_days_3d": int((sub[sub["trade_date"].isin(latest_3)]["main_net_amount"] > 0).sum()),
                "main_net_positive_days_5d": int((sub[sub["trade_date"].isin(latest_5)]["main_net_amount"] > 0).sum()),
                "main_net_consecutive_days": int(consecutive_positive_days),
                "moneyflow_days": int(sub["trade_date"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def fetch_latest_complete_market_inputs(
    pro,
    trade_dates: list[str],
    moneyflow_lookback_days: int,
    sleep_sec: float = 0.0,
) -> tuple[str, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fallback_trade_date = trade_dates[-1]
    fallback_moneyflow_dates = trade_dates[-moneyflow_lookback_days:]
    fallback_daily_basic = pd.DataFrame()
    fallback_tech = pd.DataFrame()
    fallback_moneyflow = pd.DataFrame()

    for idx in range(len(trade_dates) - 1, -1, -1):
        trade_date = trade_dates[idx]
        moneyflow_dates = trade_dates[max(0, idx - moneyflow_lookback_days + 1) : idx + 1]
        daily_basic_df = safe_call(
            f"daily_basic_{trade_date}",
            getattr(pro, "daily_basic", None),
            trade_date=trade_date,
        )
        tech_df = safe_call(
            f"stk_factor_pro_{trade_date}",
            getattr(pro, "stk_factor_pro", None),
            trade_date=trade_date,
        )
        moneyflow_df = fetch_recent_moneyflow_summary(pro, moneyflow_dates, sleep_sec=sleep_sec)

        fallback_trade_date = trade_date
        fallback_moneyflow_dates = moneyflow_dates
        fallback_daily_basic = daily_basic_df
        fallback_tech = tech_df
        fallback_moneyflow = moneyflow_df

        if not daily_basic_df.empty and not tech_df.empty and not moneyflow_df.empty:
            return trade_date, moneyflow_dates, daily_basic_df, tech_df, moneyflow_df

    return fallback_trade_date, fallback_moneyflow_dates, fallback_daily_basic, fallback_tech, fallback_moneyflow


def build_market_snapshot(
    stock_basic_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    tech_df: pd.DataFrame,
    moneyflow_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    basic_cols = [c for c in ["ts_code", "symbol", "name", "industry", "market", "list_date"] if c in stock_basic_df.columns]
    snapshot = stock_basic_df[basic_cols].drop_duplicates(subset=["ts_code"]).copy()

    if not daily_basic_df.empty:
        daily_cols = [
            c
            for c in [
                "ts_code",
                "trade_date",
                "close",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe",
                "pe_ttm",
                "pb",
                "ps_ttm",
                "total_mv",
                "circ_mv",
            ]
            if c in daily_basic_df.columns
        ]
        snapshot = snapshot.merge(daily_basic_df[daily_cols].drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")

    if not tech_df.empty:
        tech_cols = [
            c
            for c in [
                "ts_code",
                "trade_date",
                "close_qfq",
                "ma_qfq_5",
                "ma_qfq_10",
                "ma_qfq_20",
                "ma_qfq_60",
                "ma_qfq_250",
                "macd_dif_qfq",
                "macd_dea_qfq",
                "vol_ratio",
            ]
            if c in tech_df.columns
        ]
        snapshot = snapshot.merge(tech_df[tech_cols].drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")

    if not moneyflow_summary_df.empty:
        snapshot = snapshot.merge(moneyflow_summary_df, on="ts_code", how="left")

    snapshot["pb_num"] = pd.to_numeric(snapshot.get("pb"), errors="coerce")
    if "industry" in snapshot.columns:
        snapshot["industry_pb_pct_rank"] = snapshot.groupby("industry")["pb_num"].rank(pct=True, method="average")
        snapshot["industry_pb_pct_rank"] = snapshot["industry_pb_pct_rank"].fillna(0.5)
    else:
        snapshot["industry_pb_pct_rank"] = 0.5

    if "volume_ratio" in snapshot.columns:
        snapshot["volume_ratio"] = pd.to_numeric(snapshot["volume_ratio"], errors="coerce")
    if "vol_ratio" in snapshot.columns:
        vol_ratio_series = pd.to_numeric(snapshot["vol_ratio"], errors="coerce")
        if "volume_ratio" in snapshot.columns:
            snapshot["volume_ratio"] = snapshot["volume_ratio"].fillna(vol_ratio_series)
        else:
            snapshot["volume_ratio"] = vol_ratio_series

    return snapshot


def build_market_regime_snapshot(snapshot: pd.DataFrame) -> dict:
    if snapshot.empty:
        return {
            "market_regime": "unknown",
            "market_trend_breadth": None,
            "market_flow_breadth": None,
            "market_hot_ratio": None,
            "market_regime_score": None,
        }

    trend_mask = pd.Series(False, index=snapshot.index)
    if {"close_qfq", "ma_qfq_20"}.issubset(snapshot.columns):
        trend_mask = (
            pd.to_numeric(snapshot["close_qfq"], errors="coerce")
            > pd.to_numeric(snapshot["ma_qfq_20"], errors="coerce")
        ).fillna(False)

    flow_mask = pd.Series(False, index=snapshot.index)
    if "main_net_amount_3d" in snapshot.columns:
        flow_mask = pd.to_numeric(snapshot["main_net_amount_3d"], errors="coerce").fillna(0.0) > 0

    hot_mask = pd.Series(False, index=snapshot.index)
    if "volume_ratio" in snapshot.columns:
        hot_mask = pd.to_numeric(snapshot["volume_ratio"], errors="coerce").fillna(0.0) >= 1.2

    trend_breadth = float(trend_mask.mean()) if len(snapshot) else 0.0
    flow_breadth = float(flow_mask.mean()) if len(snapshot) else 0.0
    hot_ratio = float(hot_mask.mean()) if len(snapshot) else 0.0
    regime_score = trend_breadth * 50 + flow_breadth * 35 + hot_ratio * 15

    if regime_score >= 58:
        regime = "risk_on"
    elif regime_score >= 46:
        regime = "neutral"
    else:
        regime = "defensive"

    return {
        "market_regime": regime,
        "market_trend_breadth": to_number(trend_breadth * 100),
        "market_flow_breadth": to_number(flow_breadth * 100),
        "market_hot_ratio": to_number(hot_ratio * 100),
        "market_regime_score": to_number(regime_score),
    }


def build_qfq_daily(daily_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()

    work = daily_df.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work["trade_date_dt"] = pd.to_datetime(work["trade_date"], format="%Y%m%d", errors="coerce")
    work = work.dropna(subset=["trade_date_dt"]).sort_values("trade_date_dt").reset_index(drop=True)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")

    if adj_df.empty or "adj_factor" not in adj_df.columns:
        work["close_qfq_calc"] = work["close"]
        return work

    factors = adj_df[["trade_date", "adj_factor"]].copy()
    factors["trade_date"] = factors["trade_date"].astype(str)
    factors["trade_date_dt"] = pd.to_datetime(factors["trade_date"], format="%Y%m%d", errors="coerce")
    factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="coerce")
    work = work.merge(factors[["trade_date_dt", "adj_factor"]], on="trade_date_dt", how="left")

    valid_factor = work["adj_factor"].dropna()
    if valid_factor.empty or valid_factor.iloc[-1] == 0:
        work["close_qfq_calc"] = work["close"]
    else:
        latest_factor = valid_factor.iloc[-1]
        work["close_qfq_calc"] = work["close"] * work["adj_factor"] / latest_factor
    return work


def summarize_price_metrics(daily_df: pd.DataFrame, adj_df: pd.DataFrame, window: int = 250) -> dict:
    qfq = build_qfq_daily(daily_df, adj_df)
    if qfq.empty or "close_qfq_calc" not in qfq.columns:
        return {
            "latest_close_qfq_calc": None,
            "price_position_250": None,
            "range_low_250": None,
            "range_high_250": None,
            "return_20d": None,
            "return_60d": None,
        }

    recent = qfq.tail(window).copy()
    closes = recent["close_qfq_calc"].dropna()
    if closes.empty:
        return {
            "latest_close_qfq_calc": None,
            "price_position_250": None,
            "range_low_250": None,
            "range_high_250": None,
            "return_20d": None,
            "return_60d": None,
        }

    latest_close = closes.iloc[-1]
    low_250 = closes.min()
    high_250 = closes.max()
    position = None if high_250 == low_250 else (latest_close - low_250) / (high_250 - low_250)

    ordered = qfq["close_qfq_calc"].dropna().reset_index(drop=True)
    return_20d = None
    return_60d = None
    if len(ordered) > 20 and ordered.iloc[-21] != 0:
        return_20d = (ordered.iloc[-1] / ordered.iloc[-21] - 1.0) * 100
    if len(ordered) > 60 and ordered.iloc[-61] != 0:
        return_60d = (ordered.iloc[-1] / ordered.iloc[-61] - 1.0) * 100

    return {
        "latest_close_qfq_calc": to_number(latest_close),
        "price_position_250": to_number(position, 4),
        "range_low_250": to_number(low_250),
        "range_high_250": to_number(high_250),
        "return_20d": to_number(return_20d),
        "return_60d": to_number(return_60d),
    }


def summarize_indicator_metrics(indicator_df: pd.DataFrame) -> dict:
    latest = latest_row(indicator_df)
    return {
        "report_period": latest.get("end_date"),
        "roe": to_number(latest.get("roe")),
        "gross_margin": to_number(latest.get("gross_margin") or latest.get("grossprofit_margin")),
        "dt_netprofit_yoy": to_number(latest.get("dt_netprofit_yoy")),
        "netprofit_yoy": to_number(latest.get("netprofit_yoy")),
        "ocf_yoy": to_number(latest.get("ocf_yoy")),
        "q_sales_yoy": to_number(latest.get("q_sales_yoy") or latest.get("tr_yoy") or latest.get("or_yoy")),
        "debt_to_assets": to_number(latest.get("debt_to_assets")),
    }


def summarize_forecast_metrics(forecast_df: pd.DataFrame) -> dict:
    latest = latest_row(forecast_df)
    forecast_type = str(latest.get("type") or "")
    negative = bool(forecast_type) and (("亏" in forecast_type) or (forecast_type in {"预减", "略减", "续亏", "首亏"}))
    return {
        "forecast_ann_date": latest.get("ann_date"),
        "forecast_type": forecast_type or None,
        "forecast_negative": negative,
        "forecast_p_change_min": to_number(latest.get("p_change_min")),
        "forecast_p_change_max": to_number(latest.get("p_change_max")),
        "forecast_summary": latest.get("summary"),
    }


def summarize_cyq_metrics(cyq_df: pd.DataFrame, latest_close: Optional[float] = None) -> dict:
    ordered = sort_desc(cyq_df)
    latest = latest_row(ordered)
    winner_rate_latest = to_float(latest.get("winner_rate"))
    weight_avg = to_float(latest.get("weight_avg"))
    cost_50pct = to_float(latest.get("cost_50pct"))

    winner_change_5d = None
    if not ordered.empty and "winner_rate" in ordered.columns:
        winner_series = pd.to_numeric(ordered["winner_rate"], errors="coerce").dropna().reset_index(drop=True)
        if len(winner_series) > 4:
            winner_change_5d = winner_series.iloc[0] - winner_series.iloc[4]

    close_vs_weight_avg = None
    if latest_close not in (None, 0) and weight_avg not in (None, 0):
        close_vs_weight_avg = (latest_close / weight_avg - 1.0) * 100

    return {
        "winner_rate": to_number(winner_rate_latest),
        "winner_rate_change_5d": to_number(winner_change_5d),
        "weight_avg": to_number(weight_avg),
        "cost_50pct": to_number(cost_50pct),
        "close_vs_weight_avg_pct": to_number(close_vs_weight_avg),
    }


def summarize_holdertrade_signal(
    holdertrade_df: pd.DataFrame,
    end_date: str,
    ann_start_date: str,
    allowed_holder_types: set[str],
    recent_signal_lookback_days: int,
    active_reduction_min_ratio: float = 0.3,
) -> dict:
    if holdertrade_df.empty:
        return {
            "latest_ann_date": None,
            "event_count": 0,
            "announcement_days": 0,
            "holder_count": 0,
            "holder_type_tags": "",
            "holder_preview": "",
            "total_change_vol": 0.0,
            "total_change_ratio": 0.0,
            "avg_increase_price": None,
            "latest_after_ratio": None,
            "event_score": 0.0,
            "latest_change_dir": None,
            "latest_change_date": None,
            "latest_change_holder": None,
            "recent_increase_ratio": 0.0,
            "recent_decrease_ratio": 0.0,
            "recent_core_decrease_ratio": 0.0,
            "recent_decrease_events": 0,
            "recent_signal_balance": 0.0,
            "mixed_signal_flag": False,
            "active_reduction_plan_flag": False,
            "active_reduction_plan_ratio": 0.0,
            "active_reduction_plan_ann_date": None,
            "active_reduction_plan_close_date": None,
            "active_reduction_holder": None,
        }

    work = holdertrade_df.copy()
    for col in ["change_vol", "change_ratio", "avg_price", "after_ratio"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["ann_date"] = work["ann_date"].astype(str)
    work["ann_date_dt"] = pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce")
    work = work.dropna(subset=["ann_date_dt"]).copy()
    if "holder_type" in work.columns:
        work["holder_type"] = work["holder_type"].fillna("").astype(str)
    else:
        work["holder_type"] = ""
    if "holder_name" in work.columns:
        work["holder_name"] = work["holder_name"].fillna("").astype(str)
    else:
        work["holder_name"] = ""
    if "in_de" in work.columns:
        work["in_de"] = work["in_de"].fillna("").astype(str).str.upper()
    else:
        work["in_de"] = ""
    if "close_date" in work.columns:
        work["close_date"] = work["close_date"].fillna("").astype(str)
        work["close_date_dt"] = pd.to_datetime(work["close_date"], format="%Y%m%d", errors="coerce")
    else:
        work["close_date"] = ""
        work["close_date_dt"] = pd.NaT

    all_ordered = work.sort_values(["ann_date_dt", "change_ratio"], ascending=[False, False]).reset_index(drop=True)
    latest = all_ordered.iloc[0]

    in_events = work[work["in_de"] == "IN"].copy()
    if allowed_holder_types:
        in_events = in_events[in_events["holder_type"].isin(sorted(allowed_holder_types))].copy()
    in_events = in_events[in_events["ann_date"] >= ann_start_date].copy()

    latest_in_date = None
    holder_types: list[str] = []
    holder_preview: list[str] = []
    total_change_ratio = 0.0
    total_change_vol = 0.0
    avg_increase_price = None
    event_count = 0
    announcement_days = 0
    holder_count = 0

    if not in_events.empty:
        ordered_in = in_events.sort_values(["ann_date_dt", "change_ratio"], ascending=[False, False]).reset_index(drop=True)
        latest_in_date = ordered_in.iloc[0]["ann_date_dt"]
        holder_types = sorted({value for value in ordered_in["holder_type"].tolist() if value})
        holder_preview = [value for value in ordered_in["holder_name"].tolist() if value][:3]
        total_change_ratio = float(pd.to_numeric(ordered_in["change_ratio"], errors="coerce").fillna(0.0).sum())
        total_change_vol = float(pd.to_numeric(ordered_in["change_vol"], errors="coerce").fillna(0.0).sum())
        avg_increase_price = pd.to_numeric(ordered_in["avg_price"], errors="coerce").mean()
        event_count = int(len(ordered_in))
        announcement_days = int(ordered_in["ann_date"].nunique())
        holder_count = int(ordered_in["holder_name"].nunique())

    end_ts = pd.Timestamp(end_date)
    days_since_latest_ann = int((end_ts - latest_in_date.normalize()).days) if latest_in_date is not None else 999

    score = 0.0
    if total_change_ratio >= 1.5:
        score += 12.0
    elif total_change_ratio >= 0.6:
        score += 9.0
    elif total_change_ratio >= 0.2:
        score += 6.0
    elif total_change_ratio > 0:
        score += 3.0
    if event_count >= 3:
        score += 6.0
    elif event_count == 2:
        score += 4.0
    elif event_count == 1:
        score += 2.0
    if announcement_days >= 3:
        score += 4.0
    elif announcement_days == 2:
        score += 3.0
    elif announcement_days == 1:
        score += 1.0
    if days_since_latest_ann <= 3:
        score += 5.0
    elif days_since_latest_ann <= 10:
        score += 4.0
    elif days_since_latest_ann <= 20:
        score += 2.0
    if "C" in holder_types:
        score += 4.0
    if "G" in holder_types:
        score += 2.0
    event_score = round(clip_score(score, 0.0, 30.0), 2)

    cutoff = end_ts - pd.Timedelta(days=max(1, recent_signal_lookback_days - 1))
    recent = work[work["ann_date_dt"] >= cutoff].copy()
    recent_in = recent[recent["in_de"] == "IN"].copy()
    recent_de = recent[recent["in_de"] == "DE"].copy()
    recent_core_de = recent_de[recent_de["holder_type"].isin(sorted(allowed_holder_types))].copy() if allowed_holder_types else recent_de.copy()
    recent_increase_ratio = float(pd.to_numeric(recent_in["change_ratio"], errors="coerce").fillna(0.0).sum())
    recent_decrease_ratio = float(pd.to_numeric(recent_de["change_ratio"], errors="coerce").fillna(0.0).sum())
    recent_core_decrease_ratio = float(pd.to_numeric(recent_core_de["change_ratio"], errors="coerce").fillna(0.0).sum())
    active_de = recent_core_de[recent_core_de["close_date_dt"].isna() | (recent_core_de["close_date_dt"] >= end_ts)].copy()
    active_reduction_plan_ratio = float(pd.to_numeric(active_de["change_ratio"], errors="coerce").fillna(0.0).sum()) if not active_de.empty else 0.0
    active_latest = active_de.sort_values(["ann_date_dt", "change_ratio"], ascending=[False, False]).iloc[0] if not active_de.empty else None

    return {
        "latest_ann_date": latest_in_date.strftime("%Y%m%d") if latest_in_date is not None else None,
        "event_count": event_count,
        "announcement_days": announcement_days,
        "holder_count": holder_count,
        "holder_type_tags": ",".join(holder_types),
        "holder_preview": " / ".join(holder_preview),
        "total_change_vol": to_number(total_change_vol, 0),
        "total_change_ratio": to_number(total_change_ratio, 4),
        "avg_increase_price": to_number(avg_increase_price),
        "latest_after_ratio": to_number(latest.get("after_ratio")),
        "event_score": event_score,
        "latest_change_dir": latest.get("in_de"),
        "latest_change_date": latest.get("ann_date"),
        "latest_change_holder": latest.get("holder_name"),
        "recent_increase_ratio": to_number(recent_increase_ratio, 4),
        "recent_decrease_ratio": to_number(recent_decrease_ratio, 4),
        "recent_core_decrease_ratio": to_number(recent_core_decrease_ratio, 4),
        "recent_decrease_events": int(len(recent_de)),
        "recent_signal_balance": to_number(recent_increase_ratio - recent_decrease_ratio, 4),
        "mixed_signal_flag": bool(recent_increase_ratio > 0 and recent_decrease_ratio > 0),
        "active_reduction_plan_flag": bool(active_reduction_plan_ratio >= active_reduction_min_ratio),
        "active_reduction_plan_ratio": to_number(active_reduction_plan_ratio, 4),
        "active_reduction_plan_ann_date": active_latest.get("ann_date") if active_latest is not None else None,
        "active_reduction_plan_close_date": active_latest.get("close_date") if active_latest is not None else None,
        "active_reduction_holder": active_latest.get("holder_name") if active_latest is not None else None,
    }


def summarize_unlock_metrics(
    share_float_df: pd.DataFrame,
    end_date: str,
    lookahead_days: int = 30,
    max_near_unlock_ratio: float = 3.0,
    max_unlock_ratio_30d: float = 8.0,
) -> dict:
    if share_float_df.empty or "float_date" not in share_float_df.columns:
        return {
            "nearest_unlock_date": None,
            "days_to_nearest_unlock": None,
            "nearest_unlock_ratio": 0.0,
            "unlock_ratio_30d": 0.0,
            "unlock_risk_veto": False,
        }

    work = share_float_df.copy()
    work["float_date"] = work["float_date"].astype(str)
    work["float_date_dt"] = pd.to_datetime(work["float_date"], format="%Y%m%d", errors="coerce")
    work = work.dropna(subset=["float_date_dt"]).copy()
    if "float_ratio" in work.columns:
        work["float_ratio"] = pd.to_numeric(work["float_ratio"], errors="coerce").fillna(0.0)
    else:
        work["float_ratio"] = 0.0

    screen_end_ts = pd.Timestamp(end_date)
    future_cutoff = screen_end_ts + pd.Timedelta(days=max(1, lookahead_days))
    future = work[(work["float_date_dt"] >= screen_end_ts) & (work["float_date_dt"] <= future_cutoff)].copy()
    if future.empty:
        return {
            "nearest_unlock_date": None,
            "days_to_nearest_unlock": None,
            "nearest_unlock_ratio": 0.0,
            "unlock_ratio_30d": 0.0,
            "unlock_risk_veto": False,
        }

    ordered = future.sort_values(["float_date_dt", "float_ratio"], ascending=[True, False]).reset_index(drop=True)
    nearest = ordered.iloc[0]
    nearest_ratio = pd.to_numeric(
        ordered[ordered["float_date_dt"] == nearest["float_date_dt"]]["float_ratio"],
        errors="coerce",
    ).fillna(0.0).sum()
    unlock_ratio_30d = pd.to_numeric(ordered["float_ratio"], errors="coerce").fillna(0.0).sum()
    days_to_unlock = int((nearest["float_date_dt"].normalize() - screen_end_ts.normalize()).days)
    unlock_risk_veto = bool(
        (days_to_unlock <= 10 and nearest_ratio >= max_near_unlock_ratio)
        or unlock_ratio_30d >= max_unlock_ratio_30d
    )
    return {
        "nearest_unlock_date": nearest["float_date"],
        "days_to_nearest_unlock": days_to_unlock,
        "nearest_unlock_ratio": to_number(nearest_ratio, 4),
        "unlock_ratio_30d": to_number(unlock_ratio_30d, 4),
        "unlock_risk_veto": unlock_risk_veto,
    }


def fetch_single_stock_metrics(
    pro,
    ts_code: str,
    ann_start_date: str,
    end_date: str,
    price_lookback_days: int,
    cyq_lookback_days: int,
    recent_signal_lookback_days: int,
    allowed_holder_types: set[str],
    sleep_sec: float = 0.0,
    cyq_sleep_sec: float = 0.0,
) -> tuple[dict, pd.DataFrame]:
    price_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(420, price_lookback_days * 2))).strftime("%Y%m%d")
    cyq_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(60, cyq_lookback_days * 3))).strftime("%Y%m%d")
    unlock_end_date = (pd.Timestamp(end_date) + pd.Timedelta(days=30)).strftime("%Y%m%d")

    holdertrade_df = sort_desc(
        safe_call(
            f"stk_holdertrade_{ts_code}",
            getattr(pro, "stk_holdertrade", None),
            sleep_sec=sleep_sec,
            ts_code=ts_code,
            start_date=ann_start_date,
            end_date=end_date,
        )
    )
    daily_df = sort_desc(
        safe_call(
            f"daily_{ts_code}",
            getattr(pro, "daily", None),
            sleep_sec=sleep_sec,
            ts_code=ts_code,
            start_date=price_start_date,
            end_date=end_date,
        )
    )
    adj_df = sort_desc(
        safe_call(
            f"adj_factor_{ts_code}",
            getattr(pro, "adj_factor", None),
            sleep_sec=sleep_sec,
            ts_code=ts_code,
            start_date=price_start_date,
            end_date=end_date,
        )
    )
    indicator_df = sort_desc(
        safe_call(
            f"fina_indicator_{ts_code}",
            getattr(pro, "fina_indicator", None),
            sleep_sec=sleep_sec,
            ts_code=ts_code,
        )
    )
    forecast_df = sort_desc(
        safe_call(
            f"forecast_{ts_code}",
            getattr(pro, "forecast", None),
            sleep_sec=sleep_sec,
            ts_code=ts_code,
        )
    )
    cyq_df = sort_desc(
        safe_call(
            f"cyq_perf_{ts_code}",
            getattr(pro, "cyq_perf", None),
            sleep_sec=cyq_sleep_sec,
            ts_code=ts_code,
            start_date=cyq_start_date,
            end_date=end_date,
        )
    )
    share_float_df = sort_desc(
        safe_call(
            f"share_float_{ts_code}",
            getattr(pro, "share_float", None),
            sleep_sec=sleep_sec,
            ts_code=ts_code,
            start_date=end_date,
            end_date=unlock_end_date,
        )
    )

    price_metrics = summarize_price_metrics(daily_df, adj_df, window=price_lookback_days)
    return (
        {
            "ts_code": ts_code,
            **summarize_holdertrade_signal(
                holdertrade_df,
                end_date=end_date,
                ann_start_date=ann_start_date,
                allowed_holder_types=allowed_holder_types,
                recent_signal_lookback_days=recent_signal_lookback_days,
                active_reduction_min_ratio=0.3,
            ),
            **summarize_unlock_metrics(share_float_df, end_date=end_date),
            **price_metrics,
            **summarize_indicator_metrics(indicator_df),
            **summarize_forecast_metrics(forecast_df),
            **summarize_cyq_metrics(cyq_df, latest_close=to_float(price_metrics.get("latest_close_qfq_calc"))),
            "cyq_checked": not cyq_df.empty,
        },
        holdertrade_df,
    )


def build_earnings_score(row: dict) -> float:
    score = 0.0
    if not to_bool(row.get("forecast_negative")):
        score += 8.0
    dt_netprofit_yoy = to_float(row.get("dt_netprofit_yoy"))
    if dt_netprofit_yoy is not None:
        if dt_netprofit_yoy > 20:
            score += 10.0
        elif dt_netprofit_yoy > 0:
            score += 8.0
        elif dt_netprofit_yoy > -10:
            score += 3.0
    ocf_yoy = to_float(row.get("ocf_yoy"))
    if ocf_yoy is not None:
        if ocf_yoy > 15:
            score += 6.0
        elif ocf_yoy > 0:
            score += 4.0
    roe = to_float(row.get("roe"))
    if roe is not None:
        if roe >= 12:
            score += 6.0
        elif roe >= 8:
            score += 5.0
        elif roe >= 5:
            score += 3.0
    debt_to_assets = to_float(row.get("debt_to_assets"))
    if debt_to_assets is not None:
        if debt_to_assets <= 50:
            score += 4.0
        elif debt_to_assets <= 65:
            score += 2.0
        else:
            score -= 4.0
    return round(clip_score(score, 0.0, 30.0), 2)


def build_value_score(row: dict) -> float:
    score = 0.0
    position = to_float(row.get("price_position_250"))
    pb_pct = to_float(row.get("industry_pb_pct_rank"))
    return_20d = to_float(row.get("return_20d"))
    if position is not None:
        score += max(0.0, (0.62 - min(position, 0.62)) / 0.62) * 12.0
    if pb_pct is not None:
        score += max(0.0, (0.80 - min(pb_pct, 0.80)) / 0.80) * 8.0
    if return_20d is not None and return_20d < -12:
        score -= 2.0
    return round(clip_score(score, 0.0, 20.0), 2)


def build_reversal_score(row: dict, min_volume_ratio: float = 1.2) -> float:
    score = 0.0
    close_qfq = to_float(row.get("close_qfq"))
    ma_20 = to_float(row.get("ma_qfq_20"))
    ma_5 = to_float(row.get("ma_qfq_5"))
    ma_10 = to_float(row.get("ma_qfq_10"))
    macd_dif = to_float(row.get("macd_dif_qfq"))
    macd_dea = to_float(row.get("macd_dea_qfq"))
    volume_ratio = to_float(row.get("volume_ratio"))
    net_amount_3d = to_float(row.get("main_net_amount_3d"))
    net_amount_5d = to_float(row.get("main_net_amount_5d"))
    winner_change_5d = to_float(row.get("winner_rate_change_5d"))
    return_20d = to_float(row.get("return_20d"))
    if close_qfq is not None and ma_20 is not None and close_qfq > ma_20:
        score += 5.0
    if ma_5 is not None and ma_10 is not None and ma_5 > ma_10:
        score += 4.0
    if macd_dif is not None and macd_dea is not None and macd_dif > macd_dea:
        score += 4.0
    if volume_ratio is not None and volume_ratio >= min_volume_ratio:
        score += 3.0
    if net_amount_3d is not None and net_amount_3d > 0:
        score += 2.0
    if net_amount_5d is not None and net_amount_5d > 0:
        score += 1.0
    if winner_change_5d is not None and winner_change_5d > 0:
        score += 1.0
    if return_20d is not None and -5 <= return_20d <= 20:
        score += 1.0
    return round(clip_score(score, 0.0, 20.0), 2)


def build_chip_score(row: dict) -> float:
    score = 0.0
    winner_rate = to_float(row.get("winner_rate"))
    winner_change_5d = to_float(row.get("winner_rate_change_5d"))
    close_vs_weight_avg_pct = to_float(row.get("close_vs_weight_avg_pct"))
    if winner_rate is not None:
        if winner_rate >= 60:
            score += 8.0
        elif winner_rate >= 45:
            score += 6.0
        elif winner_rate >= 30:
            score += 4.0
        elif winner_rate >= 15:
            score += 2.0
    if winner_change_5d is not None:
        if winner_change_5d >= 8:
            score += 8.0
        elif winner_change_5d > 3:
            score += 6.0
        elif winner_change_5d > 0:
            score += 4.0
    if close_vs_weight_avg_pct is not None:
        if -3 <= close_vs_weight_avg_pct <= 5:
            score += 4.0
        elif 5 < close_vs_weight_avg_pct <= 10:
            score += 2.0
        elif close_vs_weight_avg_pct < -3:
            score -= 1.0
    return round(clip_score(score, 0.0, 20.0), 2)


def build_fund_flow_score(row: dict) -> float:
    score = 0.0
    flow_3d_rank = to_float(row.get("main_net_amount_3d_rank_pct"))
    flow_5d_rank = to_float(row.get("main_net_amount_5d_rank_pct"))
    net_amount_3d = to_float(row.get("main_net_amount_3d"))
    net_amount_5d = to_float(row.get("main_net_amount_5d"))
    volume_ratio = to_float(row.get("volume_ratio"))
    positive_days_3d = to_float(row.get("main_net_positive_days_3d"))
    positive_days_5d = to_float(row.get("main_net_positive_days_5d"))
    consecutive_days = to_float(row.get("main_net_consecutive_days"))
    if flow_3d_rank is not None:
        score += flow_3d_rank * 10.0
    if flow_5d_rank is not None:
        score += flow_5d_rank * 4.0
    if net_amount_3d is not None and net_amount_3d > 0:
        score += 3.0
    if net_amount_5d is not None and net_amount_5d > 0:
        score += 1.5
    if positive_days_3d is not None:
        score += min(positive_days_3d, 3.0) * 1.2
    if positive_days_5d is not None:
        score += min(positive_days_5d, 5.0) * 0.4
    if consecutive_days is not None:
        score += min(consecutive_days, 5.0) * 0.8
    if volume_ratio is not None:
        if volume_ratio >= 2.0:
            score += 1.5
        elif volume_ratio >= 1.2:
            score += 1.0
    return round(clip_score(score, 0.0, 20.0), 2)


def build_event_bonus_score(row: dict) -> float:
    event_score = to_float(row.get("event_score"))
    if event_score is None:
        return 0.0
    latest_change_dir = str(row.get("latest_change_dir") or "").upper()
    bonus = event_score * 0.25
    if latest_change_dir == "IN":
        bonus += 1.0
    return round(clip_score(bonus, 0.0, 8.0), 2)


def build_overheat_penalty_score(row: dict) -> float:
    score = 0.0
    price_position = to_float(row.get("price_position_250"))
    return_20d = to_float(row.get("return_20d"))
    return_60d = to_float(row.get("return_60d"))
    close_vs_weight_avg_pct = to_float(row.get("close_vs_weight_avg_pct"))
    volume_ratio = to_float(row.get("volume_ratio"))
    if price_position is not None:
        if price_position >= 0.98:
            score += 3.0
        elif price_position >= 0.90:
            score += 1.5
    if return_20d is not None:
        if return_20d >= 45:
            score += 4.0
        elif return_20d >= 30:
            score += 2.0
    if return_60d is not None and return_60d >= 80:
        score += 1.5
    if close_vs_weight_avg_pct is not None:
        if close_vs_weight_avg_pct >= 12:
            score += 2.0
        elif close_vs_weight_avg_pct >= 8:
            score += 1.0
    if volume_ratio is not None and volume_ratio >= 2.5 and (return_20d or 0) >= 25:
        score += 1.0
    return round(clip_score(score, 0.0, 10.0), 2)


def build_risk_penalty_score(row: dict) -> float:
    score = 0.0
    latest_change_dir = str(row.get("latest_change_dir") or "").upper()
    recent_decrease_ratio = to_float(row.get("recent_decrease_ratio"))
    recent_core_decrease_ratio = to_float(row.get("recent_core_decrease_ratio"))
    recent_signal_balance = to_float(row.get("recent_signal_balance"))
    active_reduction_plan_ratio = to_float(row.get("active_reduction_plan_ratio"))
    nearest_unlock_ratio = to_float(row.get("nearest_unlock_ratio"))
    unlock_ratio_30d = to_float(row.get("unlock_ratio_30d"))
    if latest_change_dir == "DE":
        score += 2.5
    if recent_decrease_ratio is not None:
        if recent_decrease_ratio >= 1.0:
            score += 4.0
        elif recent_decrease_ratio >= 0.3:
            score += 2.5
        elif recent_decrease_ratio > 0:
            score += 1.0
    if recent_core_decrease_ratio is not None:
        if recent_core_decrease_ratio >= 0.8:
            score += 4.0
        elif recent_core_decrease_ratio >= 0.2:
            score += 2.0
    if to_bool(row.get("mixed_signal_flag")):
        score += 1.5
    if recent_signal_balance is not None and recent_signal_balance < 0:
        score += 2.0
    if to_bool(row.get("active_reduction_plan_flag")):
        score += 6.0
    elif active_reduction_plan_ratio is not None and active_reduction_plan_ratio > 0:
        score += 2.0
    if to_bool(row.get("unlock_risk_veto")):
        score += 5.0
    else:
        if nearest_unlock_ratio is not None and nearest_unlock_ratio >= 2.0:
            score += 1.5
        if unlock_ratio_30d is not None and unlock_ratio_30d >= 5.0:
            score += 2.0
    return round(clip_score(score, 0.0, 20.0), 2)


def build_stable_score(row: dict, market_regime: str = "neutral") -> float:
    regime_bias = 2.0 if market_regime == "defensive" else (1.0 if market_regime == "neutral" else 0.0)
    score = (
        (to_float(row.get("earnings_score")) or 0.0) * 0.90
        + (to_float(row.get("reversal_score")) or 0.0) * 0.75
        + (to_float(row.get("fund_flow_score")) or 0.0) * 0.95
        + (to_float(row.get("value_score")) or 0.0) * 0.35
        + (to_float(row.get("event_bonus_score")) or 0.0) * 0.35
        + (to_float(row.get("chip_score")) or 0.0) * 0.50
        + regime_bias
        - (to_float(row.get("risk_penalty_score")) or 0.0)
        - (to_float(row.get("overheat_penalty_score")) or 0.0)
    )
    return round(clip_score(score, 0.0, 100.0), 2)


def build_aggressive_score(row: dict, market_regime: str = "neutral") -> float:
    regime_bias = 4.0 if market_regime == "risk_on" else (0.0 if market_regime == "neutral" else -4.0)
    return_20d = to_float(row.get("return_20d"))
    price_position = to_float(row.get("price_position_250"))
    volume_ratio = to_float(row.get("volume_ratio"))
    breakout_bonus = 0.0
    if return_20d is not None:
        if -5 <= return_20d <= 18:
            breakout_bonus += 4.0
        elif 18 < return_20d <= 30:
            breakout_bonus += 2.0
        elif return_20d > 30:
            breakout_bonus -= 3.0
    if price_position is not None:
        if 0.15 <= price_position <= 0.65:
            breakout_bonus += 4.0
        elif price_position > 0.85:
            breakout_bonus -= 2.0
    if volume_ratio is not None and volume_ratio >= 1.8:
        breakout_bonus += 2.0

    score = (
        (to_float(row.get("earnings_score")) or 0.0) * 0.45
        + (to_float(row.get("reversal_score")) or 0.0) * 0.90
        + (to_float(row.get("fund_flow_score")) or 0.0) * 1.05
        + (to_float(row.get("value_score")) or 0.0) * 0.15
        + (to_float(row.get("event_bonus_score")) or 0.0) * 0.20
        + (to_float(row.get("chip_score")) or 0.0) * 0.65
        + breakout_bonus
        + regime_bias
        - (to_float(row.get("risk_penalty_score")) or 0.0) * 0.80
        - (to_float(row.get("overheat_penalty_score")) or 0.0) * 0.45
    )
    return round(clip_score(score, 0.0, 100.0), 2)


def score_watchlist(df: pd.DataFrame, market_regime: str) -> pd.DataFrame:
    work = df.copy()

    def numeric_series(column: str) -> pd.Series:
        if column in work.columns:
            return pd.to_numeric(work[column], errors="coerce")
        return pd.Series(np.nan, index=work.index, dtype=float)

    work["main_net_amount_3d_rank_pct"] = rank_pct(numeric_series("main_net_amount_3d"))
    work["main_net_amount_5d_rank_pct"] = rank_pct(numeric_series("main_net_amount_5d"))
    work["earnings_score"] = work.apply(lambda row: build_earnings_score(row.to_dict()), axis=1)
    work["value_score"] = work.apply(lambda row: build_value_score(row.to_dict()), axis=1)
    work["reversal_score"] = work.apply(lambda row: build_reversal_score(row.to_dict(), 1.2), axis=1)
    work["event_bonus_score"] = work.apply(lambda row: build_event_bonus_score(row.to_dict()), axis=1)
    work["chip_score"] = work.apply(lambda row: build_chip_score(row.to_dict()), axis=1)
    work["fund_flow_score"] = work.apply(lambda row: build_fund_flow_score(row.to_dict()), axis=1)
    work["overheat_penalty_score"] = work.apply(lambda row: build_overheat_penalty_score(row.to_dict()), axis=1)
    work["risk_penalty_score"] = work.apply(lambda row: build_risk_penalty_score(row.to_dict()), axis=1)
    work["final_score"] = (
        pd.to_numeric(work["earnings_score"], errors="coerce").fillna(0.0) * 0.85
        + pd.to_numeric(work["reversal_score"], errors="coerce").fillna(0.0) * 0.95
        + pd.to_numeric(work["fund_flow_score"], errors="coerce").fillna(0.0) * 1.10
        + pd.to_numeric(work["chip_score"], errors="coerce").fillna(0.0) * 0.40
        + pd.to_numeric(work["value_score"], errors="coerce").fillna(0.0) * 0.20
        + pd.to_numeric(work["event_bonus_score"], errors="coerce").fillna(0.0) * 0.35
        - pd.to_numeric(work["risk_penalty_score"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["overheat_penalty_score"], errors="coerce").fillna(0.0) * 0.60
    ).clip(lower=0.0, upper=100.0).round(2)
    work["stable_score"] = work.apply(lambda row: build_stable_score(row.to_dict(), market_regime), axis=1)
    work["aggressive_score"] = work.apply(lambda row: build_aggressive_score(row.to_dict(), market_regime), axis=1)
    if market_regime == "risk_on":
        work["priority_score"] = (
            pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0) * 0.60
            + pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0) * 0.40
        ).round(2)
    elif market_regime == "defensive":
        work["priority_score"] = (
            pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0) * 0.70
            + pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0) * 0.30
        ).round(2)
    else:
        work["priority_score"] = (
            pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0) * 0.55
            + pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0) * 0.45
        ).round(2)

    work["market_regime"] = market_regime
    work["preferred_pool"] = np.where(
        pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0)
        > pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0),
        "aggressive",
        "stable",
    )
    work["stable_candidate"] = (
        (~work["forecast_negative"].fillna(False))
        & (~work["active_reduction_plan_flag"].fillna(False))
        & (~work["unlock_risk_veto"].fillna(False))
        & (pd.to_numeric(work["overheat_penalty_score"], errors="coerce").fillna(0.0) < 6.0)
        & (pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0) >= 55.0)
    )
    work["aggressive_candidate"] = (
        (~work["forecast_negative"].fillna(False))
        & (~work["active_reduction_plan_flag"].fillna(False))
        & (~work["unlock_risk_veto"].fillna(False))
        & (pd.to_numeric(work["fund_flow_score"], errors="coerce").fillna(0.0) >= 12.0)
        & (pd.to_numeric(work["overheat_penalty_score"], errors="coerce").fillna(0.0) < 8.0)
        & (pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0) >= 50.0)
    )

    return work.sort_values(
        ["priority_score", "stable_candidate", "aggressive_candidate", "final_score", "fund_flow_score", "reversal_score", "event_bonus_score"],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)


def display_columns() -> list[str]:
    return [
        "ts_code",
        "name",
        "industry",
        "market_regime",
        "preferred_pool",
        "priority_score",
        "stable_score",
        "aggressive_score",
        "final_score",
        "event_bonus_score",
        "earnings_score",
        "value_score",
        "reversal_score",
        "fund_flow_score",
        "overheat_penalty_score",
        "risk_penalty_score",
        "chip_score",
        "main_net_amount_3d",
        "main_net_amount_5d",
        "main_net_positive_days_3d",
        "main_net_positive_days_5d",
        "main_net_consecutive_days",
        "winner_rate",
        "winner_rate_change_5d",
        "recent_decrease_ratio",
        "active_reduction_plan_flag",
        "active_reduction_plan_ratio",
        "unlock_risk_veto",
        "nearest_unlock_date",
        "nearest_unlock_ratio",
        "unlock_ratio_30d",
        "latest_change_dir",
        "forecast_type",
    ]


def main() -> None:
    args = parse_args()
    raw_codes = split_codes(args.ts_codes)
    if not raw_codes:
        raise SystemExit("Please provide at least one stock code.")

    ts_codes = []
    for code in raw_codes:
        normalized = normalize_ts_code(code)
        if normalized and normalized not in ts_codes:
            ts_codes.append(normalized)

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    use_custom_http_endpoint = bool(custom_http_url)
    ensure_token(token)

    pro = ts.pro_api(token)
    pro = configure_tushare_client(pro, token, use_custom_http_endpoint, custom_http_url)

    now_ts = pd.Timestamp.now()
    today_str = now_ts.strftime("%Y%m%d")
    requested_end_date = args.end_date.strip() or today_str
    screen_end_date = choose_screen_end_date(now_ts, requested_end_date, today_str, args.cutoff_hour)
    ann_start_date = (pd.Timestamp(screen_end_date) - pd.Timedelta(days=args.ann_lookback_days)).strftime("%Y%m%d")

    trade_dates = get_recent_open_trade_dates(pro, screen_end_date, count=max(args.moneyflow_lookback_days, 10))
    latest_trade_date, market_moneyflow_dates, daily_basic_latest, tech_latest, moneyflow_summary = fetch_latest_complete_market_inputs(
        pro,
        trade_dates,
        moneyflow_lookback_days=args.moneyflow_lookback_days,
        sleep_sec=0.12,
    )

    stock_basic_all = safe_call(
        "stock_basic_all",
        getattr(pro, "stock_basic", None),
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    if stock_basic_all.empty:
        raise SystemExit("stock_basic returned no data.")
    stock_basic_all = stock_basic_all.fillna("")

    selected_basics = stock_basic_all[stock_basic_all["ts_code"].isin(ts_codes)].copy()
    missing_codes = [code for code in ts_codes if code not in set(selected_basics["ts_code"].tolist())]
    if selected_basics.empty:
        raise SystemExit("None of the provided stock codes were found in stock_basic.")

    market_snapshot_all = build_market_snapshot(stock_basic_all, daily_basic_latest, tech_latest, moneyflow_summary)
    market_regime_snapshot = build_market_regime_snapshot(market_snapshot_all)
    market_regime = market_regime_snapshot["market_regime"]

    selected_market_snapshot = market_snapshot_all[market_snapshot_all["ts_code"].isin(ts_codes)].copy()
    candidate_base = selected_basics.merge(selected_market_snapshot, on="ts_code", how="left", suffixes=("", "_mkt"))

    allowed_holder_type_set = {"C", "G"}
    deep_rows: list[dict] = []
    holdertrade_frames: list[pd.DataFrame] = []
    total_targets = len(ts_codes)
    for idx, ts_code in enumerate(ts_codes, start=1):
        print(f"[watchlist {idx}/{total_targets}] {ts_code}")
        metrics, holdertrade_df = fetch_single_stock_metrics(
            pro,
            ts_code=ts_code,
            ann_start_date=ann_start_date,
            end_date=screen_end_date,
            price_lookback_days=250,
            cyq_lookback_days=args.cyq_lookback_days,
            recent_signal_lookback_days=20,
            allowed_holder_types=allowed_holder_type_set,
            sleep_sec=0.12,
            cyq_sleep_sec=0.12,
        )
        deep_rows.append(metrics)
        if not holdertrade_df.empty:
            holdertrade_frames.append(holdertrade_df)

    deep_metrics = pd.DataFrame(deep_rows)
    holdertrade_raw = pd.concat(holdertrade_frames, ignore_index=True) if holdertrade_frames else pd.DataFrame()
    watchlist_scores = candidate_base.merge(deep_metrics, on="ts_code", how="left")
    for col in [
        "close",
        "volume_ratio",
        "pb",
        "pe_ttm",
        "total_mv",
        "close_qfq",
        "ma_qfq_5",
        "ma_qfq_10",
        "ma_qfq_20",
        "ma_qfq_250",
        "macd_dif_qfq",
        "macd_dea_qfq",
        "main_net_amount_3d",
        "main_net_amount_5d",
        "main_net_positive_days_3d",
        "main_net_positive_days_5d",
        "main_net_consecutive_days",
        "industry_pb_pct_rank",
        "recent_increase_ratio",
        "recent_decrease_ratio",
        "recent_core_decrease_ratio",
        "recent_signal_balance",
        "active_reduction_plan_ratio",
        "nearest_unlock_ratio",
        "unlock_ratio_30d",
        "winner_rate",
        "winner_rate_change_5d",
    ]:
        if col in watchlist_scores.columns:
            watchlist_scores[col] = pd.to_numeric(watchlist_scores[col], errors="coerce")
    for col in [
        "event_score",
        "event_count",
        "total_change_ratio",
        "main_net_amount_3d",
        "main_net_amount_5d",
        "main_net_positive_days_3d",
        "main_net_positive_days_5d",
        "main_net_consecutive_days",
        "recent_increase_ratio",
        "recent_decrease_ratio",
        "recent_core_decrease_ratio",
        "recent_signal_balance",
        "active_reduction_plan_ratio",
        "nearest_unlock_ratio",
        "unlock_ratio_30d",
    ]:
        if col in watchlist_scores.columns:
            watchlist_scores[col] = watchlist_scores[col].fillna(0.0)
    for col in ["active_reduction_plan_flag", "unlock_risk_veto", "mixed_signal_flag"]:
        if col in watchlist_scores.columns:
            watchlist_scores[col] = watchlist_scores[col].fillna(False)

    watchlist_scores = score_watchlist(watchlist_scores, market_regime=market_regime)
    today_direction = "偏进攻" if market_regime == "risk_on" else "偏稳健"
    if today_direction == "偏进攻":
        preferred = watchlist_scores.sort_values(["aggressive_candidate", "aggressive_score", "priority_score"], ascending=[False, False, False])
    else:
        preferred = watchlist_scores.sort_values(["stable_candidate", "stable_score", "priority_score"], ascending=[False, False, False])
    best_pick_candidate = preferred.head(1).copy()
    if best_pick_candidate.empty:
        best_pick_candidate = watchlist_scores.head(1).copy()

    repo_root = Path(__file__).resolve().parent.parent
    output_root = repo_root / "output" / "jupyter-notebook" / "tushare_watchlist_exports"
    output_root.mkdir(parents=True, exist_ok=True)
    run_tag = f"watchlist_pick_{screen_end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = output_root / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "requested_ts_codes": ts_codes,
        "missing_codes": missing_codes,
        "requested_end_date": requested_end_date,
        "screen_end_date": screen_end_date,
        "latest_trade_date": latest_trade_date,
        "market_moneyflow_dates": market_moneyflow_dates,
        "market_regime": market_regime,
        "market_regime_score": market_regime_snapshot.get("market_regime_score"),
        "market_trend_breadth": market_regime_snapshot.get("market_trend_breadth"),
        "market_flow_breadth": market_regime_snapshot.get("market_flow_breadth"),
        "market_hot_ratio": market_regime_snapshot.get("market_hot_ratio"),
        "today_direction": today_direction,
        "best_pick_ts_code": best_pick_candidate.iloc[0]["ts_code"] if not best_pick_candidate.empty else None,
        "export_dir": str(export_dir.resolve()),
    }

    holdertrade_raw.to_csv(export_dir / "holdertrade_raw.csv", index=False)
    watchlist_scores.to_csv(export_dir / "watchlist_scores.csv", index=False)
    best_pick_candidate.to_csv(export_dir / "best_pick_candidate.csv", index=False)
    with (export_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    cols = [c for c in display_columns() if c in watchlist_scores.columns]
    best_cols = [c for c in display_columns() if c in best_pick_candidate.columns]

    print(f"today_direction={today_direction}")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print("===== watchlist ranking =====")
    print(ensure_columns(watchlist_scores, cols)[cols].head(args.show_top).to_string(index=False))
    print("===== best pick =====")
    print(ensure_columns(best_pick_candidate, best_cols)[best_cols].to_string(index=False))


if __name__ == "__main__":
    main()
