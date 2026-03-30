from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MARKET_REGIME_CONFIG: dict[str, Any] = {
    "include_star": False,
    "min_universe_count": 1200,
    "breadth_short_ma": 20,
    "breadth_long_ma": 60,
    "regime_up_threshold": 0.62,
    "regime_down_threshold": 0.38,
    "breadth_above_ma20_weight": 0.35,
    "breadth_above_ma60_weight": 0.25,
    "breadth_ma20_up_weight": 0.20,
    "ew_trend_weight": 0.20,
}


def merge_market_regime_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_MARKET_REGIME_CONFIG)
    if config:
        merged.update({key: value for key, value in config.items() if value is not None})
    return merged


def _board_from_meta(meta: dict[str, Any], include_star: bool = False) -> str:
    market = str(meta.get("market") or "").strip()
    ts_code = str(meta.get("ts_code") or "").upper()
    if market == "创业板" or ts_code.startswith(("300", "301")):
        return "gem"
    if market == "主板":
        return "main"
    if market == "科创板" or ts_code.startswith(("688", "689")):
        return "star" if include_star else ""
    if market == "北交所" or ts_code.startswith(("8", "4")):
        return ""
    return "main" if ts_code.endswith((".SH", ".SZ")) and not ts_code.startswith(("688", "689", "8", "4")) else ""


def _is_allowed_stock(meta: dict[str, Any], include_star: bool = False) -> bool:
    name = str(meta.get("name") or "").upper()
    if "ST" in name or "退" in name:
        return False
    return _board_from_meta(meta, include_star=include_star) in {"main", "gem", "star"}


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def build_market_regime_snapshot(
    market_daily_history: pd.DataFrame,
    stock_basic_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if market_daily_history is None or market_daily_history.empty or stock_basic_df is None or stock_basic_df.empty:
        return pd.DataFrame()

    cfg = merge_market_regime_config(config)
    include_star = bool(cfg["include_star"])
    basic = stock_basic_df.copy().fillna("")
    basic = basic[basic.apply(lambda row: _is_allowed_stock(row.to_dict(), include_star=include_star), axis=1)].copy()
    if basic.empty:
        return pd.DataFrame()

    allowed_codes = set(basic["ts_code"].astype(str))
    history = market_daily_history.copy()
    history["ts_code"] = history["ts_code"].astype(str)
    history = history[history["ts_code"].isin(allowed_codes)].copy()
    if history.empty:
        return pd.DataFrame()

    history["trade_date"] = history["trade_date"].astype(str)
    for column in ["close", "pre_close"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(subset=["close", "pre_close"]).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    if history.empty:
        return pd.DataFrame()

    close_series = history.groupby("ts_code")["close"]
    history["ma20"] = close_series.transform(lambda s: s.rolling(int(cfg["breadth_short_ma"]), min_periods=int(cfg["breadth_short_ma"])).mean())
    history["ma60"] = close_series.transform(lambda s: s.rolling(int(cfg["breadth_long_ma"]), min_periods=int(cfg["breadth_long_ma"])).mean())
    history["ma20_prev"] = history.groupby("ts_code")["ma20"].shift(1)
    history["ret1"] = np.where(history["pre_close"] > 0, history["close"] / history["pre_close"] - 1.0, np.nan)
    history["above_ma20"] = history["close"] > history["ma20"]
    history["above_ma60"] = history["close"] > history["ma60"]
    history["ma20_up"] = history["ma20"] > history["ma20_prev"]

    breadth = (
        history.groupby("trade_date", as_index=False)
        .agg(
            universe_count=("ts_code", "nunique"),
            breadth_above_ma20_ratio=("above_ma20", "mean"),
            breadth_above_ma60_ratio=("above_ma60", "mean"),
            breadth_ma20_up_ratio=("ma20_up", "mean"),
            equal_weight_ret=("ret1", "mean"),
            median_ret=("ret1", "median"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    breadth = breadth[breadth["universe_count"] >= int(cfg["min_universe_count"])].copy()
    if breadth.empty:
        return pd.DataFrame()

    breadth["equal_weight_ret"] = breadth["equal_weight_ret"].fillna(0.0)
    breadth["equal_weight_index"] = (1.0 + breadth["equal_weight_ret"]).cumprod()
    breadth["ew_ma20"] = breadth["equal_weight_index"].rolling(int(cfg["breadth_short_ma"]), min_periods=int(cfg["breadth_short_ma"])).mean()
    breadth["ew_ma60"] = breadth["equal_weight_index"].rolling(int(cfg["breadth_long_ma"]), min_periods=int(cfg["breadth_long_ma"])).mean()
    breadth["ew_trend_score"] = 0.5
    ew_up_mask = (
        (breadth["equal_weight_index"] > breadth["ew_ma60"])
        & (breadth["ew_ma20"] > breadth["ew_ma60"])
        & (breadth["ew_ma20"] > breadth["ew_ma20"].shift(1))
    )
    ew_down_mask = (
        (breadth["equal_weight_index"] < breadth["ew_ma60"])
        & (breadth["ew_ma20"] < breadth["ew_ma60"])
        & (breadth["ew_ma20"] < breadth["ew_ma20"].shift(1))
    )
    breadth.loc[ew_up_mask.fillna(False), "ew_trend_score"] = 1.0
    breadth.loc[ew_down_mask.fillna(False), "ew_trend_score"] = 0.0

    breadth["market_regime_score"] = (
        _clip01(breadth["breadth_above_ma20_ratio"]) * float(cfg["breadth_above_ma20_weight"])
        + _clip01(breadth["breadth_above_ma60_ratio"]) * float(cfg["breadth_above_ma60_weight"])
        + _clip01(breadth["breadth_ma20_up_ratio"]) * float(cfg["breadth_ma20_up_weight"])
        + _clip01(breadth["ew_trend_score"]) * float(cfg["ew_trend_weight"])
    )

    up_threshold = float(cfg["regime_up_threshold"])
    down_threshold = float(cfg["regime_down_threshold"])
    breadth["market_regime"] = np.where(
        breadth["market_regime_score"] >= up_threshold,
        "上涨趋势",
        np.where(breadth["market_regime_score"] <= down_threshold, "下跌趋势", "震荡趋势"),
    )
    breadth["market_regime_reason"] = np.select(
        [
            breadth["market_regime"] == "上涨趋势",
            breadth["market_regime"] == "下跌趋势",
        ],
        [
            "广度较强，等权趋势向上",
            "广度较弱，等权趋势向下",
        ],
        default="广度与等权趋势分化，市场以震荡为主",
    )
    return breadth
