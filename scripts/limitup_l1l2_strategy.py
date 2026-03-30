from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


DEFAULT_LIMITUP_L1L2_CONFIG: dict[str, Any] = {
    "mode_limit": "AUTO",
    "restart_every_limit_up": True,
    "tol_tick": 2,
    "need_seal": True,
    "tick_size": 0.01,
    "zz_left": 2,
    "zz_right": 2,
    "zz_dev_pct": 3.0,
    "l1_break_pct": 0.2,
    "l1_max_bars": 35,
    "min_bars_between_lows": 2,
    "max_bars_between_lows": 40,
    "higher_low_min_pct": 0.3,
    "higher_low_max_pct": 12.0,
    "impulse_pct": 6.0,
    "impulse_max_bars": 25,
    "pullback_min_pct": 2.0,
    "use_shrink_after_l2": False,
    "shrink_bars_need": 2,
    "vol_ma_len": 5,
    "shrink_mult": 0.9,
    "ema_len": 20,
    "need_ema_cross": False,
    "need_bull_candle": False,
    "use_vol_filter": True,
    "vol_mult_up": 1.05,
    "use_trend_filter": True,
    "trend_ema_len": 60,
    "hold_l2_break_pct": 0.3,
    "rebound_max_bars": 25,
    "atr_len": 14,
    "recent_buy_window": 0,
    "candidate_score_threshold": 55.0,
}


def clip_score(value: Any, low: float, high: float) -> float:
    numeric = to_float(value)
    if numeric is None:
        return low
    return float(min(max(numeric, low), high))


def to_float(value: Any) -> float | None:
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


def to_number(value: Any, digits: int = 2) -> float | int | None:
    numeric = to_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def to_bool(value: Any) -> bool:
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


def merge_limitup_l1l2_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_LIMITUP_L1L2_CONFIG)
    if config:
        merged.update({key: value for key, value in config.items() if value is not None})
    return merged


def infer_limit_pct(ts_code: str = "", name: str = "", mode_limit: str = "AUTO") -> float:
    code = str(ts_code or "").upper()
    stock_name = str(name or "").upper()
    is_st = "ST" in stock_name or "ST" in code
    is_20 = code.startswith(("300", "301", "688", "689"))
    mode = str(mode_limit or "AUTO").upper()
    if mode == "20%":
        return 0.20
    if mode == "10%":
        return 0.10
    if mode == "5%":
        return 0.05
    if is_st:
        return 0.05
    if is_20:
        return 0.20
    return 0.10


def _empty_result(ts_code: str = "") -> dict[str, Any]:
    return {
        "ts_code": ts_code,
        "limitup_l1l2_stage": "",
        "limitup_l1l2_signal": False,
        "limitup_l1l2_buy_signal": False,
        "limitup_l1l2_buy_recent": False,
        "limitup_l1l2_ready": False,
        "limitup_l1l2_score": 0.0,
        "limitup_l1l2_reason": "",
        "limitup_l1l2_limit_date": None,
        "limitup_l1l2_l1_date": None,
        "limitup_l1l2_l2_date": None,
        "limitup_l1l2_buy_date": None,
        "limitup_l1l2_limit_price": None,
        "limitup_l1l2_l1_price": None,
        "limitup_l1l2_l2_price": None,
        "limitup_l1l2_impulse_high": None,
        "limitup_l1l2_current_price": None,
        "limitup_l1l2_bars_since_limit": None,
        "limitup_l1l2_bars_since_l2": None,
        "limitup_l1l2_bars_since_buy": None,
        "limitup_l1l2_bars_lu_to_l1": None,
        "limitup_l1l2_bars_l1_to_l2": None,
        "limitup_l1l2_impulse_pct": None,
        "limitup_l1l2_pullback_pct": None,
        "limitup_l1l2_l2_above_l1_pct": None,
        "limitup_l1l2_confirm_vol_ratio": None,
        "limitup_l1l2_close_vs_l2_pct": None,
        "limitup_l1l2_hold_buffer_pct": None,
        "limitup_l1l2_trend_ok": False,
        "limitup_l1l2_volume_ok": False,
        "limitup_l1l2_limit_sealed": False,
    }


def _confirmed_pivot_events(values: np.ndarray, left: int, right: int, pivot_kind: str) -> list[dict[str, Any] | None]:
    events: list[dict[str, Any] | None] = [None] * len(values)
    window_size = left + right + 1
    if len(values) < window_size:
        return events
    windows = sliding_window_view(np.asarray(values, dtype=float), window_size)
    if pivot_kind == "low":
        filled = np.where(np.isnan(windows), np.inf, windows)
        extreme = filled.min(axis=1)
        extreme[np.isinf(extreme)] = np.nan
    else:
        filled = np.where(np.isnan(windows), -np.inf, windows)
        extreme = filled.max(axis=1)
        extreme[np.isneginf(extreme)] = np.nan
    centers = windows[:, left]
    valid = ~np.isnan(centers) & ~np.isnan(extreme)
    is_extreme = np.abs(centers - extreme) <= 1e-12
    if right > 0:
        right_windows = windows[:, left + 1 :]
        right_match = np.any(np.abs(right_windows - extreme[:, None]) <= 1e-12, axis=1)
    else:
        right_match = np.zeros(len(windows), dtype=bool)
    confirmed = valid & is_extreme & ~right_match
    pivot_bars = np.flatnonzero(confirmed) + left
    for pivot_bar in pivot_bars.tolist():
        confirm_bar = pivot_bar + right
        events[confirm_bar] = {"pivot_bar": int(pivot_bar), "price": float(values[pivot_bar])}
    return events


def _rolling_mean(values: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    length = len(values)
    if length == 0:
        return np.asarray([], dtype=float)
    min_periods = window if min_periods is None else min_periods
    min_periods = max(1, min(window, int(min_periods)))
    valid_mask = ~np.isnan(values)
    filled = np.where(valid_mask, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccnt = np.concatenate(([0], np.cumsum(valid_mask.astype(int))))
    end_idx = np.arange(1, length + 1, dtype=int)
    start_idx = np.maximum(end_idx - window, 0)
    window_sums = csum[end_idx] - csum[start_idx]
    window_counts = ccnt[end_idx] - ccnt[start_idx]
    out = np.full(length, np.nan, dtype=float)
    valid = window_counts >= min_periods
    out[valid] = window_sums[valid] / window_counts[valid]
    return out


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) == 0:
        return out
    alpha = 2.0 / (float(span) + 1.0)
    last = np.nan
    for idx, value in enumerate(arr):
        if np.isnan(value):
            out[idx] = last
            continue
        last = value if np.isnan(last) else (alpha * value + (1.0 - alpha) * last)
        out[idx] = last
    return out


def build_limitup_l1l2_score(features: dict[str, Any]) -> float:
    stage = str(features.get("limitup_l1l2_stage") or "")
    l2_age = to_float(features.get("limitup_l1l2_bars_since_l2"))
    bars_lu_to_l1 = to_float(features.get("limitup_l1l2_bars_lu_to_l1"))
    bars_l1_to_l2 = to_float(features.get("limitup_l1l2_bars_l1_to_l2"))
    impulse_pct = to_float(features.get("limitup_l1l2_impulse_pct"))
    pullback_pct = to_float(features.get("limitup_l1l2_pullback_pct"))
    l2_above_l1_pct = to_float(features.get("limitup_l1l2_l2_above_l1_pct"))
    confirm_vol_ratio = to_float(features.get("limitup_l1l2_confirm_vol_ratio"))
    close_vs_l2_pct = to_float(features.get("limitup_l1l2_close_vs_l2_pct"))
    hold_buffer_pct = to_float(features.get("limitup_l1l2_hold_buffer_pct"))

    score = 0.0
    if stage == "pattern":
        score += 48.0
    elif stage == "l1_tracking":
        score += 10.0

    if to_bool(features.get("limitup_l1l2_limit_sealed")):
        score += 6.0

    if bars_lu_to_l1 is not None:
        if 2 <= bars_lu_to_l1 <= 15:
            score += 8.0
        elif bars_lu_to_l1 <= 25:
            score += 5.0
        elif bars_lu_to_l1 <= 35:
            score += 2.0

    if impulse_pct is not None:
        if 8 <= impulse_pct <= 18:
            score += 10.0
        elif 6 <= impulse_pct < 8 or 18 < impulse_pct <= 28:
            score += 7.0
        elif 28 < impulse_pct <= 40:
            score += 4.0
        elif impulse_pct > 40:
            score += 1.0

    if bars_l1_to_l2 is not None:
        if 4 <= bars_l1_to_l2 <= 18:
            score += 12.0
        elif 2 <= bars_l1_to_l2 <= 30:
            score += 8.0
        elif bars_l1_to_l2 <= 40:
            score += 4.0

    if l2_above_l1_pct is not None:
        if 0.8 <= l2_above_l1_pct <= 5:
            score += 18.0
        elif 0.3 <= l2_above_l1_pct <= 8:
            score += 14.0
        elif l2_above_l1_pct <= 12:
            score += 8.0
        if l2_above_l1_pct > 9:
            score -= 4.0

    if pullback_pct is not None:
        if 2 <= pullback_pct <= 8:
            score += 8.0
        elif 8 < pullback_pct <= 15:
            score += 5.0
        elif pullback_pct > 15:
            score -= 4.0

    if confirm_vol_ratio is not None:
        if confirm_vol_ratio >= 2.0:
            score += 6.0
        elif confirm_vol_ratio >= 1.5:
            score += 5.0
        elif confirm_vol_ratio >= 1.2:
            score += 4.0
        elif confirm_vol_ratio >= 1.05:
            score += 2.0

    if to_bool(features.get("limitup_l1l2_trend_ok")):
        score += 4.0
    if to_bool(features.get("limitup_l1l2_volume_ok")):
        score += 2.0

    if hold_buffer_pct is not None:
        if hold_buffer_pct >= 0.8:
            score += 8.0
        elif hold_buffer_pct >= 0:
            score += 5.0
        else:
            score -= 8.0

    if stage == "pattern" and l2_age is not None:
        if l2_age == 0:
            score += 12.0
        elif l2_age <= 3:
            score += 9.0
        elif l2_age <= 8:
            score += 6.0
        elif l2_age <= 15:
            score += 2.0
        elif l2_age > 25:
            score -= 5.0

    if close_vs_l2_pct is not None:
        if -2.0 <= close_vs_l2_pct <= 6.0:
            score += 10.0
        elif 6.0 < close_vs_l2_pct <= 12.0:
            score += 5.0
        elif close_vs_l2_pct > 18.0:
            score -= 8.0
        elif close_vs_l2_pct < -4.0:
            score -= 10.0

    return round(clip_score(score, 0.0, 100.0), 2)


def calculate_limitup_l1l2_features(
    daily_df: pd.DataFrame,
    ts_code: str = "",
    name: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = merge_limitup_l1l2_config(config)
    empty = _empty_result(ts_code)
    if daily_df is None or daily_df.empty:
        return empty

    work = daily_df.copy()
    if "trade_date" not in work.columns:
        return empty

    rename_map = {}
    if "vol" in work.columns and "volume" not in work.columns:
        rename_map["vol"] = "volume"
    if rename_map:
        work = work.rename(columns=rename_map)

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(work.columns)):
        return empty

    work["trade_date"] = work["trade_date"].astype(str)
    work = work.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "pre_close"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if len(work) < max(int(cfg["trend_ema_len"]), int(cfg["zz_left"]) + int(cfg["zz_right"]) + 5):
        return empty

    close_arr = work["close"].to_numpy(dtype=float)
    high_arr = work["high"].to_numpy(dtype=float)
    low_arr = work["low"].to_numpy(dtype=float)
    open_arr = work["open"].to_numpy(dtype=float)
    volume_arr = work["volume"].to_numpy(dtype=float)
    trade_dates = work["trade_date"].tolist()

    ema20_arr = _ema(close_arr, int(cfg["ema_len"]))
    ema60_arr = _ema(close_arr, int(cfg["trend_ema_len"]))
    vol_ma_arr = _rolling_mean(volume_arr, int(cfg["vol_ma_len"]), min_periods=1)
    prev_close_arr = np.roll(close_arr, 1)
    prev_close_arr[0] = np.nan

    limit_pct = infer_limit_pct(ts_code=ts_code, name=name, mode_limit=str(cfg.get("mode_limit", "AUTO")))
    tick = max(float(cfg["tick_size"]), 0.0001)
    limit_price = np.round((prev_close_arr * (1.0 + limit_pct)) / tick) * tick
    near_limit_ok = close_arr >= (limit_price - tick * int(cfg["tol_tick"]))
    close_tick = np.round(close_arr / tick) * tick
    high_tick = np.round(high_arr / tick) * tick
    seal_ok = close_tick == high_tick
    if not to_bool(cfg["need_seal"]):
        seal_ok = np.ones(len(work), dtype=bool)
    is_limit_up = np.asarray(near_limit_ok, dtype=bool) & np.asarray(seal_ok, dtype=bool)
    if len(is_limit_up):
        is_limit_up[0] = False

    trend_ok_series = np.ones(len(work), dtype=bool)
    if to_bool(cfg["use_trend_filter"]):
        ema60_prev = np.roll(ema60_arr, 1)
        ema60_prev[0] = np.nan
        trend_ok_series = np.asarray((ema60_arr > ema60_prev) & (close_arr > ema60_arr), dtype=bool)
        trend_ok_series[np.isnan(ema60_arr) | np.isnan(ema60_prev)] = False

    vol_up_ok_series = np.ones(len(work), dtype=bool)
    if to_bool(cfg["use_vol_filter"]):
        vol_up_ok_series = np.asarray(volume_arr > (vol_ma_arr * float(cfg["vol_mult_up"])), dtype=bool)
        vol_up_ok_series[np.isnan(vol_ma_arr)] = False

    low_events = _confirmed_pivot_events(low_arr, int(cfg["zz_left"]), int(cfg["zz_right"]), "low")
    state = 0
    lu_bar: int | None = None
    l1_price: float | None = None
    l1_bar: int | None = None
    l2_price: float | None = None
    l2_bar: int | None = None
    bars_after_l2 = 0

    def build_context(anchor_bar: int | None = None) -> dict[str, Any]:
        idx = len(work) - 1 if anchor_bar is None else int(anchor_bar)
        current_l2 = l2_price
        current_l1 = l1_price
        current_imp = None
        start_bar = l1_bar
        end_bar = l2_bar if l2_bar is not None else idx
        if start_bar is not None and end_bar is not None and end_bar >= start_bar:
            current_imp = float(np.nanmax(high_arr[start_bar : end_bar + 1]))
        volume_ratio = None
        if idx >= 0 and idx < len(work):
            base_vol_ma = to_float(vol_ma_arr[idx])
            if base_vol_ma not in (None, 0):
                volume_ratio = volume_arr[idx] / base_vol_ma
        return {
            "limitup_l1l2_limit_date": trade_dates[lu_bar] if lu_bar is not None else None,
            "limitup_l1l2_l1_date": trade_dates[l1_bar] if l1_bar is not None else None,
            "limitup_l1l2_l2_date": trade_dates[l2_bar] if l2_bar is not None else None,
            "limitup_l1l2_buy_date": trade_dates[idx] if anchor_bar is not None else None,
            "limitup_l1l2_limit_price": to_number(high_arr[lu_bar] if lu_bar is not None else None, 4),
            "limitup_l1l2_l1_price": to_number(current_l1, 4),
            "limitup_l1l2_l2_price": to_number(current_l2, 4),
            "limitup_l1l2_impulse_high": to_number(current_imp, 4),
            "limitup_l1l2_current_price": to_number(close_arr[idx], 4),
            "limitup_l1l2_bars_since_limit": (len(work) - 1 - lu_bar) if lu_bar is not None else None,
            "limitup_l1l2_bars_since_l2": (len(work) - 1 - l2_bar) if l2_bar is not None else None,
            "limitup_l1l2_bars_since_buy": (len(work) - 1 - idx) if anchor_bar is not None else None,
            "limitup_l1l2_bars_lu_to_l1": (l1_bar - lu_bar) if lu_bar is not None and l1_bar is not None else None,
            "limitup_l1l2_bars_l1_to_l2": (l2_bar - l1_bar) if l1_bar is not None and l2_bar is not None else None,
            "limitup_l1l2_impulse_pct": to_number(((current_imp / current_l1) - 1.0) * 100.0 if current_imp and current_l1 else None, 4),
            "limitup_l1l2_pullback_pct": to_number(((current_imp - current_l2) / current_imp) * 100.0 if current_imp and current_l2 else None, 4),
            "limitup_l1l2_l2_above_l1_pct": to_number(((current_l2 / current_l1) - 1.0) * 100.0 if current_l1 and current_l2 else None, 4),
            "limitup_l1l2_confirm_vol_ratio": to_number(volume_ratio, 4),
            "limitup_l1l2_close_vs_l2_pct": to_number(((close_arr[idx] / current_l2) - 1.0) * 100.0 if current_l2 else None, 4),
            "limitup_l1l2_hold_buffer_pct": to_number(((low_arr[idx] / current_l2) - 1.0) * 100.0 if current_l2 else None, 4),
            "limitup_l1l2_trend_ok": bool(trend_ok_series[idx]),
            "limitup_l1l2_volume_ok": bool(vol_up_ok_series[idx]),
            "limitup_l1l2_limit_sealed": bool(is_limit_up[lu_bar]) if lu_bar is not None else False,
        }

    for i in range(len(work)):
        new_low = low_events[i]

        if bool(is_limit_up[i]) and (to_bool(cfg["restart_every_limit_up"]) or state == 0):
            state = 1
            lu_bar = i
            l1_price = None
            l1_bar = None
            l2_price = None
            l2_bar = None
            bars_after_l2 = 0
        pivot_low_bar = int(new_low["pivot_bar"]) if new_low is not None else None
        pivot_low_price = float(new_low["price"]) if new_low is not None else None

        if state == 1 and lu_bar is not None:
            if (i - lu_bar) > int(cfg["l1_max_bars"]):
                state = 0
            elif pivot_low_bar is not None and pivot_low_bar > lu_bar and (pivot_low_bar - lu_bar) <= int(cfg["l1_max_bars"]):
                l1_price = pivot_low_price
                l1_bar = pivot_low_bar
                l2_price = None
                l2_bar = None
                bars_after_l2 = 0
                state = 2

        elif state in {2, 3} and l1_price is not None and l1_bar is not None:
            if pivot_low_bar is not None and pivot_low_bar > l1_bar:
                if pivot_low_price <= l1_price * (1.0 - float(cfg["l1_break_pct"]) / 100.0) or pivot_low_price <= l1_price:
                    l1_price = pivot_low_price
                    l1_bar = pivot_low_bar
                    l2_price = None
                    l2_bar = None
                    bars_after_l2 = 0
                    state = 2
                else:
                    bars_between = pivot_low_bar - l1_bar
                    if bars_between > int(cfg["max_bars_between_lows"]):
                        l1_price = pivot_low_price
                        l1_bar = pivot_low_bar
                        l2_price = None
                        l2_bar = None
                        bars_after_l2 = 0
                        state = 2
                    else:
                        min_ok = pivot_low_price >= l1_price * (1.0 + float(cfg["higher_low_min_pct"]) / 100.0)
                        max_ok = pivot_low_price <= l1_price * (1.0 + float(cfg["higher_low_max_pct"]) / 100.0)
                        within_range = bars_between >= int(cfg["min_bars_between_lows"])
                        if within_range and min_ok and max_ok:
                            l2_price = pivot_low_price
                            l2_bar = pivot_low_bar
                            bars_after_l2 = 0
                            state = 3

            if state == 3 and l2_bar is not None:
                bars_after_l2 = i - l2_bar
                if bars_after_l2 > int(cfg["rebound_max_bars"]):
                    state = 0

    result = _empty_result(ts_code)
    pattern_valid = bool(
        state == 3
        and l1_price is not None
        and l2_price is not None
        and l2_price > l1_price
        and float(low_arr[-1]) > l1_price * (1.0 - float(cfg["l1_break_pct"]) / 100.0)
    )
    active_context = build_context() if pattern_valid else None
    tracking_context = build_context() if state in {2, 3} and l1_price is not None else None

    recent_buy_window = int(cfg["recent_buy_window"])
    stage = ""
    reason = "无有效形态"
    source_context: dict[str, Any] | None = None

    if active_context is not None:
        stage = "pattern"
        reason = "涨停后两次回调，L2高于L1"
        source_context = active_context
    elif tracking_context is not None:
        stage = "l1_tracking"
        reason = "已有L1，等待更高的L2"
        source_context = tracking_context

    if source_context:
        result.update(source_context)
    result["limitup_l1l2_stage"] = stage
    result["limitup_l1l2_reason"] = reason if stage else ""
    l2_age = to_float(result.get("limitup_l1l2_bars_since_l2"))
    result["limitup_l1l2_buy_signal"] = stage == "pattern" and l2_age == 0
    result["limitup_l1l2_buy_recent"] = stage == "pattern" and recent_buy_window > 0 and l2_age is not None and l2_age <= recent_buy_window
    result["limitup_l1l2_ready"] = stage == "pattern"
    result["limitup_l1l2_score"] = build_limitup_l1l2_score(result)
    result["limitup_l1l2_signal"] = bool(
        stage == "pattern"
        and result["limitup_l1l2_score"] >= float(cfg["candidate_score_threshold"])
    )
    return result


def build_limitup_l1l2_snapshot(
    daily_history_df: pd.DataFrame,
    stock_basic_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if daily_history_df is None or daily_history_df.empty or "ts_code" not in daily_history_df.columns:
        return pd.DataFrame()

    name_map: dict[str, str] = {}
    if stock_basic_df is not None and not stock_basic_df.empty and {"ts_code", "name"}.issubset(stock_basic_df.columns):
        basic = stock_basic_df[["ts_code", "name"]].drop_duplicates(subset=["ts_code"], keep="last").copy()
        name_map = dict(zip(basic["ts_code"].astype(str), basic["name"].fillna("").astype(str)))

    rows: list[dict[str, Any]] = []
    ordered = daily_history_df.copy()
    ordered["ts_code"] = ordered["ts_code"].astype(str)
    for ts_code, sub in ordered.groupby("ts_code", sort=False, dropna=False):
        rows.append(
            calculate_limitup_l1l2_features(
                sub,
                ts_code=str(ts_code),
                name=name_map.get(str(ts_code), ""),
                config=config,
            )
        )
    return pd.DataFrame(rows)
