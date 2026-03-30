from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PLATFORM_BREAKOUT_CONFIG: dict[str, Any] = {
    "limit_window_bars": 5,
    "platform_min_bars": 3,
    "platform_max_bars": 10,
    "platform_amp_max_pct": 15.0,
    "platform_close_span_max_pct": 12.0,
    "platform_net_change_max_pct": 9.0,
    "platform_std_max_pct": 4.8,
    "main_limit_tol_pct": 0.6,
    "gem_limit_tol_pct": 1.0,
    "breakout_close_buffer_pct": 2.0,
    "limit_volume_mult_min": 1.3,
    "one_word_range_pct_max": 1.2,
    "limit_body_pct_min": 1.6,
    "ma_spread_max_pct": 10.0,
    "main_pullback_min_ratio": 0.50,
    "main_pullback_max_ratio": 0.67,
    "gem_pullback_min_ratio": 0.45,
    "gem_pullback_max_ratio": 0.70,
    "golden_pullback_low": 0.50,
    "golden_pullback_high": 0.618,
    "pullback_avg_vol_ratio_max": 0.70,
    "tail_shrink_limit_ratio": 0.60,
    "platform_support_break_pct": 2.0,
    "ma20_break_pct": 2.0,
    "main_big_drop_pct": -4.5,
    "gem_big_drop_pct": -8.0,
    "big_drop_volume_mult": 1.10,
    "strength_volume_prev_mult": 1.05,
    "strength_volume_ma_mult": 0.95,
    "candidate_score_threshold": 60.0,
}


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


def clip_score(value: Any, low: float, high: float) -> float:
    numeric = to_float(value)
    if numeric is None:
        return low
    return float(min(max(numeric, low), high))


def merge_platform_breakout_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_PLATFORM_BREAKOUT_CONFIG)
    if config:
        merged.update({key: value for key, value in config.items() if value is not None})
    return merged


def _board_from_meta(meta: dict[str, Any]) -> str:
    market = str(meta.get("market") or "").strip()
    ts_code = str(meta.get("ts_code") or "").upper()
    if market == "创业板" or ts_code.startswith(("300", "301")):
        return "gem"
    if market == "主板":
        return "main"
    if market in {"科创板", "北交所"}:
        return ""
    if ts_code.startswith(("688", "689", "8")):
        return ""
    return "main" if ts_code.endswith((".SH", ".SZ")) else ""


def _is_allowed_stock(meta: dict[str, Any]) -> bool:
    name = str(meta.get("name") or "").upper()
    market = str(meta.get("market") or "").strip()
    ts_code = str(meta.get("ts_code") or "").upper()
    if "ST" in name:
        return False
    if market in {"科创板", "北交所"}:
        return False
    if ts_code.startswith(("688", "689", "8")):
        return False
    return _board_from_meta(meta) in {"main", "gem"}


def _empty_result(ts_code: str = "") -> dict[str, Any]:
    return {
        "ts_code": ts_code,
        "platform_breakout_stage": "",
        "platform_breakout_signal": False,
        "platform_breakout_score": 0.0,
        "platform_breakout_reason": "",
        "platform_breakout_board": "",
        "platform_breakout_limit_date": None,
        "platform_breakout_platform_start": None,
        "platform_breakout_platform_end": None,
        "platform_breakout_pullback_low_date": None,
        "platform_breakout_stop_date": None,
        "platform_breakout_strength_date": None,
        "platform_breakout_platform_high": None,
        "platform_breakout_platform_low": None,
        "platform_breakout_limit_high": None,
        "platform_breakout_pullback_low_price": None,
        "platform_breakout_support_floor": None,
        "platform_breakout_current_price": None,
        "platform_breakout_platform_days": None,
        "platform_breakout_platform_amp_pct": None,
        "platform_breakout_platform_close_span_pct": None,
        "platform_breakout_platform_volume_ratio": None,
        "platform_breakout_limit_gain_pct": None,
        "platform_breakout_limit_volume_ratio": None,
        "platform_breakout_breakout_close_buffer_pct": None,
        "platform_breakout_limit_body_pct": None,
        "platform_breakout_limit_upper_shadow_pct": None,
        "platform_breakout_pullback_ratio": None,
        "platform_breakout_pullback_avg_vol_ratio": None,
        "platform_breakout_tail_vol_ratio": None,
        "platform_breakout_support_buffer_pct": None,
        "platform_breakout_ma20_buffer_pct": None,
        "platform_breakout_ma_spread_pct": None,
        "platform_breakout_ma5_slope_pct": None,
        "platform_breakout_ma10_slope_pct": None,
        "platform_breakout_ma20_slope_pct": None,
        "platform_breakout_current_volume_ratio": None,
        "platform_breakout_current_close_to_high_pct": None,
        "platform_breakout_pre20_runup_pct": None,
        "platform_breakout_big_drop_count": 0,
        "platform_breakout_stop_signal": False,
        "platform_breakout_strength_signal": False,
        "platform_breakout_strength_break_prev_high": False,
        "platform_breakout_strength_reclaim_ma5": False,
        "platform_breakout_strength_engulf": False,
    }


def _percent_distance(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return (float(numerator) / float(denominator) - 1.0) * 100.0


def _upper_shadow_pct(open_price: float, high_price: float, close_price: float, pre_close: float | None) -> float | None:
    if pre_close in (None, 0):
        return None
    return max(float(high_price) - max(float(open_price), float(close_price)), 0.0) / float(pre_close) * 100.0


def _body_pct(open_price: float, close_price: float, pre_close: float | None) -> float | None:
    if pre_close in (None, 0):
        return None
    return abs(float(close_price) - float(open_price)) / float(pre_close) * 100.0


def _range_pct(high_price: float, low_price: float, pre_close: float | None) -> float | None:
    if pre_close in (None, 0):
        return None
    return (float(high_price) - float(low_price)) / float(pre_close) * 100.0


def _is_stop_signal(
    idx: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    vol_ma5_arr: np.ndarray,
    pre_close_arr: np.ndarray,
) -> bool:
    if idx < 0 or idx >= len(close_arr):
        return False
    open_price = float(open_arr[idx])
    high_price = float(high_arr[idx])
    low_price = float(low_arr[idx])
    close_price = float(close_arr[idx])
    volume = float(volume_arr[idx])
    range_value = max(high_price - low_price, 1e-9)
    body = abs(close_price - open_price)
    lower_shadow = min(open_price, close_price) - low_price
    body_ratio = body / range_value
    lower_shadow_ratio = max(lower_shadow, 0.0) / range_value
    vol_ma5 = vol_ma5_arr[idx]
    range_pct = _range_pct(high_price, low_price, pre_close_arr[idx])

    bullish = close_price > open_price
    long_lower_shadow = lower_shadow_ratio >= 0.38 and body_ratio <= 0.55
    shrink_small_body = body_ratio <= 0.32 and (np.isnan(vol_ma5) or volume <= vol_ma5 * 0.98)
    narrow_range = range_pct is not None and range_pct <= 3.2
    return bool(bullish or long_lower_shadow or (shrink_small_body and narrow_range))


def _strength_flags(
    idx: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    ma5_arr: np.ndarray,
) -> dict[str, bool]:
    if idx <= 0 or idx >= len(close_arr):
        return {
            "reclaim_ma5": False,
            "break_prev_high": False,
            "engulf": False,
        }
    current_open = float(open_arr[idx])
    current_close = float(close_arr[idx])
    prev_open = float(open_arr[idx - 1])
    prev_close = float(close_arr[idx - 1])
    prev_high = float(high_arr[idx - 1])
    ma5 = ma5_arr[idx]
    reclaim_ma5 = not np.isnan(ma5) and current_close > ma5
    break_prev_high = current_close > prev_high
    engulf = current_close > current_open and current_close >= max(prev_open, prev_close) and current_open <= min(prev_open, prev_close)
    return {
        "reclaim_ma5": bool(reclaim_ma5),
        "break_prev_high": bool(break_prev_high),
        "engulf": bool(engulf),
    }


def build_platform_breakout_score(features: dict[str, Any]) -> float:
    platform_amp = to_float(features.get("platform_breakout_platform_amp_pct"))
    platform_days = to_float(features.get("platform_breakout_platform_days"))
    platform_vol_ratio = to_float(features.get("platform_breakout_platform_volume_ratio"))
    limit_vol_ratio = to_float(features.get("platform_breakout_limit_volume_ratio"))
    breakout_buffer = to_float(features.get("platform_breakout_breakout_close_buffer_pct"))
    limit_body_pct = to_float(features.get("platform_breakout_limit_body_pct"))
    limit_upper_shadow_pct = to_float(features.get("platform_breakout_limit_upper_shadow_pct"))
    pullback_ratio = to_float(features.get("platform_breakout_pullback_ratio"))
    pullback_avg_vol_ratio = to_float(features.get("platform_breakout_pullback_avg_vol_ratio"))
    support_buffer_pct = to_float(features.get("platform_breakout_support_buffer_pct"))
    ma20_buffer_pct = to_float(features.get("platform_breakout_ma20_buffer_pct"))
    ma_spread_pct = to_float(features.get("platform_breakout_ma_spread_pct"))
    ma5_slope_pct = to_float(features.get("platform_breakout_ma5_slope_pct"))
    ma10_slope_pct = to_float(features.get("platform_breakout_ma10_slope_pct"))
    ma20_slope_pct = to_float(features.get("platform_breakout_ma20_slope_pct"))
    current_volume_ratio = to_float(features.get("platform_breakout_current_volume_ratio"))
    current_close_to_high_pct = to_float(features.get("platform_breakout_current_close_to_high_pct"))
    pre20_runup_pct = to_float(features.get("platform_breakout_pre20_runup_pct"))
    big_drop_count = int(to_float(features.get("platform_breakout_big_drop_count")) or 0)

    score = 0.0

    if platform_amp is not None:
        if platform_amp <= 6:
            score += 8.0
        elif platform_amp <= 9:
            score += 6.0
        elif platform_amp <= 12:
            score += 4.0
        else:
            score += 2.0
    if platform_days is not None:
        if 4 <= platform_days <= 8:
            score += 6.0
        elif platform_days in {3, 9}:
            score += 4.0
        else:
            score += 2.0
    if platform_vol_ratio is not None:
        if platform_vol_ratio <= 0.85:
            score += 6.0
        elif platform_vol_ratio <= 1.0:
            score += 4.0
        elif platform_vol_ratio <= 1.1:
            score += 2.0

    if limit_vol_ratio is not None:
        if limit_vol_ratio >= 2.2:
            score += 10.0
        elif limit_vol_ratio >= 1.8:
            score += 8.0
        elif limit_vol_ratio >= 1.5:
            score += 6.0
        elif limit_vol_ratio >= 1.3:
            score += 4.0
    if breakout_buffer is not None:
        if breakout_buffer >= 6:
            score += 8.0
        elif breakout_buffer >= 4:
            score += 6.0
        elif breakout_buffer >= 2:
            score += 4.0
    if limit_body_pct is not None:
        if limit_body_pct >= 5.0:
            score += 4.0
        elif limit_body_pct >= 3.0:
            score += 3.0
        elif limit_body_pct >= 2.0:
            score += 2.0
    if limit_upper_shadow_pct is not None:
        if limit_upper_shadow_pct <= 0.5:
            score += 3.0
        elif limit_upper_shadow_pct <= 1.2:
            score += 2.0
        elif limit_upper_shadow_pct <= 2.0:
            score += 1.0

    if pullback_ratio is not None:
        if 0.50 <= pullback_ratio <= 0.618:
            score += 10.0
        elif 0.45 <= pullback_ratio < 0.50 or 0.618 < pullback_ratio <= 0.67:
            score += 7.0
        elif 0.67 < pullback_ratio <= 0.70:
            score += 4.0
    if pullback_avg_vol_ratio is not None:
        if pullback_avg_vol_ratio <= 0.45:
            score += 8.0
        elif pullback_avg_vol_ratio <= 0.55:
            score += 6.0
        elif pullback_avg_vol_ratio <= 0.70:
            score += 4.0
    if support_buffer_pct is not None and ma20_buffer_pct is not None:
        if support_buffer_pct >= 0 and ma20_buffer_pct >= 0:
            score += 7.0
        elif support_buffer_pct >= -1.0 and ma20_buffer_pct >= -1.0:
            score += 4.0
        else:
            score += 1.0

    if ma_spread_pct is not None:
        if 2.0 <= ma_spread_pct <= 6.0:
            score += 5.0
        elif ma_spread_pct <= 8.0:
            score += 4.0
        elif ma_spread_pct <= 10.0:
            score += 2.0
    positive_slopes = [
        slope
        for slope in [ma5_slope_pct, ma10_slope_pct, ma20_slope_pct]
        if slope is not None and slope > 0
    ]
    if len(positive_slopes) == 3:
        avg_slope = sum(positive_slopes) / 3.0
        if 0.2 <= avg_slope <= 1.8:
            score += 5.0
        else:
            score += 3.0
    elif len(positive_slopes) == 2:
        score += 2.0

    strength_signal_count = int(bool(features.get("platform_breakout_strength_reclaim_ma5"))) + int(
        bool(features.get("platform_breakout_strength_break_prev_high"))
    ) + int(bool(features.get("platform_breakout_strength_engulf")))
    if strength_signal_count >= 3:
        score += 8.0
    elif strength_signal_count == 2:
        score += 6.0
    elif strength_signal_count == 1:
        score += 4.0

    if current_volume_ratio is not None:
        if current_volume_ratio >= 1.35:
            score += 6.0
        elif current_volume_ratio >= 1.1:
            score += 5.0
        elif current_volume_ratio >= 0.95:
            score += 3.0
    if current_close_to_high_pct is not None:
        if current_close_to_high_pct <= 0.5:
            score += 6.0
        elif current_close_to_high_pct <= 1.2:
            score += 4.0
        elif current_close_to_high_pct <= 2.0:
            score += 2.0

    penalty = 0.0
    if pre20_runup_pct is not None:
        if pre20_runup_pct >= 45:
            penalty += 10.0
        elif pre20_runup_pct >= 30:
            penalty += 6.0
        elif pre20_runup_pct >= 20:
            penalty += 3.0
    penalty += min(big_drop_count, 2) * 4.0
    if current_volume_ratio is not None and current_volume_ratio < 1.0:
        penalty += 4.0
    if current_close_to_high_pct is not None:
        if current_close_to_high_pct > 3.0:
            penalty += 4.0
        elif current_close_to_high_pct > 2.0:
            penalty += 2.0
    if limit_upper_shadow_pct is not None and limit_upper_shadow_pct > 2.0:
        penalty += 3.0

    return round(clip_score(score - penalty, 0.0, 100.0), 2)


def calculate_platform_breakout_features(
    daily_df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = meta or {}
    cfg = merge_platform_breakout_config(config)
    ts_code = str(meta.get("ts_code") or daily_df.get("ts_code", pd.Series([""])).iloc[-1] or "")
    result = _empty_result(ts_code)

    if daily_df is None or daily_df.empty:
        result["platform_breakout_reason"] = "missing_history"
        return result
    if not _is_allowed_stock(meta):
        result["platform_breakout_reason"] = "universe_excluded"
        return result

    work = daily_df.copy().sort_values("trade_date").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "pre_close", "vol"]:
        if column not in work.columns:
            result["platform_breakout_reason"] = f"missing_{column}"
            return result
    if len(work) < 25:
        result["platform_breakout_reason"] = "history_too_short"
        return result

    board = _board_from_meta(meta)
    if board not in {"main", "gem"}:
        result["platform_breakout_reason"] = "board_excluded"
        return result

    open_arr = pd.to_numeric(work["open"], errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(work["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(work["low"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(work["close"], errors="coerce").to_numpy(dtype=float)
    pre_close_arr = pd.to_numeric(work["pre_close"], errors="coerce").to_numpy(dtype=float)
    volume_arr = pd.to_numeric(work["vol"], errors="coerce").to_numpy(dtype=float)
    trade_dates = work["trade_date"].astype(str).tolist()

    if np.isnan(close_arr[-1]) or np.isnan(volume_arr[-1]):
        result["platform_breakout_reason"] = "latest_bar_invalid"
        return result

    close_series = pd.Series(close_arr)
    volume_series = pd.Series(volume_arr)
    ma5_arr = close_series.rolling(5).mean().to_numpy(dtype=float)
    ma10_arr = close_series.rolling(10).mean().to_numpy(dtype=float)
    ma20_arr = close_series.rolling(20).mean().to_numpy(dtype=float)
    vol_ma5_arr = volume_series.rolling(5).mean().to_numpy(dtype=float)

    current_idx = len(work) - 1
    if current_idx < 2 or any(np.isnan(arr[current_idx]) for arr in [ma5_arr, ma10_arr, ma20_arr]):
        result["platform_breakout_reason"] = "ma_not_ready"
        return result

    ma_order_ok = ma5_arr[current_idx] > ma10_arr[current_idx] > ma20_arr[current_idx]
    ma_slope_ok = (
        ma5_arr[current_idx] > ma5_arr[current_idx - 1]
        and ma10_arr[current_idx] > ma10_arr[current_idx - 1]
        and ma20_arr[current_idx] > ma20_arr[current_idx - 1]
    )
    ma_spread_pct = _percent_distance(ma5_arr[current_idx], ma20_arr[current_idx])
    if not ma_order_ok:
        result["platform_breakout_reason"] = "ma_order_failed"
        return result
    if not ma_slope_ok:
        result["platform_breakout_reason"] = "ma_slope_failed"
        return result
    if ma_spread_pct is None or ma_spread_pct > float(cfg["ma_spread_max_pct"]):
        result["platform_breakout_reason"] = "ma_spread_too_wide"
        return result

    pct_change = np.where(pre_close_arr > 0, (close_arr / pre_close_arr - 1.0) * 100.0, np.nan)
    limit_tol = float(cfg["gem_limit_tol_pct"] if board == "gem" else cfg["main_limit_tol_pct"])
    limit_threshold = 20.0 - limit_tol if board == "gem" else 10.0 - limit_tol
    limit_window_bars = int(cfg["limit_window_bars"])
    recent_start = max(0, len(work) - limit_window_bars)
    recent_high = np.nanmax(high_arr[recent_start:])
    candidate_limit_indexes = [
        idx
        for idx in range(recent_start, len(work) - 1)
        if not np.isnan(pct_change[idx]) and pct_change[idx] >= limit_threshold and close_arr[idx] >= high_arr[idx] * 0.985
    ]
    if not candidate_limit_indexes:
        result["platform_breakout_reason"] = "no_recent_limit_up"
        return result

    best_candidate: dict[str, Any] | None = None
    best_score = -1.0
    for limit_idx in reversed(candidate_limit_indexes):
        if high_arr[limit_idx] < recent_high * 0.999:
            continue
        if current_idx - limit_idx < 2:
            continue
        if limit_idx < 5:
            continue

        limit_range_pct = _range_pct(high_arr[limit_idx], low_arr[limit_idx], pre_close_arr[limit_idx])
        limit_body_pct = _body_pct(open_arr[limit_idx], close_arr[limit_idx], pre_close_arr[limit_idx])
        if limit_range_pct is None or limit_range_pct <= float(cfg["one_word_range_pct_max"]):
            continue
        if limit_body_pct is None or limit_body_pct <= float(cfg["limit_body_pct_min"]):
            continue

        pre_limit_avg_vol = np.nanmean(volume_arr[limit_idx - 5 : limit_idx])
        if pre_limit_avg_vol <= 0 or volume_arr[limit_idx] < pre_limit_avg_vol * float(cfg["limit_volume_mult_min"]):
            continue

        platform_choice: dict[str, Any] | None = None
        for platform_days in range(int(cfg["platform_min_bars"]), int(cfg["platform_max_bars"]) + 1):
            start_idx = limit_idx - platform_days
            if start_idx < 0:
                continue
            platform_high = float(np.nanmax(high_arr[start_idx:limit_idx]))
            platform_low = float(np.nanmin(low_arr[start_idx:limit_idx]))
            if platform_low <= 0:
                continue
            platform_amp_pct = (platform_high / platform_low - 1.0) * 100.0
            platform_close_slice = close_arr[start_idx:limit_idx]
            platform_close_span_pct = (np.nanmax(platform_close_slice) / np.nanmin(platform_close_slice) - 1.0) * 100.0
            platform_net_change_pct = abs(close_arr[limit_idx - 1] / close_arr[start_idx] - 1.0) * 100.0
            platform_std_pct = np.nanstd(platform_close_slice) / np.nanmean(platform_close_slice) * 100.0
            if platform_amp_pct > float(cfg["platform_amp_max_pct"]):
                continue
            if platform_close_span_pct > float(cfg["platform_close_span_max_pct"]):
                continue
            if platform_net_change_pct > float(cfg["platform_net_change_max_pct"]):
                continue
            if platform_std_pct > float(cfg["platform_std_max_pct"]):
                continue

            breakout_close_buffer_pct = _percent_distance(close_arr[limit_idx], platform_high)
            if high_arr[limit_idx] <= platform_high or breakout_close_buffer_pct is None or breakout_close_buffer_pct < float(
                cfg["breakout_close_buffer_pct"]
            ):
                continue

            pre_platform_avg_vol = np.nan
            if start_idx >= 5:
                pre_platform_avg_vol = np.nanmean(volume_arr[start_idx - 5 : start_idx])
            platform_avg_vol = np.nanmean(volume_arr[start_idx:limit_idx])
            platform_vol_ratio = None
            if not np.isnan(pre_platform_avg_vol) and pre_platform_avg_vol > 0:
                platform_vol_ratio = float(platform_avg_vol / pre_platform_avg_vol)

            temp_score = 0.0
            temp_score += max(0.0, 16.0 - platform_amp_pct)
            temp_score += 4.0 if 4 <= platform_days <= 8 else 2.0
            if platform_vol_ratio is not None:
                temp_score += max(0.0, 1.15 - platform_vol_ratio) * 8.0

            candidate = {
                "start_idx": start_idx,
                "platform_days": platform_days,
                "platform_high": platform_high,
                "platform_low": platform_low,
                "platform_amp_pct": platform_amp_pct,
                "platform_close_span_pct": platform_close_span_pct,
                "platform_vol_ratio": platform_vol_ratio,
                "breakout_close_buffer_pct": breakout_close_buffer_pct,
                "selection_temp_score": temp_score,
            }
            if platform_choice is None or candidate["selection_temp_score"] > platform_choice["selection_temp_score"]:
                platform_choice = candidate
        if platform_choice is None:
            continue

        support_floor = platform_choice["platform_high"] * (1.0 - float(cfg["platform_support_break_pct"]) / 100.0)
        pullback_search_low = low_arr[limit_idx + 1 : current_idx]
        if len(pullback_search_low) == 0:
            continue
        low_offset = int(np.nanargmin(pullback_search_low))
        pullback_low_idx = limit_idx + 1 + low_offset
        pullback_low = float(low_arr[pullback_low_idx])
        denominator = float(high_arr[limit_idx] - platform_choice["platform_low"])
        if denominator <= 0:
            continue
        pullback_ratio = (float(high_arr[limit_idx]) - pullback_low) / denominator
        min_ratio = float(cfg["gem_pullback_min_ratio"] if board == "gem" else cfg["main_pullback_min_ratio"])
        max_ratio = float(cfg["gem_pullback_max_ratio"] if board == "gem" else cfg["main_pullback_max_ratio"])
        if pullback_ratio < min_ratio or pullback_ratio > max_ratio:
            continue

        pullback_phase = volume_arr[limit_idx + 1 : current_idx]
        if len(pullback_phase) == 0:
            continue
        pullback_avg_vol_ratio = float(np.nanmean(pullback_phase) / volume_arr[limit_idx]) if volume_arr[limit_idx] > 0 else None
        if pullback_avg_vol_ratio is None or pullback_avg_vol_ratio > float(cfg["pullback_avg_vol_ratio_max"]):
            continue

        tail_idx = current_idx - 1
        tail_vol_ratio = float(volume_arr[tail_idx] / volume_arr[limit_idx]) if volume_arr[limit_idx] > 0 else None
        tail_shrink_ok = False
        if tail_vol_ratio is not None and tail_vol_ratio <= float(cfg["tail_shrink_limit_ratio"]):
            tail_shrink_ok = True
        if not np.isnan(vol_ma5_arr[tail_idx]) and volume_arr[tail_idx] <= vol_ma5_arr[tail_idx]:
            tail_shrink_ok = True
        if not tail_shrink_ok:
            continue

        big_drop_threshold = float(cfg["gem_big_drop_pct"] if board == "gem" else cfg["main_big_drop_pct"])
        big_drop_count = 0
        pullback_broken = False
        for idx in range(limit_idx + 1, current_idx):
            vol_base = vol_ma5_arr[idx]
            if np.isnan(vol_base) or vol_base <= 0:
                vol_base = volume_arr[idx - 1] if idx > 0 else volume_arr[idx]
            heavy_volume = volume_arr[idx] >= max(volume_arr[limit_idx] * float(cfg["big_drop_volume_mult"]), vol_base * float(cfg["big_drop_volume_mult"]))
            bearish_bar = close_arr[idx] < open_arr[idx]
            if bearish_bar and pct_change[idx] <= big_drop_threshold and heavy_volume:
                pullback_broken = True
                big_drop_count += 1
        if pullback_broken:
            continue

        if pullback_low < support_floor:
            continue
        if np.isnan(ma20_arr[pullback_low_idx]) or pullback_low < ma20_arr[pullback_low_idx] * (1.0 - float(cfg["ma20_break_pct"]) / 100.0):
            continue

        stop_signal = any(
            _is_stop_signal(idx, open_arr, high_arr, low_arr, close_arr, volume_arr, vol_ma5_arr, pre_close_arr)
            for idx in range(max(limit_idx + 1, current_idx - 1), current_idx + 1)
        )
        if not stop_signal:
            continue

        strength_flags = _strength_flags(current_idx, open_arr, high_arr, low_arr, close_arr, ma5_arr)
        strength_signal = any(strength_flags.values())
        if not strength_signal:
            continue

        current_vol_ref = max(
            volume_arr[current_idx - 1] * float(cfg["strength_volume_prev_mult"]),
            (vol_ma5_arr[current_idx] if not np.isnan(vol_ma5_arr[current_idx]) else volume_arr[current_idx - 1])
            * float(cfg["strength_volume_ma_mult"]),
        )
        if volume_arr[current_idx] < current_vol_ref:
            continue

        current_close_to_high_pct = (
            max(float(high_arr[current_idx]) - float(close_arr[current_idx]), 0.0) / float(close_arr[current_idx]) * 100.0
            if close_arr[current_idx] > 0
            else None
        )
        support_buffer_pct = _percent_distance(pullback_low, platform_choice["platform_high"])
        ma20_buffer_pct = _percent_distance(pullback_low, ma20_arr[pullback_low_idx])
        current_volume_ratio = (
            float(volume_arr[current_idx] / max(volume_arr[current_idx - 1], vol_ma5_arr[current_idx] if not np.isnan(vol_ma5_arr[current_idx]) else 1.0))
            if current_idx > 0
            else None
        )
        pre20_idx = max(0, limit_idx - 20)
        pre20_runup_pct = abs(_percent_distance(close_arr[limit_idx], close_arr[pre20_idx]) or 0.0)

        candidate_features = {
            "ts_code": ts_code,
            "platform_breakout_stage": "ready",
            "platform_breakout_signal": True,
            "platform_breakout_board": board,
            "platform_breakout_limit_date": trade_dates[limit_idx],
            "platform_breakout_platform_start": trade_dates[platform_choice["start_idx"]],
            "platform_breakout_platform_end": trade_dates[limit_idx - 1],
            "platform_breakout_pullback_low_date": trade_dates[pullback_low_idx],
            "platform_breakout_stop_date": trade_dates[tail_idx],
            "platform_breakout_strength_date": trade_dates[current_idx],
            "platform_breakout_platform_high": to_number(platform_choice["platform_high"], 4),
            "platform_breakout_platform_low": to_number(platform_choice["platform_low"], 4),
            "platform_breakout_limit_high": to_number(high_arr[limit_idx], 4),
            "platform_breakout_pullback_low_price": to_number(pullback_low, 4),
            "platform_breakout_support_floor": to_number(support_floor, 4),
            "platform_breakout_current_price": to_number(close_arr[current_idx], 4),
            "platform_breakout_platform_days": int(platform_choice["platform_days"]),
            "platform_breakout_platform_amp_pct": to_number(platform_choice["platform_amp_pct"], 4),
            "platform_breakout_platform_close_span_pct": to_number(platform_choice["platform_close_span_pct"], 4),
            "platform_breakout_platform_volume_ratio": to_number(platform_choice["platform_vol_ratio"], 4),
            "platform_breakout_limit_gain_pct": to_number(pct_change[limit_idx], 4),
            "platform_breakout_limit_volume_ratio": to_number(volume_arr[limit_idx] / pre_limit_avg_vol, 4),
            "platform_breakout_breakout_close_buffer_pct": to_number(platform_choice["breakout_close_buffer_pct"], 4),
            "platform_breakout_limit_body_pct": to_number(limit_body_pct, 4),
            "platform_breakout_limit_upper_shadow_pct": to_number(
                _upper_shadow_pct(open_arr[limit_idx], high_arr[limit_idx], close_arr[limit_idx], pre_close_arr[limit_idx]), 4
            ),
            "platform_breakout_pullback_ratio": to_number(pullback_ratio, 4),
            "platform_breakout_pullback_avg_vol_ratio": to_number(pullback_avg_vol_ratio, 4),
            "platform_breakout_tail_vol_ratio": to_number(tail_vol_ratio, 4),
            "platform_breakout_support_buffer_pct": to_number(support_buffer_pct, 4),
            "platform_breakout_ma20_buffer_pct": to_number(ma20_buffer_pct, 4),
            "platform_breakout_ma_spread_pct": to_number(ma_spread_pct, 4),
            "platform_breakout_ma5_slope_pct": to_number(_percent_distance(ma5_arr[current_idx], ma5_arr[current_idx - 1]), 4),
            "platform_breakout_ma10_slope_pct": to_number(_percent_distance(ma10_arr[current_idx], ma10_arr[current_idx - 1]), 4),
            "platform_breakout_ma20_slope_pct": to_number(_percent_distance(ma20_arr[current_idx], ma20_arr[current_idx - 1]), 4),
            "platform_breakout_current_volume_ratio": to_number(current_volume_ratio, 4),
            "platform_breakout_current_close_to_high_pct": to_number(current_close_to_high_pct, 4),
            "platform_breakout_pre20_runup_pct": to_number(pre20_runup_pct, 4),
            "platform_breakout_big_drop_count": int(big_drop_count),
            "platform_breakout_stop_signal": bool(stop_signal),
            "platform_breakout_strength_signal": bool(strength_signal),
            "platform_breakout_strength_break_prev_high": bool(strength_flags["break_prev_high"]),
            "platform_breakout_strength_reclaim_ma5": bool(strength_flags["reclaim_ma5"]),
            "platform_breakout_strength_engulf": bool(strength_flags["engulf"]),
        }
        candidate_features["platform_breakout_score"] = build_platform_breakout_score(candidate_features)
        candidate_features["platform_breakout_reason"] = (
            f"平台{candidate_features['platform_breakout_platform_days']}天,"
            f" 振幅{candidate_features['platform_breakout_platform_amp_pct']}%,"
            f" 回撤{candidate_features['platform_breakout_pullback_ratio']},"
            f" 转强量能{candidate_features['platform_breakout_current_volume_ratio']}"
        )
        if candidate_features["platform_breakout_score"] >= float(cfg["candidate_score_threshold"]) and candidate_features[
            "platform_breakout_score"
        ] > best_score:
            best_candidate = candidate_features
            best_score = float(candidate_features["platform_breakout_score"])

    if best_candidate is None:
        result["platform_breakout_reason"] = "no_valid_breakout_retest"
        return result

    result.update(best_candidate)
    return result


def build_platform_breakout_snapshot(
    market_daily_history: pd.DataFrame,
    stock_basic_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if market_daily_history is None or market_daily_history.empty or stock_basic_df is None or stock_basic_df.empty:
        return pd.DataFrame()

    cfg = merge_platform_breakout_config(config)
    basic_lookup = {
        str(row["ts_code"]): row
        for row in stock_basic_df.fillna("").to_dict(orient="records")
        if str(row.get("ts_code") or "")
    }

    daily_history = market_daily_history.copy()
    daily_history = daily_history[daily_history["ts_code"].astype(str).isin(basic_lookup.keys())].copy()
    daily_history["trade_date"] = daily_history["trade_date"].astype(str)

    rows: list[dict[str, Any]] = []
    for ts_code, sub in daily_history.groupby("ts_code", sort=False):
        meta = basic_lookup.get(str(ts_code), {"ts_code": ts_code})
        features = calculate_platform_breakout_features(sub, meta=meta, config=cfg)
        if features.get("platform_breakout_signal"):
            rows.append(features)
    return pd.DataFrame(rows)
