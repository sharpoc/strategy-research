from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REAL_BREAKOUT_CONFIG: dict[str, Any] = {
    "pre_runup_lookback_bars": 40,
    "pre_runup_min_pct": 12.0,
    "pre_runup_optimal_low_pct": 16.0,
    "pre_runup_optimal_high_pct": 32.0,
    "platform_min_bars": 5,
    "platform_max_bars": 14,
    "platform_amp_max_pct": 12.0,
    "platform_close_span_max_pct": 7.5,
    "platform_std_max_pct": 3.8,
    "platform_single_bar_max_pct": 6.5,
    "platform_abnormal_bar_max": 1,
    "platform_shrink_vol_ratio_max": 0.82,
    "platform_tail_shrink_ratio_max": 0.95,
    "breakout_lookback_bars": 3,
    "breakout_close_buffer_pct": 1.2,
    "breakout_volume_ratio_min": 1.35,
    "breakout_volume_ratio_max": 4.20,
    "breakout_amount_ratio_min": 1.10,
    "breakout_amount_ratio_max": 3.80,
    "breakout_body_pct_min": 1.4,
    "breakout_upper_shadow_pct_max": 2.0,
    "breakout_close_to_high_pct_max": 1.4,
    "support_hold_break_pct": 1.8,
    "retest_volume_ratio_max": 0.92,
    "retest_breakout_volume_ratio_max": 0.88,
    "follow_volume_ratio_min": 0.95,
    "current_volume_ratio_extreme_max": 4.50,
    "ma20_slope_min_pct": 0.0,
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


def merge_real_breakout_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_REAL_BREAKOUT_CONFIG)
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
    if ts_code.startswith(("688", "689", "8")) or ts_code.endswith(".BJ"):
        return False
    return _board_from_meta(meta) in {"main", "gem"}


def _empty_result(ts_code: str = "") -> dict[str, Any]:
    return {
        "ts_code": ts_code,
        "real_breakout_stage": "",
        "real_breakout_signal": False,
        "real_breakout_score": 0.0,
        "real_breakout_reason": "",
        "real_breakout_board": "",
        "real_breakout_platform_start": None,
        "real_breakout_platform_end": None,
        "real_breakout_breakout_date": None,
        "real_breakout_current_date": None,
        "real_breakout_platform_high": None,
        "real_breakout_platform_low": None,
        "real_breakout_support_price": None,
        "real_breakout_breakout_close": None,
        "real_breakout_breakout_high": None,
        "real_breakout_current_price": None,
        "real_breakout_platform_days": None,
        "real_breakout_platform_amp_pct": None,
        "real_breakout_platform_close_span_pct": None,
        "real_breakout_platform_std_pct": None,
        "real_breakout_platform_vol_ratio": None,
        "real_breakout_platform_tail_vol_ratio": None,
        "real_breakout_pre_runup_pct": None,
        "real_breakout_pre_runup_days": None,
        "real_breakout_breakout_close_buffer_pct": None,
        "real_breakout_breakout_volume_ratio": None,
        "real_breakout_breakout_amount_ratio": None,
        "real_breakout_breakout_body_pct": None,
        "real_breakout_breakout_upper_shadow_pct": None,
        "real_breakout_breakout_close_to_high_pct": None,
        "real_breakout_current_volume_ratio": None,
        "real_breakout_current_amount_ratio": None,
        "real_breakout_current_buffer_pct": None,
        "real_breakout_ma20_slope_pct": None,
        "real_breakout_ma20_buffer_pct": None,
        "real_breakout_ma60_buffer_pct": None,
        "real_breakout_abnormal_bar_count": 0,
        "real_breakout_extreme_volume_flag": False,
        "real_breakout_retest_ok": False,
        "real_breakout_follow_ok": False,
        "real_breakout_breakout_today": False,
    }


def _pct_distance(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return (float(numerator) / float(denominator) - 1.0) * 100.0


def _body_pct(open_price: float, close_price: float, pre_close: float | None) -> float | None:
    if pre_close in (None, 0):
        return None
    return abs(float(close_price) - float(open_price)) / float(pre_close) * 100.0


def _upper_shadow_pct(open_price: float, high_price: float, close_price: float, pre_close: float | None) -> float | None:
    if pre_close in (None, 0):
        return None
    upper_shadow = max(float(high_price) - max(float(open_price), float(close_price)), 0.0)
    return upper_shadow / float(pre_close) * 100.0


def _close_to_high_pct(high_price: float, close_price: float) -> float | None:
    if close_price in (None, 0):
        return None
    return max(float(high_price) - float(close_price), 0.0) / float(close_price) * 100.0


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


def _score_between(value: float | None, ideal_low: float, ideal_high: float, high_score: float, mid_score: float, miss_score: float) -> float:
    if value is None:
        return miss_score
    if ideal_low <= value <= ideal_high:
        return high_score
    return mid_score


def build_real_breakout_score(features: dict[str, Any]) -> float:
    stage = str(features.get("real_breakout_stage") or "")
    pre_runup_pct = to_float(features.get("real_breakout_pre_runup_pct"))
    platform_days = to_float(features.get("real_breakout_platform_days"))
    platform_amp_pct = to_float(features.get("real_breakout_platform_amp_pct"))
    platform_close_span_pct = to_float(features.get("real_breakout_platform_close_span_pct"))
    platform_vol_ratio = to_float(features.get("real_breakout_platform_vol_ratio"))
    platform_tail_vol_ratio = to_float(features.get("real_breakout_platform_tail_vol_ratio"))
    breakout_close_buffer_pct = to_float(features.get("real_breakout_breakout_close_buffer_pct"))
    breakout_volume_ratio = to_float(features.get("real_breakout_breakout_volume_ratio"))
    breakout_amount_ratio = to_float(features.get("real_breakout_breakout_amount_ratio"))
    breakout_body_pct = to_float(features.get("real_breakout_breakout_body_pct"))
    breakout_upper_shadow_pct = to_float(features.get("real_breakout_breakout_upper_shadow_pct"))
    breakout_close_to_high_pct = to_float(features.get("real_breakout_breakout_close_to_high_pct"))
    current_volume_ratio = to_float(features.get("real_breakout_current_volume_ratio"))
    current_buffer_pct = to_float(features.get("real_breakout_current_buffer_pct"))
    ma20_slope_pct = to_float(features.get("real_breakout_ma20_slope_pct"))
    ma20_buffer_pct = to_float(features.get("real_breakout_ma20_buffer_pct"))
    ma60_buffer_pct = to_float(features.get("real_breakout_ma60_buffer_pct"))
    abnormal_bar_count = int(to_float(features.get("real_breakout_abnormal_bar_count")) or 0)

    score = 0.0
    stage_scores = {
        "retest_hold": 12.0,
        "breakout_today": 10.0,
        "follow_through": 8.0,
    }
    score += stage_scores.get(stage, 0.0)

    score += _score_between(pre_runup_pct, 16.0, 32.0, 12.0, 5.0, -8.0)
    if pre_runup_pct is not None and pre_runup_pct > 45.0:
        score -= 4.0

    score += _score_between(platform_days, 6.0, 10.0, 6.0, 3.0, -3.0)
    score += _score_between(platform_amp_pct, 0.0, 8.0, 9.0, 4.0, -6.0)
    score += _score_between(platform_close_span_pct, 0.0, 4.8, 5.0, 2.0, -3.0)
    score += _score_between(platform_vol_ratio, 0.0, 0.72, 8.0, 4.0, -5.0)
    score += _score_between(platform_tail_vol_ratio, 0.0, 0.88, 4.0, 2.0, -3.0)

    score += _score_between(breakout_close_buffer_pct, 1.5, 4.5, 8.0, 4.0, -6.0)
    score += _score_between(breakout_volume_ratio, 1.35, 2.80, 8.0, 4.0, -6.0)
    score += _score_between(breakout_amount_ratio, 1.10, 2.80, 5.0, 2.0, -3.0)
    score += _score_between(breakout_body_pct, 1.8, 7.0, 4.0, 2.0, -4.0)

    if breakout_upper_shadow_pct is not None:
        if breakout_upper_shadow_pct <= 0.8:
            score += 5.0
        elif breakout_upper_shadow_pct <= 1.8:
            score += 2.0
        else:
            score -= 5.0
    if breakout_close_to_high_pct is not None:
        if breakout_close_to_high_pct <= 0.8:
            score += 3.0
        elif breakout_close_to_high_pct <= 1.4:
            score += 1.0
        else:
            score -= 3.0

    if current_buffer_pct is not None:
        if 0.6 <= current_buffer_pct <= 7.0:
            score += 5.0
        elif current_buffer_pct < 0:
            score -= 8.0
    if current_volume_ratio is not None:
        if 1.0 <= current_volume_ratio <= 2.8:
            score += 4.0
        elif current_volume_ratio > 4.5:
            score -= 5.0
    if ma20_slope_pct is not None:
        if ma20_slope_pct > 0.10:
            score += 4.0
        elif ma20_slope_pct >= 0.0:
            score += 2.0
        else:
            score -= 4.0
    if ma20_buffer_pct is not None:
        if ma20_buffer_pct >= 0.8:
            score += 4.0
        elif ma20_buffer_pct < 0:
            score -= 6.0
    if ma60_buffer_pct is not None:
        if ma60_buffer_pct >= 1.5:
            score += 2.0
        elif ma60_buffer_pct < 0:
            score -= 3.0

    if abnormal_bar_count == 0:
        score += 5.0
    elif abnormal_bar_count == 1:
        score += 1.0
    else:
        score -= 6.0

    if bool(features.get("real_breakout_extreme_volume_flag")):
        score -= 5.0

    return round(clip_score(score, 0.0, 100.0), 2)


def calculate_real_breakout_features(
    daily_history: pd.DataFrame,
    meta: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    ts_code = str(meta.get("ts_code") or "")
    result = _empty_result(ts_code)
    if daily_history is None or daily_history.empty or not _is_allowed_stock(meta):
        result["real_breakout_reason"] = "invalid_universe"
        return result

    sub = daily_history.copy().sort_values("trade_date").reset_index(drop=True)
    if len(sub) < max(int(config["platform_max_bars"]) + int(config["pre_runup_lookback_bars"]), 70):
        result["real_breakout_reason"] = "insufficient_history"
        return result

    trade_dates = sub["trade_date"].astype(str).to_numpy()
    open_arr = pd.to_numeric(sub["open"], errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(sub["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(sub["low"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(sub["close"], errors="coerce").to_numpy(dtype=float)
    pre_close_arr = pd.to_numeric(sub.get("pre_close"), errors="coerce").to_numpy(dtype=float)
    vol_arr = pd.to_numeric(sub.get("vol"), errors="coerce").to_numpy(dtype=float)
    amount_arr = pd.to_numeric(sub.get("amount"), errors="coerce").to_numpy(dtype=float)
    pct_arr = np.where(pre_close_arr > 0, (close_arr / pre_close_arr - 1.0) * 100.0, np.nan)

    ma5_arr = _rolling_mean(close_arr, 5, min_periods=3)
    ma10_arr = _rolling_mean(close_arr, 10, min_periods=5)
    ma20_arr = _rolling_mean(close_arr, 20, min_periods=10)
    ma60_arr = _rolling_mean(close_arr, 60, min_periods=30)
    vol_ma5_arr = _rolling_mean(vol_arr, 5, min_periods=3)
    amount_ma20_arr = _rolling_mean(amount_arr, 20, min_periods=10)

    current_idx = len(sub) - 1
    board = _board_from_meta(meta)
    result["real_breakout_board"] = board
    result["real_breakout_current_date"] = trade_dates[current_idx]

    best_candidate: dict[str, Any] | None = None
    best_score = -1e9
    breakout_lookback = int(config["breakout_lookback_bars"])
    platform_min = int(config["platform_min_bars"])
    platform_max = int(config["platform_max_bars"])
    pre_runup_lookback = int(config["pre_runup_lookback_bars"])

    for breakout_idx in range(max(platform_min + 20, current_idx - breakout_lookback), current_idx + 1):
        current_stage = ""
        if breakout_idx == current_idx:
            current_stage = "breakout_today"

        breakout_close = close_arr[breakout_idx]
        breakout_high = high_arr[breakout_idx]
        breakout_open = open_arr[breakout_idx]
        breakout_pre_close = pre_close_arr[breakout_idx]
        breakout_body_pct = _body_pct(breakout_open, breakout_close, breakout_pre_close)
        breakout_upper_shadow_pct = _upper_shadow_pct(breakout_open, breakout_high, breakout_close, breakout_pre_close)
        breakout_close_to_high_pct = _close_to_high_pct(breakout_high, breakout_close)
        breakout_vol_ma_ref = vol_ma5_arr[breakout_idx]
        breakout_volume_ratio = (
            float(vol_arr[breakout_idx] / breakout_vol_ma_ref)
            if breakout_vol_ma_ref not in (None, 0) and not np.isnan(breakout_vol_ma_ref)
            else None
        )
        breakout_amount_ratio = (
            float(amount_arr[breakout_idx] / amount_ma20_arr[breakout_idx])
            if amount_ma20_arr[breakout_idx] not in (None, 0) and not np.isnan(amount_ma20_arr[breakout_idx])
            else None
        )
        if breakout_body_pct is None or breakout_body_pct < float(config["breakout_body_pct_min"]):
            continue
        if breakout_upper_shadow_pct is not None and breakout_upper_shadow_pct > float(config["breakout_upper_shadow_pct_max"]):
            continue
        if breakout_close_to_high_pct is not None and breakout_close_to_high_pct > float(config["breakout_close_to_high_pct_max"]):
            continue
        if breakout_volume_ratio is None or breakout_volume_ratio < float(config["breakout_volume_ratio_min"]):
            continue
        if breakout_volume_ratio > float(config["breakout_volume_ratio_max"]):
            continue
        if breakout_amount_ratio is None or breakout_amount_ratio < float(config["breakout_amount_ratio_min"]):
            continue
        if breakout_amount_ratio > float(config["breakout_amount_ratio_max"]):
            continue

        for platform_days in range(platform_min, platform_max + 1):
            start_idx = breakout_idx - platform_days
            if start_idx < 20:
                continue

            platform_slice = slice(start_idx, breakout_idx)
            platform_high = float(np.nanmax(high_arr[platform_slice]))
            platform_low = float(np.nanmin(low_arr[platform_slice]))
            if platform_high <= 0 or platform_low <= 0 or platform_high <= platform_low:
                continue

            breakout_close_buffer_pct = _pct_distance(breakout_close, platform_high)
            if breakout_close_buffer_pct is None or breakout_close_buffer_pct < float(config["breakout_close_buffer_pct"]):
                continue
            if breakout_high <= platform_high:
                continue

            platform_close_arr = close_arr[platform_slice]
            platform_vol_arr = vol_arr[platform_slice]
            platform_pct_arr = pct_arr[platform_slice]
            platform_amp_pct = (platform_high / platform_low - 1.0) * 100.0
            platform_close_span_pct = (
                (np.nanmax(platform_close_arr) / np.nanmin(platform_close_arr) - 1.0) * 100.0
                if np.nanmin(platform_close_arr) > 0
                else None
            )
            platform_std_pct = float(np.nanstd(platform_pct_arr)) if np.isfinite(platform_pct_arr).any() else None
            abnormal_bar_count = int(np.nansum(np.abs(platform_pct_arr) >= float(config["platform_single_bar_max_pct"])))
            if platform_amp_pct > float(config["platform_amp_max_pct"]):
                continue
            if platform_close_span_pct is None or platform_close_span_pct > float(config["platform_close_span_max_pct"]):
                continue
            if platform_std_pct is None or platform_std_pct > float(config["platform_std_max_pct"]):
                continue
            if abnormal_bar_count > int(config["platform_abnormal_bar_max"]):
                continue

            pretrend_start_idx = max(0, start_idx - pre_runup_lookback)
            pretrend_slice = slice(pretrend_start_idx, start_idx)
            pretrend_low_idx_local = int(np.nanargmin(low_arr[pretrend_slice]))
            pretrend_low_idx = pretrend_start_idx + pretrend_low_idx_local
            pretrend_low = float(low_arr[pretrend_low_idx])
            if pretrend_low <= 0:
                continue
            pre_runup_pct = (platform_high / pretrend_low - 1.0) * 100.0
            if pre_runup_pct < float(config["pre_runup_min_pct"]):
                continue
            pre_runup_days = start_idx - pretrend_low_idx

            pretrend_vol_arr = vol_arr[pretrend_slice]
            if len(pretrend_vol_arr) < 3:
                continue
            pretrend_avg_vol = float(np.nanmean(pretrend_vol_arr))
            platform_avg_vol = float(np.nanmean(platform_vol_arr))
            if pretrend_avg_vol <= 0 or platform_avg_vol <= 0:
                continue
            platform_vol_ratio = platform_avg_vol / pretrend_avg_vol
            platform_tail_days = max(2, min(4, platform_days // 2 + 1))
            platform_tail_avg_vol = float(np.nanmean(platform_vol_arr[-platform_tail_days:]))
            platform_tail_vol_ratio = platform_tail_avg_vol / platform_avg_vol if platform_avg_vol > 0 else None
            if platform_vol_ratio > float(config["platform_shrink_vol_ratio_max"]):
                continue
            if platform_tail_vol_ratio is None or platform_tail_vol_ratio > float(config["platform_tail_shrink_ratio_max"]):
                continue

            ma20_buffer_pct = _pct_distance(close_arr[current_idx], ma20_arr[current_idx])
            ma60_buffer_pct = _pct_distance(close_arr[current_idx], ma60_arr[current_idx])
            ma20_slope_pct = _pct_distance(ma20_arr[current_idx], ma20_arr[current_idx - 3]) if current_idx >= 3 else None
            if ma20_slope_pct is None or ma20_slope_pct < float(config["ma20_slope_min_pct"]):
                continue
            if ma20_buffer_pct is None or ma20_buffer_pct < 0:
                continue

            support_price = platform_high * (1.0 - float(config["support_hold_break_pct"]) / 100.0)
            current_volume_ratio = (
                float(vol_arr[current_idx] / vol_ma5_arr[current_idx])
                if vol_ma5_arr[current_idx] not in (None, 0) and not np.isnan(vol_ma5_arr[current_idx])
                else None
            )
            current_amount_ratio = (
                float(amount_arr[current_idx] / amount_ma20_arr[current_idx])
                if amount_ma20_arr[current_idx] not in (None, 0) and not np.isnan(amount_ma20_arr[current_idx])
                else None
            )
            current_buffer_pct = _pct_distance(close_arr[current_idx], platform_high)
            extreme_volume_flag = bool(
                (breakout_volume_ratio is not None and breakout_volume_ratio > float(config["current_volume_ratio_extreme_max"]))
                or (current_volume_ratio is not None and current_volume_ratio > float(config["current_volume_ratio_extreme_max"]))
            )

            retest_ok = False
            follow_ok = False
            if breakout_idx < current_idx:
                post_low = float(np.nanmin(low_arr[breakout_idx + 1 : current_idx + 1]))
                if post_low < support_price:
                    continue
                if close_arr[current_idx] >= platform_high and current_buffer_pct is not None and current_buffer_pct >= 0:
                    if current_volume_ratio is not None and breakout_volume_ratio is not None:
                        if current_volume_ratio <= max(float(config["retest_volume_ratio_max"]), breakout_volume_ratio * float(config["retest_breakout_volume_ratio_max"])):
                            retest_ok = bool(close_arr[current_idx] >= open_arr[current_idx])
                    if not retest_ok and close_arr[current_idx] > breakout_close and current_volume_ratio is not None:
                        follow_ok = current_volume_ratio >= float(config["follow_volume_ratio_min"])
                if not retest_ok and not follow_ok:
                    continue
                current_stage = "retest_hold" if retest_ok else "follow_through"

            candidate = {
                "ts_code": ts_code,
                "real_breakout_stage": current_stage,
                "real_breakout_signal": True,
                "real_breakout_board": board,
                "real_breakout_platform_start": trade_dates[start_idx],
                "real_breakout_platform_end": trade_dates[breakout_idx - 1],
                "real_breakout_breakout_date": trade_dates[breakout_idx],
                "real_breakout_current_date": trade_dates[current_idx],
                "real_breakout_platform_high": to_number(platform_high, 4),
                "real_breakout_platform_low": to_number(platform_low, 4),
                "real_breakout_support_price": to_number(platform_high, 4),
                "real_breakout_breakout_close": to_number(breakout_close, 4),
                "real_breakout_breakout_high": to_number(breakout_high, 4),
                "real_breakout_current_price": to_number(close_arr[current_idx], 4),
                "real_breakout_platform_days": int(platform_days),
                "real_breakout_platform_amp_pct": to_number(platform_amp_pct, 4),
                "real_breakout_platform_close_span_pct": to_number(platform_close_span_pct, 4),
                "real_breakout_platform_std_pct": to_number(platform_std_pct, 4),
                "real_breakout_platform_vol_ratio": to_number(platform_vol_ratio, 4),
                "real_breakout_platform_tail_vol_ratio": to_number(platform_tail_vol_ratio, 4),
                "real_breakout_pre_runup_pct": to_number(pre_runup_pct, 4),
                "real_breakout_pre_runup_days": int(pre_runup_days),
                "real_breakout_breakout_close_buffer_pct": to_number(breakout_close_buffer_pct, 4),
                "real_breakout_breakout_volume_ratio": to_number(breakout_volume_ratio, 4),
                "real_breakout_breakout_amount_ratio": to_number(breakout_amount_ratio, 4),
                "real_breakout_breakout_body_pct": to_number(breakout_body_pct, 4),
                "real_breakout_breakout_upper_shadow_pct": to_number(breakout_upper_shadow_pct, 4),
                "real_breakout_breakout_close_to_high_pct": to_number(breakout_close_to_high_pct, 4),
                "real_breakout_current_volume_ratio": to_number(current_volume_ratio, 4),
                "real_breakout_current_amount_ratio": to_number(current_amount_ratio, 4),
                "real_breakout_current_buffer_pct": to_number(current_buffer_pct, 4),
                "real_breakout_ma20_slope_pct": to_number(ma20_slope_pct, 4),
                "real_breakout_ma20_buffer_pct": to_number(ma20_buffer_pct, 4),
                "real_breakout_ma60_buffer_pct": to_number(ma60_buffer_pct, 4),
                "real_breakout_abnormal_bar_count": int(abnormal_bar_count),
                "real_breakout_extreme_volume_flag": extreme_volume_flag,
                "real_breakout_retest_ok": bool(retest_ok),
                "real_breakout_follow_ok": bool(follow_ok),
                "real_breakout_breakout_today": bool(breakout_idx == current_idx),
            }
            candidate["real_breakout_score"] = build_real_breakout_score(candidate)
            candidate["real_breakout_reason"] = (
                f"{candidate['real_breakout_stage']} 平台{candidate['real_breakout_platform_days']}天"
                f" 缩量比{candidate['real_breakout_platform_vol_ratio']}"
                f" 突破量比{candidate['real_breakout_breakout_volume_ratio']}"
                f" 现价偏离{candidate['real_breakout_current_buffer_pct']}%"
            )
            if candidate["real_breakout_score"] >= float(config["candidate_score_threshold"]) and candidate["real_breakout_score"] > best_score:
                best_candidate = candidate
                best_score = float(candidate["real_breakout_score"])

    if best_candidate is None:
        result["real_breakout_reason"] = "no_valid_real_breakout"
        return result

    result.update(best_candidate)
    return result


def build_real_breakout_snapshot(
    market_daily_history: pd.DataFrame,
    stock_basic_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if market_daily_history is None or market_daily_history.empty or stock_basic_df is None or stock_basic_df.empty:
        return pd.DataFrame()

    cfg = merge_real_breakout_config(config)
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
        features = calculate_real_breakout_features(sub, meta=meta, config=cfg)
        if features.get("real_breakout_signal"):
            rows.append(features)
    return pd.DataFrame(rows)
