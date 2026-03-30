from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


DEFAULT_DOUBLE_BOTTOM_CONFIG: dict[str, Any] = {
    "include_star": False,
    "min_listed_bars": 120,
    "min_price": 3.0,
    "min_avg_amount_20d": 100000.0,
    "pivot_left": 4,
    "pivot_right": 4,
    "pre_down_lookback_min": 20,
    "pre_down_lookback_max": 60,
    "min_pre_down_pct": 15.0,
    "preferred_pre_down_pct": 20.0,
    "min_rebound_pct": 8.0,
    "preferred_rebound_pct": 12.0,
    "min_bars_between_bottoms": 8,
    "max_bars_between_bottoms": 40,
    "preferred_spacing_min": 12,
    "preferred_spacing_max": 30,
    "max_l2_deviation_pct": 3.0,
    "preferred_l2_higher_pct": 0.0,
    "pullback_volume_ratio_max": 0.85,
    "preferred_pullback_volume_ratio": 0.70,
    "pullback_big_drop_main_pct": -5.0,
    "pullback_big_drop_gem_pct": -8.0,
    "pullback_big_drop_volume_mult": 1.15,
    "neckline_breakout_volume_mult": 1.30,
    "strong_breakout_volume_mult": 1.50,
    "retest_max_break_pct": 2.0,
    "right_start_max_bars_after_l2": 12,
    "pattern_stale_max_bars_after_l2": 25,
    "current_close_ma20_tolerance_pct": -1.5,
    "candidate_score_threshold": 62.0,
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


def merge_double_bottom_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_DOUBLE_BOTTOM_CONFIG)
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


def _is_allowed_stock(meta: dict[str, Any], config: dict[str, Any]) -> bool:
    name = str(meta.get("name") or "").upper()
    ts_code = str(meta.get("ts_code") or "").upper()
    market = str(meta.get("market") or "").strip()
    if "ST" in name or "退" in name:
        return False
    if market == "北交所" or ts_code.startswith(("8", "4")):
        return False
    if market == "科创板" or ts_code.startswith(("688", "689")):
        return bool(config.get("include_star"))
    return _board_from_meta(meta, include_star=bool(config.get("include_star"))) in {"main", "gem", "star"}


def _empty_result(ts_code: str = "") -> dict[str, Any]:
    return {
        "ts_code": ts_code,
        "double_bottom_signal": False,
        "double_bottom_score": 0.0,
        "double_bottom_reason": "",
        "double_bottom_board": "",
        "double_bottom_buy_type": "",
        "double_bottom_breakout_status": "",
        "double_bottom_l1_date": None,
        "double_bottom_l1_price": None,
        "double_bottom_h_date": None,
        "double_bottom_h_price": None,
        "double_bottom_l2_date": None,
        "double_bottom_l2_price": None,
        "double_bottom_neckline": None,
        "double_bottom_current_price": None,
        "double_bottom_pre_down_pct": None,
        "double_bottom_rebound_pct": None,
        "double_bottom_l2_vs_l1_pct": None,
        "double_bottom_spacing_bars": None,
        "double_bottom_pullback_volume_ratio": None,
        "double_bottom_pullback_big_drop_count": 0,
        "double_bottom_breakout_volume_ratio": None,
        "double_bottom_breakout_close_buffer_pct": None,
        "double_bottom_breakout_upper_shadow_pct": None,
        "double_bottom_retest_neckline_buffer_pct": None,
        "double_bottom_current_vs_ma20_pct": None,
        "double_bottom_ma5_above_ma10": False,
        "double_bottom_ma10_slope_pct": None,
        "double_bottom_ma20_slope_pct": None,
        "double_bottom_position_120": None,
        "double_bottom_space_to_120_high_pct": None,
        "double_bottom_bars_since_l2": None,
        "double_bottom_stop_signal": False,
        "double_bottom_core_reasons": [],
        "double_bottom_risks": [],
    }


def _percent_distance(value: float | None, anchor: float | None) -> float | None:
    if value is None or anchor in (None, 0):
        return None
    try:
        if pd.isna(value) or pd.isna(anchor):
            return None
    except Exception:
        pass
    return (float(value) / float(anchor) - 1.0) * 100.0


def _range_pct(high_price: float, low_price: float, pre_close: float | None) -> float | None:
    if pre_close in (None, 0):
        return None
    return (float(high_price) - float(low_price)) / float(pre_close) * 100.0


def _upper_shadow_pct(open_price: float, high_price: float, close_price: float, pre_close: float | None) -> float | None:
    if pre_close in (None, 0):
        return None
    upper_shadow = max(float(high_price) - max(float(open_price), float(close_price)), 0.0)
    return upper_shadow / float(pre_close) * 100.0


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
        events[pivot_bar + right] = {"pivot_bar": int(pivot_bar), "price": float(values[pivot_bar])}
    return events


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
    body_ratio = abs(close_price - open_price) / range_value
    lower_shadow = min(open_price, close_price) - low_price
    lower_shadow_ratio = max(lower_shadow, 0.0) / range_value
    vol_ma5 = vol_ma5_arr[idx]
    range_pct = _range_pct(high_price, low_price, pre_close_arr[idx])
    bullish = close_price > open_price
    long_lower_shadow = lower_shadow_ratio >= 0.35 and body_ratio <= 0.55
    shrink_small_body = body_ratio <= 0.30 and (np.isnan(vol_ma5) or volume <= vol_ma5)
    narrow_range = range_pct is not None and range_pct <= 3.5
    return bool(bullish or long_lower_shadow or (shrink_small_body and narrow_range))


def _current_strength_flags(
    idx: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    close_arr: np.ndarray,
    ma5_arr: np.ndarray,
    ma10_arr: np.ndarray,
) -> dict[str, bool]:
    if idx <= 0 or idx >= len(close_arr):
        return {"reclaim_ma5": False, "reclaim_ma10": False, "break_prev_high": False, "engulf": False}
    current_open = float(open_arr[idx])
    current_close = float(close_arr[idx])
    prev_open = float(open_arr[idx - 1])
    prev_close = float(close_arr[idx - 1])
    prev_high = float(high_arr[idx - 1])
    ma5 = ma5_arr[idx]
    ma10 = ma10_arr[idx]
    return {
        "reclaim_ma5": bool(not np.isnan(ma5) and current_close > ma5),
        "reclaim_ma10": bool(not np.isnan(ma10) and current_close > ma10),
        "break_prev_high": bool(current_close > prev_high),
        "engulf": bool(
            current_close > current_open
            and current_close >= max(prev_open, prev_close)
            and current_open <= min(prev_open, prev_close)
        ),
    }


def _evaluate_current_setup(
    current_idx: int,
    l2_bar: int,
    neckline: float,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    vol_ma5_arr: np.ndarray,
    ma5_arr: np.ndarray,
    ma10_arr: np.ndarray,
    pre_close_arr: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    vol_ma5 = vol_ma5_arr[current_idx]
    current_volume_ratio = float(volume_arr[current_idx] / vol_ma5) if not np.isnan(vol_ma5) and vol_ma5 > 0 else None
    current_upper_shadow_pct = _upper_shadow_pct(
        open_arr[current_idx],
        high_arr[current_idx],
        close_arr[current_idx],
        pre_close_arr[current_idx],
    )
    current_close_buffer_pct = _percent_distance(close_arr[current_idx], neckline)
    current_low_buffer_pct = _percent_distance(low_arr[current_idx], neckline)
    strength_flags = _current_strength_flags(current_idx, open_arr, high_arr, close_arr, ma5_arr, ma10_arr)
    stop_now = _is_stop_signal(
        current_idx,
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
        vol_ma5_arr,
        pre_close_arr,
    )

    buy_type = ""
    breakout_status = ""
    breakout_idx: int | None = None
    breakout_volume_ratio = None
    breakout_close_buffer_pct = None
    retest_buffer_pct = None

    upper_shadow_ok = current_upper_shadow_pct is None or current_upper_shadow_pct <= 2.5
    breakout_today = bool(
        high_arr[current_idx] > neckline
        and close_arr[current_idx] >= neckline
        and current_volume_ratio is not None
        and current_volume_ratio >= float(config["neckline_breakout_volume_mult"])
        and upper_shadow_ok
    )

    if breakout_today:
        buy_type = "A"
        breakout_status = "neckline_breakout"
        breakout_idx = current_idx
        breakout_volume_ratio = current_volume_ratio
        breakout_close_buffer_pct = current_close_buffer_pct
    else:
        recent_breakout_idx: int | None = None
        for idx in range(max(l2_bar + 1, current_idx - 5), current_idx):
            vol_ma = vol_ma5_arr[idx]
            volume_ratio = float(volume_arr[idx] / vol_ma) if not np.isnan(vol_ma) and vol_ma > 0 else None
            close_buffer_pct = _percent_distance(close_arr[idx], neckline)
            upper_shadow_pct = _upper_shadow_pct(open_arr[idx], high_arr[idx], close_arr[idx], pre_close_arr[idx])
            if (
                high_arr[idx] > neckline
                and close_arr[idx] >= neckline
                and volume_ratio is not None
                and volume_ratio >= float(config["neckline_breakout_volume_mult"])
                and (upper_shadow_pct is None or upper_shadow_pct <= 2.8)
                and (close_buffer_pct is not None and close_buffer_pct >= -0.2)
            ):
                recent_breakout_idx = idx
        if recent_breakout_idx is not None:
            retest_buffer_pct = current_low_buffer_pct
            retest_volume_ok = current_volume_ratio is not None and current_volume_ratio >= 0.75
            retest_not_broken = retest_buffer_pct is not None and retest_buffer_pct >= -float(config["retest_max_break_pct"])
            retest_reclaim = current_close_buffer_pct is not None and current_close_buffer_pct >= -0.3
            retest_strength = stop_now or strength_flags["break_prev_high"] or strength_flags["engulf"] or strength_flags["reclaim_ma5"]
            if retest_not_broken and retest_reclaim and retest_strength and retest_volume_ok:
                buy_type = "B"
                breakout_status = "breakout_retest_hold"
                breakout_idx = recent_breakout_idx
                vol_ma = vol_ma5_arr[recent_breakout_idx]
                breakout_volume_ratio = (
                    float(volume_arr[recent_breakout_idx] / vol_ma) if not np.isnan(vol_ma) and vol_ma > 0 else None
                )
                breakout_close_buffer_pct = _percent_distance(close_arr[recent_breakout_idx], neckline)

    if not buy_type:
        bars_since_l2 = current_idx - l2_bar
        right_side_ok = (
            bars_since_l2 <= int(config["right_start_max_bars_after_l2"])
            and close_arr[current_idx] < neckline * 1.02
            and close_arr[current_idx] > max(ma5_arr[current_idx], ma10_arr[current_idx])
            and (strength_flags["reclaim_ma5"] or strength_flags["break_prev_high"] or close_arr[current_idx] > close_arr[current_idx - 1])
        )
        if right_side_ok and current_volume_ratio is not None and current_volume_ratio >= 0.95:
            buy_type = "C"
            breakout_status = "right_side_start"
            breakout_volume_ratio = current_volume_ratio
            breakout_close_buffer_pct = current_close_buffer_pct

    return {
        "buy_type": buy_type,
        "breakout_status": breakout_status,
        "breakout_idx": breakout_idx,
        "breakout_volume_ratio": breakout_volume_ratio,
        "breakout_close_buffer_pct": breakout_close_buffer_pct,
        "breakout_upper_shadow_pct": current_upper_shadow_pct,
        "retest_neckline_buffer_pct": retest_buffer_pct,
        "strength_flags": strength_flags,
        "stop_now": stop_now,
        "current_volume_ratio": current_volume_ratio,
        "current_close_buffer_pct": current_close_buffer_pct,
    }


def build_double_bottom_score(features: dict[str, Any]) -> float:
    score = 0.0
    pre_down_pct = to_float(features.get("double_bottom_pre_down_pct"))
    rebound_pct = to_float(features.get("double_bottom_rebound_pct"))
    l2_vs_l1_pct = to_float(features.get("double_bottom_l2_vs_l1_pct"))
    spacing_bars = to_float(features.get("double_bottom_spacing_bars"))
    pullback_volume_ratio = to_float(features.get("double_bottom_pullback_volume_ratio"))
    breakout_volume_ratio = to_float(features.get("double_bottom_breakout_volume_ratio"))
    breakout_close_buffer_pct = to_float(features.get("double_bottom_breakout_close_buffer_pct"))
    breakout_upper_shadow_pct = to_float(features.get("double_bottom_breakout_upper_shadow_pct"))
    ma10_slope_pct = to_float(features.get("double_bottom_ma10_slope_pct"))
    ma20_slope_pct = to_float(features.get("double_bottom_ma20_slope_pct"))
    current_vs_ma20_pct = to_float(features.get("double_bottom_current_vs_ma20_pct"))
    position_120 = to_float(features.get("double_bottom_position_120"))
    space_to_120_high_pct = to_float(features.get("double_bottom_space_to_120_high_pct"))
    bars_since_l2 = to_float(features.get("double_bottom_bars_since_l2"))
    big_drop_count = int(to_float(features.get("double_bottom_pullback_big_drop_count")) or 0)
    buy_type = str(features.get("double_bottom_buy_type") or "")

    if pre_down_pct is not None:
        if pre_down_pct >= 20:
            score += 8.0
        elif pre_down_pct >= 15:
            score += 5.0

    if spacing_bars is not None:
        if 12 <= spacing_bars <= 30:
            score += 8.0
        elif 8 <= spacing_bars <= 40:
            score += 5.0

    if l2_vs_l1_pct is not None:
        if l2_vs_l1_pct >= 0:
            score += 14.0
        elif abs(l2_vs_l1_pct) <= 2:
            score += 10.0
        elif abs(l2_vs_l1_pct) <= 3:
            score += 5.0

    if rebound_pct is not None:
        if rebound_pct >= 15:
            score += 15.0
        elif rebound_pct >= 10:
            score += 10.0
        elif rebound_pct >= 8:
            score += 6.0

    if pullback_volume_ratio is not None:
        if pullback_volume_ratio <= 0.60:
            score += 10.0
        elif pullback_volume_ratio <= 0.70:
            score += 8.0
        elif pullback_volume_ratio <= 0.85:
            score += 5.0
    if bool(features.get("double_bottom_stop_signal")):
        score += 10.0

    if buy_type == "B":
        score += 10.0
    elif buy_type == "A":
        score += 8.0
    elif buy_type == "C":
        score += 4.0

    if breakout_volume_ratio is not None:
        if breakout_volume_ratio >= 1.5:
            score += 10.0
        elif breakout_volume_ratio >= 1.3:
            score += 8.0
        elif breakout_volume_ratio >= 1.0:
            score += 4.0

    if breakout_close_buffer_pct is not None:
        if breakout_close_buffer_pct >= 2.0:
            score += 4.0
        elif breakout_close_buffer_pct >= 0.0:
            score += 2.0

    if bool(features.get("double_bottom_ma5_above_ma10")):
        score += 3.0
    if ma10_slope_pct is not None and ma10_slope_pct >= 0:
        score += 3.0
    if current_vs_ma20_pct is not None and current_vs_ma20_pct >= 0:
        score += 4.0

    if position_120 is not None and space_to_120_high_pct is not None:
        if position_120 <= 0.62 and space_to_120_high_pct >= 15:
            score += 5.0
        elif position_120 <= 0.78 and space_to_120_high_pct >= 8:
            score += 3.0

    penalty = 0.0
    penalty += min(big_drop_count, 2) * 5.0
    if breakout_upper_shadow_pct is not None:
        if breakout_upper_shadow_pct > 3.0:
            penalty += 4.0
        elif breakout_upper_shadow_pct > 2.0:
            penalty += 2.0
    if ma20_slope_pct is not None and ma20_slope_pct < 0:
        penalty += 3.0
    if current_vs_ma20_pct is not None and current_vs_ma20_pct < -1.5:
        penalty += 4.0
    if buy_type == "C":
        penalty += 2.0
    if space_to_120_high_pct is not None and space_to_120_high_pct < 8:
        penalty += 4.0
    if bars_since_l2 is not None and bars_since_l2 > 18:
        penalty += 3.0

    return round(clip_score(score - penalty, 0.0, 100.0), 2)


def _build_core_reasons(features: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    l2_vs_l1_pct = to_float(features.get("double_bottom_l2_vs_l1_pct"))
    pullback_volume_ratio = to_float(features.get("double_bottom_pullback_volume_ratio"))
    breakout_status = str(features.get("double_bottom_breakout_status") or "")
    breakout_volume_ratio = to_float(features.get("double_bottom_breakout_volume_ratio"))
    current_vs_ma20_pct = to_float(features.get("double_bottom_current_vs_ma20_pct"))
    space_to_120_high_pct = to_float(features.get("double_bottom_space_to_120_high_pct"))

    if l2_vs_l1_pct is not None:
        if l2_vs_l1_pct >= 0:
            reasons.append("L2 不低于 L1，双底结构更标准")
        elif abs(l2_vs_l1_pct) <= 2:
            reasons.append("L2 与 L1 偏差较小，双底结构保持完整")
    if pullback_volume_ratio is not None and pullback_volume_ratio <= 0.85:
        reasons.append("H 到 L2 回踩阶段缩量，抛压相对有限")
    if breakout_status == "breakout_retest_hold":
        reasons.append("前面已有效突破颈线，当前回踩确认后仍守住关键位")
    elif breakout_status == "neckline_breakout":
        reasons.append("当前放量突破颈线，右侧确认度较高")
    elif breakout_status == "right_side_start":
        reasons.append("L2 后右侧开始启动，短线趋势正在修复")
    if breakout_volume_ratio is not None and breakout_volume_ratio >= 1.3:
        reasons.append("关键转强时量能同步回升，资金响应较明显")
    if current_vs_ma20_pct is not None and current_vs_ma20_pct >= 0:
        reasons.append("收盘重新站上 MA20，中期趋势修复更顺畅")
    if space_to_120_high_pct is not None and space_to_120_high_pct >= 15:
        reasons.append("距离 120 日高点仍有空间，上方压力相对更轻")
    return reasons[:5]


def _build_risks(features: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    buy_type = str(features.get("double_bottom_buy_type") or "")
    breakout_volume_ratio = to_float(features.get("double_bottom_breakout_volume_ratio"))
    ma20_slope_pct = to_float(features.get("double_bottom_ma20_slope_pct"))
    l2_vs_l1_pct = to_float(features.get("double_bottom_l2_vs_l1_pct"))
    space_to_120_high_pct = to_float(features.get("double_bottom_space_to_120_high_pct"))
    breakout_upper_shadow_pct = to_float(features.get("double_bottom_breakout_upper_shadow_pct"))

    if buy_type == "A":
        risks.append("刚突破颈线，仍需防假突破")
    elif buy_type == "B":
        risks.append("回踩确认虽更稳，但若再度跌回颈线下方会削弱结构")
    elif buy_type == "C":
        risks.append("当前属于右侧启动，尚未完成正式颈线突破")

    if breakout_volume_ratio is not None and breakout_volume_ratio < 1.5:
        risks.append("量能有恢复但未达到强突破级别")
    if ma20_slope_pct is not None and ma20_slope_pct <= 0:
        risks.append("MA20 尚未明显拐头，中期反转仍需继续确认")
    if l2_vs_l1_pct is not None and l2_vs_l1_pct < 0:
        risks.append("L2 略低于 L1，结构强度不如标准抬高双底")
    if space_to_120_high_pct is not None and space_to_120_high_pct < 12:
        risks.append("上方接近近 120 日高点，压力位较近")
    if breakout_upper_shadow_pct is not None and breakout_upper_shadow_pct > 2.0:
        risks.append("上影线偏长，说明颈线附近仍有抛压")

    if not risks:
        risks = ["短线已进入右侧交易区间，仍需结合次日承接确认", "若后续量能不能持续，形态强度会下降"]
    return risks[:4]


def calculate_double_bottom_features(
    daily_df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = meta or {}
    cfg = merge_double_bottom_config(config)
    ts_code = str(meta.get("ts_code") or daily_df.get("ts_code", pd.Series([""])).iloc[-1] or "")
    result = _empty_result(ts_code)

    if daily_df is None or daily_df.empty:
        result["double_bottom_reason"] = "missing_history"
        return result
    if not _is_allowed_stock(meta, cfg):
        result["double_bottom_reason"] = "universe_excluded"
        return result

    work = daily_df.copy().sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    required_cols = {"trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"}
    if not required_cols.issubset(set(work.columns)):
        result["double_bottom_reason"] = "missing_required_columns"
        return result

    for column in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["open", "high", "low", "close", "pre_close", "vol", "amount"]).reset_index(drop=True)
    if len(work) < int(cfg["min_listed_bars"]):
        result["double_bottom_reason"] = "listed_days_too_short"
        return result

    board = _board_from_meta(meta, include_star=bool(cfg["include_star"]))
    if board not in {"main", "gem", "star"}:
        result["double_bottom_reason"] = "board_excluded"
        return result

    open_arr = work["open"].to_numpy(dtype=float)
    high_arr = work["high"].to_numpy(dtype=float)
    low_arr = work["low"].to_numpy(dtype=float)
    close_arr = work["close"].to_numpy(dtype=float)
    pre_close_arr = work["pre_close"].to_numpy(dtype=float)
    volume_arr = work["vol"].to_numpy(dtype=float)
    amount_arr = work["amount"].to_numpy(dtype=float)
    trade_dates = work["trade_date"].astype(str).tolist()
    current_idx = len(work) - 1

    if close_arr[current_idx] < float(cfg["min_price"]):
        result["double_bottom_reason"] = "price_too_low"
        return result

    avg_amount_20d = float(np.nanmean(amount_arr[max(0, current_idx - 19) : current_idx + 1]))
    if avg_amount_20d < float(cfg["min_avg_amount_20d"]):
        result["double_bottom_reason"] = "liquidity_too_low"
        return result

    ma5_arr = _rolling_mean(close_arr, 5)
    ma10_arr = _rolling_mean(close_arr, 10)
    ma20_arr = _rolling_mean(close_arr, 20)
    vol_ma5_arr = _rolling_mean(volume_arr, 5)
    pct_change = np.where(pre_close_arr > 0, (close_arr / pre_close_arr - 1.0) * 100.0, np.nan)

    if any(np.isnan(arr[current_idx]) for arr in [ma5_arr, ma10_arr, ma20_arr]):
        result["double_bottom_reason"] = "ma_not_ready"
        return result

    low_events = _confirmed_pivot_events(low_arr, int(cfg["pivot_left"]), int(cfg["pivot_right"]), "low")
    high_events = _confirmed_pivot_events(high_arr, int(cfg["pivot_left"]), int(cfg["pivot_right"]), "high")
    low_pivots = [event for event in low_events if event is not None]
    high_pivots = [event for event in high_events if event is not None]
    if len(low_pivots) < 2 or not high_pivots:
        result["double_bottom_reason"] = "pivot_not_ready"
        return result

    best_candidate: dict[str, Any] | None = None
    best_key: tuple[float, int, int] = (-1.0, -1, -1)

    for low_index, l1 in enumerate(low_pivots[:-1]):
        l1_bar = int(l1["pivot_bar"])
        l1_price = float(l1["price"])
        if l1_bar < int(cfg["pre_down_lookback_min"]) + 5:
            continue

        pre_start = max(0, l1_bar - int(cfg["pre_down_lookback_max"]))
        pre_end = l1_bar - int(cfg["pre_down_lookback_min"])
        if pre_end <= pre_start:
            continue
        pre_high = float(np.nanmax(high_arr[pre_start : pre_end + 1]))
        pre_down_pct = (pre_high / l1_price - 1.0) * 100.0 if l1_price > 0 else 0.0
        if pre_down_pct < float(cfg["min_pre_down_pct"]):
            continue

        for l2 in low_pivots[low_index + 1 :]:
            l2_bar = int(l2["pivot_bar"])
            l2_price = float(l2["price"])
            spacing_bars = l2_bar - l1_bar
            if spacing_bars < int(cfg["min_bars_between_bottoms"]):
                continue
            if spacing_bars > int(cfg["max_bars_between_bottoms"]):
                break
            if current_idx - l2_bar > int(cfg["pattern_stale_max_bars_after_l2"]):
                continue

            l2_vs_l1_pct = (l2_price / l1_price - 1.0) * 100.0 if l1_price > 0 else None
            if l2_vs_l1_pct is None or abs(l2_vs_l1_pct) > float(cfg["max_l2_deviation_pct"]):
                continue

            highs_between = [event for event in high_pivots if l1_bar < int(event["pivot_bar"]) < l2_bar]
            if not highs_between:
                continue
            h = max(highs_between, key=lambda item: float(item["price"]))
            h_bar = int(h["pivot_bar"])
            h_price = float(h["price"])
            rebound_pct = (h_price / l1_price - 1.0) * 100.0 if l1_price > 0 else 0.0
            if rebound_pct < float(cfg["min_rebound_pct"]):
                continue

            rebound_slice = volume_arr[l1_bar : h_bar + 1]
            pullback_slice = volume_arr[h_bar + 1 : l2_bar + 1]
            if len(rebound_slice) == 0 or len(pullback_slice) == 0:
                continue
            rebound_avg_vol = float(np.nanmean(rebound_slice))
            pullback_avg_vol = float(np.nanmean(pullback_slice))
            if rebound_avg_vol <= 0:
                continue
            pullback_volume_ratio = pullback_avg_vol / rebound_avg_vol
            if pullback_volume_ratio > float(cfg["pullback_volume_ratio_max"]):
                continue

            big_drop_threshold = float(cfg["pullback_big_drop_gem_pct"] if board in {"gem", "star"} else cfg["pullback_big_drop_main_pct"])
            big_drop_count = 0
            broken_distribution = False
            for idx in range(h_bar + 1, l2_bar + 1):
                vol_ma = vol_ma5_arr[idx]
                if np.isnan(vol_ma) or vol_ma <= 0:
                    continue
                bearish_heavy = close_arr[idx] < open_arr[idx] and pct_change[idx] <= big_drop_threshold and volume_arr[idx] >= vol_ma * float(
                    cfg["pullback_big_drop_volume_mult"]
                )
                if bearish_heavy:
                    big_drop_count += 1
                    broken_distribution = True
            if broken_distribution:
                continue

            l2_stop_signal = any(
                _is_stop_signal(
                    idx,
                    open_arr,
                    high_arr,
                    low_arr,
                    close_arr,
                    volume_arr,
                    vol_ma5_arr,
                    pre_close_arr,
                )
                for idx in range(max(h_bar + 1, l2_bar - 1), min(current_idx, l2_bar + 2) + 1)
            )
            if not l2_stop_signal:
                continue

            neckline = h_price
            setup = _evaluate_current_setup(
                current_idx,
                l2_bar,
                neckline,
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                volume_arr,
                vol_ma5_arr,
                ma5_arr,
                ma10_arr,
                pre_close_arr,
                cfg,
            )
            if not setup["buy_type"]:
                continue

            current_vs_ma20_pct = _percent_distance(close_arr[current_idx], ma20_arr[current_idx])
            if current_vs_ma20_pct is None or current_vs_ma20_pct < float(cfg["current_close_ma20_tolerance_pct"]):
                continue

            low_120 = float(np.nanmin(low_arr[max(0, current_idx - 119) : current_idx + 1]))
            high_120 = float(np.nanmax(high_arr[max(0, current_idx - 119) : current_idx + 1]))
            position_120 = None
            if high_120 > low_120:
                position_120 = (close_arr[current_idx] - low_120) / (high_120 - low_120)
            space_to_120_high_pct = (high_120 / close_arr[current_idx] - 1.0) * 100.0 if close_arr[current_idx] > 0 else None

            candidate = {
                "ts_code": ts_code,
                "double_bottom_board": board,
                "double_bottom_buy_type": setup["buy_type"],
                "double_bottom_breakout_status": setup["breakout_status"],
                "double_bottom_l1_date": trade_dates[l1_bar],
                "double_bottom_l1_price": to_number(l1_price, 4),
                "double_bottom_h_date": trade_dates[h_bar],
                "double_bottom_h_price": to_number(h_price, 4),
                "double_bottom_l2_date": trade_dates[l2_bar],
                "double_bottom_l2_price": to_number(l2_price, 4),
                "double_bottom_neckline": to_number(neckline, 4),
                "double_bottom_current_price": to_number(close_arr[current_idx], 4),
                "double_bottom_pre_down_pct": to_number(pre_down_pct, 4),
                "double_bottom_rebound_pct": to_number(rebound_pct, 4),
                "double_bottom_l2_vs_l1_pct": to_number(l2_vs_l1_pct, 4),
                "double_bottom_spacing_bars": int(spacing_bars),
                "double_bottom_pullback_volume_ratio": to_number(pullback_volume_ratio, 4),
                "double_bottom_pullback_big_drop_count": int(big_drop_count),
                "double_bottom_breakout_volume_ratio": to_number(setup["breakout_volume_ratio"], 4),
                "double_bottom_breakout_close_buffer_pct": to_number(setup["breakout_close_buffer_pct"], 4),
                "double_bottom_breakout_upper_shadow_pct": to_number(setup["breakout_upper_shadow_pct"], 4),
                "double_bottom_retest_neckline_buffer_pct": to_number(setup["retest_neckline_buffer_pct"], 4),
                "double_bottom_current_vs_ma20_pct": to_number(current_vs_ma20_pct, 4),
                "double_bottom_ma5_above_ma10": bool(ma5_arr[current_idx] > ma10_arr[current_idx]),
                "double_bottom_ma10_slope_pct": to_number(_percent_distance(ma10_arr[current_idx], ma10_arr[current_idx - 1]), 4),
                "double_bottom_ma20_slope_pct": to_number(_percent_distance(ma20_arr[current_idx], ma20_arr[current_idx - 1]), 4),
                "double_bottom_position_120": to_number(position_120, 4),
                "double_bottom_space_to_120_high_pct": to_number(space_to_120_high_pct, 4),
                "double_bottom_bars_since_l2": int(current_idx - l2_bar),
                "double_bottom_stop_signal": bool(l2_stop_signal),
            }
            candidate["double_bottom_score"] = build_double_bottom_score(candidate)
            candidate["double_bottom_signal"] = bool(candidate["double_bottom_score"] >= float(cfg["candidate_score_threshold"]))
            candidate["double_bottom_reason"] = (
                f"{setup['buy_type']}类买点,"
                f" L1-L2间隔{spacing_bars}天,"
                f" 回踩量比{candidate['double_bottom_pullback_volume_ratio']},"
                f" 当前分数{candidate['double_bottom_score']}"
            )
            candidate["double_bottom_core_reasons"] = _build_core_reasons(candidate)
            candidate["double_bottom_risks"] = _build_risks(candidate)
            if not candidate["double_bottom_signal"]:
                continue

            buy_rank = {"B": 3, "A": 2, "C": 1}.get(setup["buy_type"], 0)
            selection_key = (float(candidate["double_bottom_score"]), buy_rank, l2_bar)
            if selection_key > best_key:
                best_key = selection_key
                best_candidate = candidate

    if best_candidate is None:
        result["double_bottom_reason"] = "no_valid_double_bottom"
        return result

    result.update(best_candidate)
    return result


def build_double_bottom_snapshot(
    market_daily_history: pd.DataFrame,
    stock_basic_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if market_daily_history is None or market_daily_history.empty or stock_basic_df is None or stock_basic_df.empty:
        return pd.DataFrame()

    cfg = merge_double_bottom_config(config)
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
        features = calculate_double_bottom_features(sub, meta=meta, config=cfg)
        if features.get("double_bottom_signal"):
            rows.append(features)
    return pd.DataFrame(rows)
