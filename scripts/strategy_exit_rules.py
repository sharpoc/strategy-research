from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EXIT_CONFIG: dict[str, Any] = {
    "common": {
        "intraday_conflict_mode": "conservative",
    },
    "limitup_l1l2": {
        "max_hold_days": 10,
        "stop_below_l2_pct": 1.2,
        "breakeven_trigger_pct": 6.0,
        "breakeven_buffer_pct": 0.5,
        "trail_arm_pct": 9.0,
        "trail_from_peak_pct": 4.5,
        "trend_exit_arm_pct": 5.5,
        "trend_exit_min_hold_days": 3,
        "target_from_impulse_multiple": 0.80,
        "min_target_pct": 8.0,
    },
    "platform_breakout": {
        "max_hold_days": 12,
        "stop_below_support_pct": 1.2,
        "breakeven_trigger_pct": 7.0,
        "breakeven_buffer_pct": 0.6,
        "trail_arm_pct": 10.0,
        "trail_from_peak_pct": 5.0,
        "trend_exit_arm_pct": 6.5,
        "trend_exit_min_hold_days": 3,
        "target_from_pattern_multiple": 0.80,
        "min_target_pct": 10.0,
    },
    "double_bottom": {
        "max_hold_days": 15,
        "bottom_stop_buffer_pct": 1.5,
        "neckline_retest_break_pct": 2.0,
        "breakeven_trigger_pct": 8.0,
        "breakeven_buffer_pct": 0.8,
        "trail_arm_pct": 12.0,
        "trail_from_peak_pct": 5.5,
        "trend_exit_arm_pct": 7.5,
        "trend_exit_min_hold_days": 4,
        "target_measured_move_multiple": 1.00,
        "min_target_pct": 10.0,
    },
    "real_breakout": {
        "max_hold_days": 12,
        "stop_below_support_pct": 1.5,
        "breakeven_trigger_pct": 6.5,
        "breakeven_buffer_pct": 0.5,
        "trail_arm_pct": 9.5,
        "trail_from_peak_pct": 4.8,
        "trend_exit_arm_pct": 6.0,
        "trend_exit_min_hold_days": 3,
        "target_from_pattern_multiple": 0.80,
        "min_target_pct": 9.0,
    },
    "holder_increase": {
        "max_hold_days": 12,
        "hard_stop_pct": 6.5,
        "ma20_stop_buffer_pct": 1.2,
        "breakeven_trigger_pct": 7.0,
        "breakeven_buffer_pct": 0.5,
        "trail_arm_pct": 11.0,
        "trail_from_peak_pct": 4.8,
        "trend_exit_arm_pct": 5.0,
        "trend_exit_min_hold_days": 3,
        "min_target_pct": 12.0,
    },
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


def to_number(value: Any, digits: int = 4) -> float | None:
    numeric = to_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def merge_exit_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in DEFAULT_EXIT_CONFIG.items()
    }
    if not config:
        return merged
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update({item_key: item_value for item_key, item_value in value.items() if item_value is not None})
        else:
            merged[key] = value
    return merged


def build_price_path_map(market_daily_history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if market_daily_history is None or market_daily_history.empty:
        return {}

    history = market_daily_history.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    history["trade_date"] = history["trade_date"].astype(str)
    for column in ["open", "high", "low", "close", "vol"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    close_group = history.groupby("ts_code")["close"]
    vol_group = history.groupby("ts_code")["vol"]
    history["ma5"] = close_group.transform(lambda s: s.rolling(5, min_periods=5).mean())
    history["ma10"] = close_group.transform(lambda s: s.rolling(10, min_periods=10).mean())
    history["ma20"] = close_group.transform(lambda s: s.rolling(20, min_periods=20).mean())
    history["vol_ma5"] = vol_group.transform(lambda s: s.rolling(5, min_periods=1).mean())
    history["prev_close"] = history.groupby("ts_code")["close"].shift(1)
    return {str(ts_code): sub.reset_index(drop=True) for ts_code, sub in history.groupby("ts_code", sort=False)}


def _pct_return(price: float | None, base: float | None) -> float | None:
    if price in (None, 0) or base in (None, 0):
        return None
    return (float(price) / float(base) - 1.0) * 100.0


def _value_from_row(row: dict[str, Any], key: str) -> float | None:
    return to_float(row.get(key))


def _resolve_intraday_exit(
    open_price: float,
    high_price: float,
    low_price: float,
    stop_price: float | None,
    target_price: float | None,
    conflict_mode: str,
) -> tuple[str, float] | None:
    if stop_price is not None and open_price <= stop_price:
        return "stop_gap", float(open_price)
    if target_price is not None and open_price >= target_price:
        return "target_gap", float(open_price)

    stop_hit = stop_price is not None and low_price <= stop_price
    target_hit = target_price is not None and high_price >= target_price
    if not stop_hit and not target_hit:
        return None

    if stop_hit and target_hit:
        if conflict_mode == "target_first":
            return "target_intraday_conflict", float(target_price)
        if conflict_mode == "nearest_open":
            stop_distance = abs(open_price - float(stop_price))
            target_distance = abs(float(target_price) - open_price)
            if target_distance < stop_distance:
                return "target_intraday_conflict", float(target_price)
        return "stop_intraday_conflict", float(stop_price)

    if stop_hit:
        return "stop_intraday", float(stop_price)
    return "target_intraday", float(target_price)


def _trend_close_break(
    idx: int,
    path: pd.DataFrame,
    arm_profit_pct: float,
    peak_return_pct: float,
    min_hold_days: int,
    hold_days: int,
) -> bool:
    if hold_days < min_hold_days or peak_return_pct < arm_profit_pct or idx <= 0:
        return False
    close_price = to_float(path.at[idx, "close"])
    prev_close = to_float(path.at[idx - 1, "close"])
    ma5 = to_float(path.at[idx, "ma5"])
    ma10 = to_float(path.at[idx, "ma10"])
    if close_price is None or prev_close is None:
        return False
    below_ma5 = ma5 is not None and close_price < ma5
    below_ma10 = ma10 is not None and close_price < ma10
    return bool((below_ma5 and close_price < prev_close) or below_ma10)


def _build_limitup_policy(row: dict[str, Any], entry_price: float, cfg: dict[str, Any]) -> dict[str, Any]:
    l1_price = _value_from_row(row, "limitup_l1l2_l1_price")
    l2_price = _value_from_row(row, "limitup_l1l2_l2_price")
    impulse_high = _value_from_row(row, "limitup_l1l2_impulse_high")
    structure_stop = None
    if l2_price is not None:
        structure_stop = l2_price * (1.0 - float(cfg["stop_below_l2_pct"]) / 100.0)
    measured_target = None
    if impulse_high is not None and l1_price is not None and l2_price is not None and impulse_high > l1_price:
        measured_target = l2_price + (impulse_high - l1_price) * float(cfg["target_from_impulse_multiple"])
    min_target = entry_price * (1.0 + float(cfg["min_target_pct"]) / 100.0)
    target_price = max(filter(lambda x: x is not None and x > 0, [measured_target, min_target]), default=None)
    return {
        "max_hold_days": int(cfg["max_hold_days"]),
        "structure_stop": structure_stop,
        "target_price": target_price,
        "breakeven_trigger_pct": float(cfg["breakeven_trigger_pct"]),
        "breakeven_buffer_pct": float(cfg["breakeven_buffer_pct"]),
        "trail_arm_pct": float(cfg["trail_arm_pct"]),
        "trail_from_peak_pct": float(cfg["trail_from_peak_pct"]),
        "trend_exit_arm_pct": float(cfg["trend_exit_arm_pct"]),
        "trend_exit_min_hold_days": int(cfg["trend_exit_min_hold_days"]),
    }


def _build_platform_policy(row: dict[str, Any], entry_price: float, cfg: dict[str, Any]) -> dict[str, Any]:
    platform_high = _value_from_row(row, "platform_breakout_platform_high")
    platform_low = _value_from_row(row, "platform_breakout_platform_low")
    limit_high = _value_from_row(row, "platform_breakout_limit_high")
    pullback_low_price = _value_from_row(row, "platform_breakout_pullback_low_price")
    support_anchor = max(value for value in [platform_high, pullback_low_price] if value is not None) if any(
        value is not None for value in [platform_high, pullback_low_price]
    ) else None
    structure_stop = None
    if support_anchor is not None:
        structure_stop = support_anchor * (1.0 - float(cfg["stop_below_support_pct"]) / 100.0)
    pattern_target = None
    if limit_high is not None and platform_low is not None and limit_high > platform_low:
        pattern_target = limit_high + (limit_high - platform_low) * float(cfg["target_from_pattern_multiple"])
    min_target = entry_price * (1.0 + float(cfg["min_target_pct"]) / 100.0)
    target_price = max(filter(lambda x: x is not None and x > 0, [pattern_target, min_target]), default=None)
    return {
        "max_hold_days": int(cfg["max_hold_days"]),
        "structure_stop": structure_stop,
        "target_price": target_price,
        "breakeven_trigger_pct": float(cfg["breakeven_trigger_pct"]),
        "breakeven_buffer_pct": float(cfg["breakeven_buffer_pct"]),
        "trail_arm_pct": float(cfg["trail_arm_pct"]),
        "trail_from_peak_pct": float(cfg["trail_from_peak_pct"]),
        "trend_exit_arm_pct": float(cfg["trend_exit_arm_pct"]),
        "trend_exit_min_hold_days": int(cfg["trend_exit_min_hold_days"]),
    }


def _build_double_bottom_policy(row: dict[str, Any], entry_price: float, cfg: dict[str, Any]) -> dict[str, Any]:
    l1_price = _value_from_row(row, "double_bottom_l1_price")
    l2_price = _value_from_row(row, "double_bottom_l2_price")
    neckline = _value_from_row(row, "double_bottom_neckline")
    buy_type = str(row.get("double_bottom_buy_type") or "")
    bottom_floor = min(value for value in [l1_price, l2_price] if value is not None) if any(
        value is not None for value in [l1_price, l2_price]
    ) else None
    bottom_stop = None
    if bottom_floor is not None:
        bottom_stop = bottom_floor * (1.0 - float(cfg["bottom_stop_buffer_pct"]) / 100.0)
    neckline_stop = None
    if neckline is not None and buy_type in {"A", "B"}:
        neckline_stop = neckline * (1.0 - float(cfg["neckline_retest_break_pct"]) / 100.0)
    structure_stop_candidates = [value for value in [bottom_stop, neckline_stop] if value is not None]
    structure_stop = max(structure_stop_candidates) if structure_stop_candidates else None
    measured_target = None
    if neckline is not None and bottom_floor is not None and neckline > bottom_floor:
        measured_target = neckline + (neckline - bottom_floor) * float(cfg["target_measured_move_multiple"])
    min_target = entry_price * (1.0 + float(cfg["min_target_pct"]) / 100.0)
    target_price = max(filter(lambda x: x is not None and x > 0, [measured_target, min_target]), default=None)
    return {
        "max_hold_days": int(cfg["max_hold_days"]),
        "structure_stop": structure_stop,
        "target_price": target_price,
        "breakeven_trigger_pct": float(cfg["breakeven_trigger_pct"]),
        "breakeven_buffer_pct": float(cfg["breakeven_buffer_pct"]),
        "trail_arm_pct": float(cfg["trail_arm_pct"]),
        "trail_from_peak_pct": float(cfg["trail_from_peak_pct"]),
        "trend_exit_arm_pct": float(cfg["trend_exit_arm_pct"]),
        "trend_exit_min_hold_days": int(cfg["trend_exit_min_hold_days"]),
    }


def _build_real_breakout_policy(row: dict[str, Any], entry_price: float, cfg: dict[str, Any]) -> dict[str, Any]:
    platform_high = _value_from_row(row, "real_breakout_platform_high")
    platform_low = _value_from_row(row, "real_breakout_platform_low")
    breakout_high = _value_from_row(row, "real_breakout_breakout_high")
    support_anchor = platform_high
    structure_stop = None
    if support_anchor is not None:
        structure_stop = support_anchor * (1.0 - float(cfg["stop_below_support_pct"]) / 100.0)
    pattern_target = None
    if breakout_high is not None and platform_high is not None and platform_low is not None and platform_high > platform_low:
        pattern_target = breakout_high + (platform_high - platform_low) * float(cfg["target_from_pattern_multiple"])
    min_target = entry_price * (1.0 + float(cfg["min_target_pct"]) / 100.0)
    target_price = max(filter(lambda x: x is not None and x > 0, [pattern_target, min_target]), default=None)
    return {
        "max_hold_days": int(cfg["max_hold_days"]),
        "structure_stop": structure_stop,
        "target_price": target_price,
        "breakeven_trigger_pct": float(cfg["breakeven_trigger_pct"]),
        "breakeven_buffer_pct": float(cfg["breakeven_buffer_pct"]),
        "trail_arm_pct": float(cfg["trail_arm_pct"]),
        "trail_from_peak_pct": float(cfg["trail_from_peak_pct"]),
        "trend_exit_arm_pct": float(cfg["trend_exit_arm_pct"]),
        "trend_exit_min_hold_days": int(cfg["trend_exit_min_hold_days"]),
    }


def _build_holder_policy(row: dict[str, Any], entry_price: float, cfg: dict[str, Any]) -> dict[str, Any]:
    ma20 = _value_from_row(row, "ma_qfq_20")
    ma10 = _value_from_row(row, "ma_qfq_10")
    hard_stop = entry_price * (1.0 - float(cfg["hard_stop_pct"]) / 100.0)
    ma20_stop = None
    if ma20 is not None and ma20 > 0:
        ma20_stop = ma20 * (1.0 - float(cfg["ma20_stop_buffer_pct"]) / 100.0)
    structure_stop_candidates = [value for value in [hard_stop, ma20_stop] if value is not None and value > 0]
    structure_stop = max(structure_stop_candidates) if structure_stop_candidates else None
    if ma10 is not None and structure_stop is not None and ma10 > structure_stop and ma10 < entry_price:
        structure_stop = max(structure_stop, ma10 * 0.985)
    target_price = entry_price * (1.0 + float(cfg["min_target_pct"]) / 100.0)
    return {
        "max_hold_days": int(cfg["max_hold_days"]),
        "structure_stop": structure_stop,
        "target_price": target_price,
        "breakeven_trigger_pct": float(cfg["breakeven_trigger_pct"]),
        "breakeven_buffer_pct": float(cfg["breakeven_buffer_pct"]),
        "trail_arm_pct": float(cfg["trail_arm_pct"]),
        "trail_from_peak_pct": float(cfg["trail_from_peak_pct"]),
        "trend_exit_arm_pct": float(cfg["trend_exit_arm_pct"]),
        "trend_exit_min_hold_days": int(cfg["trend_exit_min_hold_days"]),
    }


def _build_exit_policy(strategy_id: str, row: dict[str, Any], entry_price: float, config: dict[str, Any]) -> dict[str, Any]:
    if strategy_id == "limitup_l1l2":
        return _build_limitup_policy(row, entry_price, config[strategy_id])
    if strategy_id == "platform_breakout":
        return _build_platform_policy(row, entry_price, config[strategy_id])
    if strategy_id == "double_bottom":
        return _build_double_bottom_policy(row, entry_price, config[strategy_id])
    if strategy_id == "real_breakout":
        return _build_real_breakout_policy(row, entry_price, config[strategy_id])
    if strategy_id == "holder_increase":
        return _build_holder_policy(row, entry_price, config[strategy_id])
    raise ValueError(f"Unsupported strategy_id for exit policy: {strategy_id}")


def simulate_exit_for_signal(
    row: dict[str, Any],
    price_path_map: dict[str, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_config = merge_exit_config(config)
    strategy_id = str(row.get("strategy_id") or "")
    ts_code = str(row.get("ts_code") or "")
    entry_trade_date = str(row.get("entry_trade_date") or "")
    entry_open = to_float(row.get("entry_open"))
    if not strategy_id or not ts_code or not entry_trade_date or entry_open is None:
        return {}
    path = price_path_map.get(ts_code)
    if path is None or path.empty:
        return {}
    match = path.index[path["trade_date"].astype(str) == entry_trade_date]
    if len(match) == 0:
        return {}
    entry_idx = int(match[0])
    policy = _build_exit_policy(strategy_id, row, entry_open, merged_config)
    conflict_mode = str(merged_config.get("common", {}).get("intraday_conflict_mode", "conservative"))

    structure_stop = to_float(policy.get("structure_stop"))
    target_price = to_float(policy.get("target_price"))
    breakeven_trigger_pct = float(policy.get("breakeven_trigger_pct", 0.0))
    breakeven_buffer_pct = float(policy.get("breakeven_buffer_pct", 0.0))
    trail_arm_pct = float(policy.get("trail_arm_pct", 0.0))
    trail_from_peak_pct = float(policy.get("trail_from_peak_pct", 0.0))
    trend_exit_arm_pct = float(policy.get("trend_exit_arm_pct", 0.0))
    trend_exit_min_hold_days = int(policy.get("trend_exit_min_hold_days", 1))
    max_hold_days = int(policy.get("max_hold_days", 10))

    peak_high = float(entry_open)
    trough_low = float(entry_open)
    active_stop = structure_stop
    exit_price = None
    exit_trade_date = None
    exit_reason = ""
    exit_rule = ""

    last_idx = min(len(path) - 1, entry_idx + max_hold_days - 1)
    for idx in range(entry_idx, last_idx + 1):
        open_price = to_float(path.at[idx, "open"])
        high_price = to_float(path.at[idx, "high"])
        low_price = to_float(path.at[idx, "low"])
        close_price = to_float(path.at[idx, "close"])
        if any(value is None for value in [open_price, high_price, low_price, close_price]):
            continue

        peak_high = max(peak_high, float(high_price))
        trough_low = min(trough_low, float(low_price))
        peak_return_pct = _pct_return(peak_high, entry_open) or 0.0
        hold_days = idx - entry_idx + 1

        stop_candidates = [value for value in [structure_stop] if value is not None]
        if peak_return_pct >= breakeven_trigger_pct:
            stop_candidates.append(entry_open * (1.0 + breakeven_buffer_pct / 100.0))
        if peak_return_pct >= trail_arm_pct:
            stop_candidates.append(peak_high * (1.0 - trail_from_peak_pct / 100.0))
        active_stop = max(stop_candidates) if stop_candidates else None

        intraday_exit = _resolve_intraday_exit(
            open_price=float(open_price),
            high_price=float(high_price),
            low_price=float(low_price),
            stop_price=active_stop,
            target_price=target_price,
            conflict_mode=conflict_mode,
        )
        if intraday_exit is not None:
            exit_reason, exit_price = intraday_exit
            exit_trade_date = str(path.at[idx, "trade_date"])
            exit_rule = "price_level"
            break

        if _trend_close_break(
            idx=idx,
            path=path,
            arm_profit_pct=trend_exit_arm_pct,
            peak_return_pct=peak_return_pct,
            min_hold_days=trend_exit_min_hold_days,
            hold_days=hold_days,
        ):
            exit_price = float(close_price)
            exit_trade_date = str(path.at[idx, "trade_date"])
            exit_reason = "trend_close_break"
            exit_rule = "close_rule"
            break

        if hold_days >= max_hold_days:
            exit_price = float(close_price)
            exit_trade_date = str(path.at[idx, "trade_date"])
            exit_reason = "time_exit"
            exit_rule = "close_rule"
            break

    if exit_price is None:
        last_close = to_float(path.at[last_idx, "close"])
        if last_close is None:
            return {}
        exit_price = float(last_close)
        exit_trade_date = str(path.at[last_idx, "trade_date"])
        exit_reason = "data_end"
        exit_rule = "close_rule"

    exit_return_pct = _pct_return(exit_price, entry_open)
    peak_return_pct = _pct_return(peak_high, entry_open)
    trough_return_pct = _pct_return(trough_low, entry_open)
    return {
        "exit_trade_date": exit_trade_date,
        "exit_price": to_number(exit_price),
        "exit_reason": exit_reason,
        "exit_rule": exit_rule,
        "exit_hold_days": int((path.index[path["trade_date"].astype(str) == exit_trade_date][0] - entry_idx) + 1),
        "exit_return_pct": to_number(exit_return_pct),
        "exit_target_price": to_number(target_price),
        "exit_structure_stop": to_number(structure_stop),
        "exit_active_stop": to_number(active_stop),
        "exit_peak_price": to_number(peak_high),
        "exit_mfe_pct": to_number(peak_return_pct),
        "exit_mae_pct": to_number(trough_return_pct),
    }


def apply_exit_rules(
    daily_results: pd.DataFrame,
    price_path_map: dict[str, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if daily_results is None or daily_results.empty:
        return daily_results

    work = daily_results.copy()
    exit_rows: list[dict[str, Any]] = []
    for row in work.to_dict(orient="records"):
        has_signal = bool(row.get("has_signal"))
        if not has_signal:
            exit_rows.append({})
            continue
        exit_rows.append(simulate_exit_for_signal(row, price_path_map, config=config))

    exit_frame = pd.DataFrame(exit_rows)
    if exit_frame.empty:
        return work
    for column in exit_frame.columns:
        work[column] = exit_frame[column]
    return work


def summarize_exit_reasons(daily_results: pd.DataFrame) -> pd.DataFrame:
    if daily_results is None or daily_results.empty or "exit_reason" not in daily_results.columns:
        return pd.DataFrame()

    signal_rows = daily_results[(daily_results["has_signal"]) & daily_results["exit_reason"].fillna("").ne("")].copy()
    if signal_rows.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, sub in signal_rows.groupby(["strategy_id", "market_regime", "exit_reason"], dropna=False, sort=False):
        strategy_id, market_regime, exit_reason = keys
        valid_returns = pd.to_numeric(sub.get("exit_return_pct"), errors="coerce").dropna()
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": sub["strategy_name"].iloc[0],
                "market_regime": market_regime,
                "exit_reason": exit_reason,
                "count": int(len(sub)),
                "avg_exit_return_pct": round(float(valid_returns.mean()), 4) if not valid_returns.empty else None,
                "avg_exit_hold_days": round(float(pd.to_numeric(sub.get("exit_hold_days"), errors="coerce").dropna().mean()), 2)
                if "exit_hold_days" in sub.columns and not pd.to_numeric(sub.get("exit_hold_days"), errors="coerce").dropna().empty
                else None,
            }
        )
    return pd.DataFrame(rows)
