from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from double_bottom_strategy import build_double_bottom_snapshot
from limitup_l1l2_strategy import build_limitup_l1l2_snapshot
from platform_breakout_retest_strategy import build_platform_breakout_snapshot
from real_fund_breakout_strategy import build_real_breakout_snapshot
from run_tushare_limitup_l1l2_strategy import build_strategy_rank_score as build_limitup_rank_score


@dataclass(frozen=True)
class StrategyPluginSpec:
    strategy_id: str
    strategy_name: str
    history_bars: int
    build_candidates: Callable[[pd.DataFrame, pd.DataFrame, dict[str, Any]], pd.DataFrame]
    strategy_kind: str = "price"
    recommended_history_years: float = 2.5
    required_dataset_ids: tuple[str, ...] = ("trade_calendar", "stock_basic_all", "market_daily_all")
    optional_dataset_ids: tuple[str, ...] = ()


def local_latest_indicator_snapshot(window_history: pd.DataFrame) -> pd.DataFrame:
    history = window_history.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    for col in ["close", "vol"]:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history["ma20"] = history.groupby("ts_code")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    history["ma60"] = history.groupby("ts_code")["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    history["vol_ma5"] = history.groupby("ts_code")["vol"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    history["volume_ratio_local"] = np.where(history["vol_ma5"] > 0, history["vol"] / history["vol_ma5"], np.nan)
    latest = history.groupby("ts_code", as_index=False).tail(1).copy()
    latest = latest.rename(
        columns={
            "close": "close_qfq",
            "ma20": "ma_qfq_20",
            "ma60": "ma_qfq_60",
            "volume_ratio_local": "volume_ratio",
        }
    )
    keep_cols = [c for c in ["ts_code", "close_qfq", "ma_qfq_20", "ma_qfq_60", "volume_ratio", "trade_date"] if c in latest.columns]
    return latest[keep_cols].copy()


def _score_band(value: float | None, good_low: float, good_high: float, good_score: float, ok_score: float, miss_score: float) -> float:
    if value is None:
        return miss_score
    if good_low <= value <= good_high:
        return good_score
    return ok_score


def build_limitup_research_rank_score(row: dict[str, Any], tuning: dict[str, Any] | None = None) -> float:
    cfg = dict(tuning or {})
    score = 0.0

    bars_since_limit = pd.to_numeric(row.get("limitup_l1l2_bars_since_limit"), errors="coerce")
    bars_lu_to_l1 = pd.to_numeric(row.get("limitup_l1l2_bars_lu_to_l1"), errors="coerce")
    bars_l1_to_l2 = pd.to_numeric(row.get("limitup_l1l2_bars_l1_to_l2"), errors="coerce")
    impulse_pct = pd.to_numeric(row.get("limitup_l1l2_impulse_pct"), errors="coerce")
    pullback_pct = pd.to_numeric(row.get("limitup_l1l2_pullback_pct"), errors="coerce")
    l2_above_l1_pct = pd.to_numeric(row.get("limitup_l1l2_l2_above_l1_pct"), errors="coerce")
    confirm_vol_ratio = pd.to_numeric(row.get("limitup_l1l2_confirm_vol_ratio"), errors="coerce")
    close_vs_l2_pct = pd.to_numeric(row.get("limitup_l1l2_close_vs_l2_pct"), errors="coerce")
    hold_buffer_pct = pd.to_numeric(row.get("limitup_l1l2_hold_buffer_pct"), errors="coerce")
    volume_ratio = pd.to_numeric(row.get("volume_ratio"), errors="coerce")
    flow_3d_rank = pd.to_numeric(row.get("main_net_amount_3d_rank_pct"), errors="coerce")
    flow_5d_rank = pd.to_numeric(row.get("main_net_amount_5d_rank_pct"), errors="coerce")
    market_regime = str(row.get("market_regime") or "")

    score += _score_band(bars_since_limit, 12.0, 35.0, 18.0, 8.0, -6.0)
    if pd.notna(bars_since_limit) and bars_since_limit > 55:
        score -= 8.0
    score += _score_band(bars_lu_to_l1, 4.0, 15.0, 12.0, 4.0, -6.0)
    if pd.notna(bars_lu_to_l1) and bars_lu_to_l1 > 28:
        score -= 8.0
    score += _score_band(bars_l1_to_l2, 6.0, 18.0, 12.0, 4.0, -8.0)
    if pd.notna(bars_l1_to_l2) and bars_l1_to_l2 > 28:
        score -= 8.0
    score += _score_band(impulse_pct, 8.0, 18.0, 14.0, 5.0, -10.0)
    if pd.notna(impulse_pct) and impulse_pct > 26:
        score -= 4.0
    score += _score_band(pullback_pct, 3.0, 9.0, 10.0, 4.0, -8.0)
    if pd.notna(pullback_pct) and pullback_pct > 12:
        score -= 6.0
    score += _score_band(l2_above_l1_pct, 2.0, 7.0, 12.0, 5.0, -8.0)
    if pd.notna(l2_above_l1_pct) and l2_above_l1_pct > 8.5:
        score -= 5.0
    score += _score_band(confirm_vol_ratio, 1.35, 2.6, 8.0, 3.0, -4.0)

    if pd.notna(close_vs_l2_pct):
        if 1.0 <= close_vs_l2_pct <= 4.8:
            score += 10.0
        elif 0.0 <= close_vs_l2_pct <= 6.0:
            score += 5.0
        elif close_vs_l2_pct > 7.0:
            score -= 6.0
        elif close_vs_l2_pct < 0.0:
            score -= 8.0
    if pd.notna(hold_buffer_pct):
        if hold_buffer_pct >= 1.0:
            score += 10.0
        elif hold_buffer_pct >= 0.3:
            score += 4.0
        else:
            score -= 6.0
    if pd.notna(volume_ratio):
        if 1.0 <= volume_ratio <= 2.5:
            score += 4.0
        elif volume_ratio > 3.8:
            score -= 3.0
    if bool(row.get("limitup_l1l2_trend_ok")):
        score += 4.0
    if bool(row.get("limitup_l1l2_volume_ok")):
        score += 2.0
    if bool(row.get("limitup_l1l2_limit_sealed")):
        score += 2.0
    if pd.notna(flow_3d_rank):
        score += float(flow_3d_rank) * 6.0
    if pd.notna(flow_5d_rank):
        score += float(flow_5d_rank) * 2.0
    if market_regime == "震荡趋势":
        score += float(cfg.get("range_regime_bonus", 6.0))
    elif market_regime == "上涨趋势":
        score += float(cfg.get("uptrend_penalty", -6.0))
    elif market_regime == "下跌趋势":
        score += float(cfg.get("downtrend_penalty", -10.0))
    return round(float(score), 2)


def apply_limitup_research_filters(candidates: pd.DataFrame, tuning: dict[str, Any] | None = None) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    cfg = dict(tuning or {})
    if not cfg.get("enabled", False):
        return candidates
    work = candidates.copy()

    def _mask_between(column: str, min_value: float | None = None, max_value: float | None = None) -> pd.Series:
        values = pd.to_numeric(work.get(column), errors="coerce")
        mask = pd.Series(True, index=work.index)
        if min_value is not None:
            mask &= values >= float(min_value)
        if max_value is not None:
            mask &= values <= float(max_value)
        return mask.fillna(False)

    mask = pd.Series(True, index=work.index)
    if "max_bars_since_limit" in cfg:
        mask &= _mask_between("limitup_l1l2_bars_since_limit", max_value=cfg.get("max_bars_since_limit"))
    if "max_bars_lu_to_l1" in cfg:
        mask &= _mask_between("limitup_l1l2_bars_lu_to_l1", max_value=cfg.get("max_bars_lu_to_l1"))
    if "min_bars_l1_to_l2" in cfg or "max_bars_l1_to_l2" in cfg:
        mask &= _mask_between("limitup_l1l2_bars_l1_to_l2", min_value=cfg.get("min_bars_l1_to_l2"), max_value=cfg.get("max_bars_l1_to_l2"))
    if "min_impulse_pct" in cfg or "max_impulse_pct" in cfg:
        mask &= _mask_between("limitup_l1l2_impulse_pct", min_value=cfg.get("min_impulse_pct"), max_value=cfg.get("max_impulse_pct"))
    if "min_pullback_pct" in cfg or "max_pullback_pct" in cfg:
        mask &= _mask_between("limitup_l1l2_pullback_pct", min_value=cfg.get("min_pullback_pct"), max_value=cfg.get("max_pullback_pct"))
    if "min_l2_above_l1_pct" in cfg or "max_l2_above_l1_pct" in cfg:
        mask &= _mask_between("limitup_l1l2_l2_above_l1_pct", min_value=cfg.get("min_l2_above_l1_pct"), max_value=cfg.get("max_l2_above_l1_pct"))
    if "min_hold_buffer_pct" in cfg:
        mask &= _mask_between("limitup_l1l2_hold_buffer_pct", min_value=cfg.get("min_hold_buffer_pct"))
    if "max_close_vs_l2_pct" in cfg:
        mask &= _mask_between("limitup_l1l2_close_vs_l2_pct", max_value=cfg.get("max_close_vs_l2_pct"))

    def _bool_series(column: str) -> pd.Series:
        if column not in work.columns:
            return pd.Series(False, index=work.index)
        return work[column].fillna(False).astype(bool)

    if cfg.get("require_trend_ok", False):
        mask &= _bool_series("limitup_l1l2_trend_ok")
    if cfg.get("require_volume_ok", False):
        mask &= _bool_series("limitup_l1l2_volume_ok")
    if cfg.get("require_limit_sealed", False):
        mask &= _bool_series("limitup_l1l2_limit_sealed")
    return work[mask].copy().reset_index(drop=True)


def apply_limitup_research_entry_gate(row: dict[str, Any], market_regime: str, gate_config: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    cfg = dict(gate_config or {})
    if not cfg.get("enabled", False):
        return True, []
    failures: list[str] = []

    def _get_float(column: str) -> float | None:
        value = pd.to_numeric(row.get(column), errors="coerce")
        return None if pd.isna(value) else float(value)

    def _check_between(label: str, column: str, min_value: float | None = None, max_value: float | None = None) -> None:
        value = _get_float(column)
        if value is None:
            failures.append(f"{label}缺失")
            return
        if min_value is not None and value < float(min_value):
            failures.append(f"{label}<{float(min_value):g}")
        if max_value is not None and value > float(max_value):
            failures.append(f"{label}>{float(max_value):g}")

    allowed_market_regimes = [str(item) for item in cfg.get("allowed_market_regimes", []) if str(item)]
    if allowed_market_regimes and market_regime not in allowed_market_regimes:
        failures.append(f"市场状态不匹配:{market_regime}")
    _check_between("距涨停bar", "limitup_l1l2_bars_since_limit", cfg.get("min_bars_since_limit"), cfg.get("max_bars_since_limit"))
    _check_between("涨停到L1", "limitup_l1l2_bars_lu_to_l1", cfg.get("min_bars_lu_to_l1"), cfg.get("max_bars_lu_to_l1"))
    _check_between("L1到L2", "limitup_l1l2_bars_l1_to_l2", cfg.get("min_bars_l1_to_l2"), cfg.get("max_bars_l1_to_l2"))
    _check_between("冲高幅度", "limitup_l1l2_impulse_pct", cfg.get("min_impulse_pct"), cfg.get("max_impulse_pct"))
    _check_between("回撤幅度", "limitup_l1l2_pullback_pct", cfg.get("min_pullback_pct"), cfg.get("max_pullback_pct"))
    _check_between("L2高于L1", "limitup_l1l2_l2_above_l1_pct", cfg.get("min_l2_above_l1_pct"), cfg.get("max_l2_above_l1_pct"))
    _check_between("L2防守缓冲", "limitup_l1l2_hold_buffer_pct", cfg.get("min_hold_buffer_pct"), cfg.get("max_hold_buffer_pct"))
    _check_between("现价偏离L2", "limitup_l1l2_close_vs_l2_pct", cfg.get("min_close_vs_l2_pct"), cfg.get("max_close_vs_l2_pct"))
    if cfg.get("require_trend_ok", False) and not bool(row.get("limitup_l1l2_trend_ok")):
        failures.append("趋势过滤未通过")
    if cfg.get("require_volume_ok", False) and not bool(row.get("limitup_l1l2_volume_ok")):
        failures.append("量能过滤未通过")
    if cfg.get("require_limit_sealed", False) and not bool(row.get("limitup_l1l2_limit_sealed")):
        failures.append("涨停封板不足")
    return len(failures) == 0, failures


def build_real_breakout_research_rank_score(row: dict[str, Any], tuning: dict[str, Any] | None = None) -> float:
    cfg = dict(tuning or {})
    stage = str(row.get("real_breakout_stage") or "")
    pre_runup_pct = pd.to_numeric(row.get("real_breakout_pre_runup_pct"), errors="coerce")
    platform_days = pd.to_numeric(row.get("real_breakout_platform_days"), errors="coerce")
    platform_amp_pct = pd.to_numeric(row.get("real_breakout_platform_amp_pct"), errors="coerce")
    platform_vol_ratio = pd.to_numeric(row.get("real_breakout_platform_vol_ratio"), errors="coerce")
    platform_tail_vol_ratio = pd.to_numeric(row.get("real_breakout_platform_tail_vol_ratio"), errors="coerce")
    breakout_close_buffer_pct = pd.to_numeric(row.get("real_breakout_breakout_close_buffer_pct"), errors="coerce")
    breakout_volume_ratio = pd.to_numeric(row.get("real_breakout_breakout_volume_ratio"), errors="coerce")
    breakout_amount_ratio = pd.to_numeric(row.get("real_breakout_breakout_amount_ratio"), errors="coerce")
    breakout_upper_shadow_pct = pd.to_numeric(row.get("real_breakout_breakout_upper_shadow_pct"), errors="coerce")
    breakout_close_to_high_pct = pd.to_numeric(row.get("real_breakout_breakout_close_to_high_pct"), errors="coerce")
    current_buffer_pct = pd.to_numeric(row.get("real_breakout_current_buffer_pct"), errors="coerce")
    current_volume_ratio = pd.to_numeric(row.get("real_breakout_current_volume_ratio"), errors="coerce")
    ma20_slope_pct = pd.to_numeric(row.get("real_breakout_ma20_slope_pct"), errors="coerce")

    score = 0.0
    score += {
        "retest_hold": float(cfg.get("retest_hold_bonus", 18.0)),
        "follow_through": float(cfg.get("follow_through_bonus", 14.0)),
        "breakout_today": float(cfg.get("breakout_today_bonus", 4.0)),
    }.get(stage, 0.0)
    score += _score_band(pre_runup_pct, 15.0, 28.0, 10.0, 4.0, -8.0)
    if pd.notna(pre_runup_pct) and pre_runup_pct > 40.0:
        score -= 4.0
    score += _score_band(platform_days, 6.0, 10.0, 8.0, 4.0, -4.0)
    score += _score_band(platform_amp_pct, 3.0, 8.5, 10.0, 4.0, -6.0)
    score += _score_band(platform_vol_ratio, 0.0, 0.72, 12.0, 6.0, -8.0)
    score += _score_band(platform_tail_vol_ratio, 0.0, 0.90, 6.0, 3.0, -4.0)
    score += _score_band(breakout_close_buffer_pct, 1.2, 4.5, 8.0, 3.0, -5.0)
    score += _score_band(breakout_volume_ratio, 1.35, 2.10, 12.0, 5.0, -10.0)
    if pd.notna(breakout_volume_ratio) and breakout_volume_ratio > 2.6:
        score -= 4.0
    score += _score_band(breakout_amount_ratio, 1.10, 2.40, 7.0, 3.0, -4.0)
    if pd.notna(breakout_upper_shadow_pct):
        if breakout_upper_shadow_pct <= 0.8:
            score += 3.0
        elif breakout_upper_shadow_pct <= 1.5:
            score += 1.0
        else:
            score -= 5.0
    if pd.notna(breakout_close_to_high_pct):
        if breakout_close_to_high_pct <= 0.8:
            score += 3.0
        elif breakout_close_to_high_pct <= 1.4:
            score += 1.0
        else:
            score -= 3.0
    if pd.notna(current_buffer_pct):
        if 0.5 <= current_buffer_pct <= 6.5:
            score += 10.0
        elif 0.0 <= current_buffer_pct <= 8.5:
            score += 4.0
        elif current_buffer_pct > 10.0:
            score -= 6.0
        else:
            score -= 10.0
    if pd.notna(current_volume_ratio):
        if stage == "retest_hold":
            if current_volume_ratio <= 1.15:
                score += 4.0
            elif current_volume_ratio > 1.60:
                score -= 4.0
        elif stage == "follow_through":
            if 1.0 <= current_volume_ratio <= 2.2:
                score += 4.0
            elif current_volume_ratio < 0.9:
                score -= 3.0
    if pd.notna(ma20_slope_pct):
        if 0.0 <= ma20_slope_pct <= 2.2:
            score += 5.0
        elif ma20_slope_pct > 2.2:
            score += 2.0
        else:
            score -= 6.0
    if bool(row.get("real_breakout_retest_ok")):
        score += 4.0
    if bool(row.get("real_breakout_follow_ok")):
        score += 2.0
    if bool(row.get("real_breakout_breakout_today")):
        score += float(cfg.get("breakout_today_penalty", -8.0))
    if bool(row.get("real_breakout_extreme_volume_flag")):
        score -= 6.0
    base_score = pd.to_numeric(row.get("real_breakout_score"), errors="coerce")
    if pd.notna(base_score):
        score += float(base_score) * float(cfg.get("base_score_weight", 0.12))
    return round(float(min(max(score, 0.0), 100.0)), 2)


def apply_real_breakout_research_filters(candidates: pd.DataFrame, tuning: dict[str, Any] | None = None) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    cfg = dict(tuning or {})
    if not cfg.get("enabled", False):
        return candidates
    work = candidates.copy()

    def _mask_between(column: str, min_value: float | None = None, max_value: float | None = None) -> pd.Series:
        values = pd.to_numeric(work.get(column), errors="coerce")
        mask = pd.Series(True, index=work.index)
        if min_value is not None:
            mask &= values >= float(min_value)
        if max_value is not None:
            mask &= values <= float(max_value)
        return mask.fillna(False)

    mask = pd.Series(True, index=work.index)
    allowed_stages = [str(item) for item in cfg.get("allowed_stages", []) if str(item)]
    if allowed_stages:
        mask &= work.get("real_breakout_stage", "").astype(str).isin(allowed_stages)
    if cfg.get("disallow_breakout_today", False):
        mask &= ~work.get("real_breakout_breakout_today", False).fillna(False).astype(bool)
    if "min_pre_runup_pct" in cfg or "max_pre_runup_pct" in cfg:
        mask &= _mask_between("real_breakout_pre_runup_pct", min_value=cfg.get("min_pre_runup_pct"), max_value=cfg.get("max_pre_runup_pct"))
    if "min_platform_days" in cfg or "max_platform_days" in cfg:
        mask &= _mask_between("real_breakout_platform_days", min_value=cfg.get("min_platform_days"), max_value=cfg.get("max_platform_days"))
    if "max_platform_amp_pct" in cfg:
        mask &= _mask_between("real_breakout_platform_amp_pct", max_value=cfg.get("max_platform_amp_pct"))
    if "max_platform_vol_ratio" in cfg:
        mask &= _mask_between("real_breakout_platform_vol_ratio", max_value=cfg.get("max_platform_vol_ratio"))
    if "max_platform_tail_vol_ratio" in cfg:
        mask &= _mask_between("real_breakout_platform_tail_vol_ratio", max_value=cfg.get("max_platform_tail_vol_ratio"))
    if "max_breakout_close_buffer_pct" in cfg:
        mask &= _mask_between("real_breakout_breakout_close_buffer_pct", max_value=cfg.get("max_breakout_close_buffer_pct"))
    if "min_breakout_volume_ratio" in cfg or "max_breakout_volume_ratio" in cfg:
        mask &= _mask_between("real_breakout_breakout_volume_ratio", min_value=cfg.get("min_breakout_volume_ratio"), max_value=cfg.get("max_breakout_volume_ratio"))
    if "max_current_buffer_pct" in cfg or "min_current_buffer_pct" in cfg:
        mask &= _mask_between("real_breakout_current_buffer_pct", min_value=cfg.get("min_current_buffer_pct"), max_value=cfg.get("max_current_buffer_pct"))
    if "min_ma20_slope_pct" in cfg:
        mask &= _mask_between("real_breakout_ma20_slope_pct", min_value=cfg.get("min_ma20_slope_pct"))
    if "min_base_score" in cfg:
        mask &= _mask_between("real_breakout_score", min_value=cfg.get("min_base_score"))
    return work[mask].copy().reset_index(drop=True)


def apply_real_breakout_research_entry_gate(row: dict[str, Any], market_regime: str, gate_config: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    cfg = dict(gate_config or {})
    if not cfg.get("enabled", False):
        return True, []
    failures: list[str] = []

    def _get_float(column: str) -> float | None:
        value = pd.to_numeric(row.get(column), errors="coerce")
        return None if pd.isna(value) else float(value)

    def _check_between(label: str, column: str, min_value: float | None = None, max_value: float | None = None) -> None:
        value = _get_float(column)
        if value is None:
            failures.append(f"{label}缺失")
            return
        if min_value is not None and value < float(min_value):
            failures.append(f"{label}<{float(min_value):g}")
        if max_value is not None and value > float(max_value):
            failures.append(f"{label}>{float(max_value):g}")

    allowed_market_regimes = [str(item) for item in cfg.get("allowed_market_regimes", []) if str(item)]
    if allowed_market_regimes and market_regime not in allowed_market_regimes:
        failures.append(f"市场状态不匹配:{market_regime}")
    allowed_stages = [str(item) for item in cfg.get("allowed_stages", []) if str(item)]
    stage = str(row.get("real_breakout_stage") or "")
    if allowed_stages and stage not in allowed_stages:
        failures.append(f"阶段不匹配:{stage}")
    if cfg.get("disallow_breakout_today", False) and bool(row.get("real_breakout_breakout_today")):
        failures.append("不做当日突破")
    if cfg.get("require_retest_or_follow", False) and stage not in {"retest_hold", "follow_through"}:
        failures.append("非回踩确认/右侧延续")
    if cfg.get("require_retest_ok", False) and not bool(row.get("real_breakout_retest_ok")):
        failures.append("回踩确认不足")
    _check_between("突破量比", "real_breakout_breakout_volume_ratio", cfg.get("min_breakout_volume_ratio"), cfg.get("max_breakout_volume_ratio"))
    _check_between("现价偏离平台", "real_breakout_current_buffer_pct", cfg.get("min_current_buffer_pct"), cfg.get("max_current_buffer_pct"))
    _check_between("平台缩量比", "real_breakout_platform_vol_ratio", cfg.get("min_platform_vol_ratio"), cfg.get("max_platform_vol_ratio"))
    _check_between("MA20斜率", "real_breakout_ma20_slope_pct", cfg.get("min_ma20_slope_pct"), cfg.get("max_ma20_slope_pct"))
    return len(failures) == 0, failures


def apply_plugin_entry_gate(plugin: StrategyPluginSpec, row: dict[str, Any], market_regime: str, config: dict[str, Any]) -> tuple[bool, list[str], bool]:
    gate_cfg = dict(config.get("_research_entry_gate") or {})
    if not gate_cfg.get("enabled", False):
        return True, [], False
    if plugin.strategy_id == "limitup_l1l2":
        passed, failures = apply_limitup_research_entry_gate(row=row, market_regime=market_regime, gate_config=gate_cfg)
        return passed, failures, True
    if plugin.strategy_id == "real_breakout":
        passed, failures = apply_real_breakout_research_entry_gate(row=row, market_regime=market_regime, gate_config=gate_cfg)
        return passed, failures, True
    return True, [], True


def build_limitup_candidates(window_history: pd.DataFrame, stock_basic_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pattern_snapshot = build_limitup_l1l2_snapshot(window_history, stock_basic_df=stock_basic_df, config=config)
    if pattern_snapshot.empty:
        return pd.DataFrame()
    latest_snapshot = local_latest_indicator_snapshot(window_history)
    candidates = stock_basic_df[["ts_code", "name", "industry", "market", "list_date"]].drop_duplicates("ts_code").copy()
    candidates = candidates.merge(latest_snapshot, on="ts_code", how="left")
    candidates = candidates.merge(pattern_snapshot, on="ts_code", how="inner")
    candidates = candidates[candidates["limitup_l1l2_signal"].fillna(False)].copy()
    if candidates.empty:
        return pd.DataFrame()
    tuning = dict(config.get("_research_limitup_tuning") or {})
    if tuning.get("enabled", False):
        candidates = apply_limitup_research_filters(candidates, tuning)
        if candidates.empty:
            return pd.DataFrame()
        candidates["strategy_rank_score"] = candidates.apply(lambda row: build_limitup_research_rank_score(row.to_dict(), tuning=tuning), axis=1)
    else:
        candidates["strategy_rank_score"] = candidates.apply(lambda row: build_limitup_rank_score(row.to_dict()), axis=1)
    return candidates.sort_values(
        ["strategy_rank_score", "limitup_l1l2_buy_signal", "limitup_l1l2_buy_recent", "limitup_l1l2_score", "volume_ratio"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def build_platform_candidates(window_history: pd.DataFrame, stock_basic_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pattern_snapshot = build_platform_breakout_snapshot(window_history, stock_basic_df=stock_basic_df, config=config)
    if pattern_snapshot.empty:
        return pd.DataFrame()
    latest_snapshot = local_latest_indicator_snapshot(window_history)
    candidates = stock_basic_df[["ts_code", "name", "industry", "market", "list_date"]].drop_duplicates("ts_code").copy()
    candidates = candidates.merge(latest_snapshot, on="ts_code", how="left")
    candidates = candidates.merge(pattern_snapshot, on="ts_code", how="inner")
    candidates["strategy_rank_score"] = pd.to_numeric(candidates["platform_breakout_score"], errors="coerce").fillna(0.0)
    candidates = candidates[candidates["platform_breakout_signal"].fillna(False)].copy()
    if candidates.empty:
        return pd.DataFrame()
    return candidates.sort_values(
        ["strategy_rank_score", "platform_breakout_current_volume_ratio", "platform_breakout_pullback_ratio", "platform_breakout_limit_volume_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_double_bottom_candidates(window_history: pd.DataFrame, stock_basic_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pattern_snapshot = build_double_bottom_snapshot(window_history, stock_basic_df=stock_basic_df, config=config)
    if pattern_snapshot.empty:
        return pd.DataFrame()
    latest_snapshot = local_latest_indicator_snapshot(window_history)
    candidates = stock_basic_df[["ts_code", "name", "industry", "market", "list_date"]].drop_duplicates("ts_code").copy()
    candidates = candidates.merge(latest_snapshot, on="ts_code", how="left")
    candidates = candidates.merge(pattern_snapshot, on="ts_code", how="inner")
    candidates["buy_type_rank"] = candidates["double_bottom_buy_type"].map({"B": 3, "A": 2, "C": 1}).fillna(0)
    candidates["strategy_rank_score"] = pd.to_numeric(candidates["double_bottom_score"], errors="coerce").fillna(0.0)
    candidates = candidates[candidates["double_bottom_signal"].fillna(False)].copy()
    if candidates.empty:
        return pd.DataFrame()
    return candidates.sort_values(
        ["strategy_rank_score", "buy_type_rank", "double_bottom_breakout_volume_ratio", "double_bottom_space_to_120_high_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_real_breakout_candidates(window_history: pd.DataFrame, stock_basic_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pattern_snapshot = build_real_breakout_snapshot(window_history, stock_basic_df=stock_basic_df, config=config)
    if pattern_snapshot.empty:
        return pd.DataFrame()
    candidates = stock_basic_df[["ts_code", "name", "industry", "market", "list_date"]].drop_duplicates("ts_code").copy()
    candidates = candidates.merge(pattern_snapshot, on="ts_code", how="inner")
    candidates = candidates[candidates["real_breakout_signal"].fillna(False)].copy()
    if candidates.empty:
        return pd.DataFrame()
    tuning = dict(config.get("_research_real_breakout_tuning") or {})
    if tuning.get("enabled", False):
        candidates = apply_real_breakout_research_filters(candidates, tuning)
        if candidates.empty:
            return pd.DataFrame()
        candidates["strategy_rank_score"] = candidates.apply(lambda row: build_real_breakout_research_rank_score(row.to_dict(), tuning=tuning), axis=1)
        candidates["stage_rank"] = candidates["real_breakout_stage"].map({"retest_hold": 3, "follow_through": 2, "breakout_today": 1}).fillna(0)
    else:
        candidates["stage_rank"] = candidates["real_breakout_stage"].map({"retest_hold": 3, "breakout_today": 2, "follow_through": 1}).fillna(0)
        candidates["strategy_rank_score"] = pd.to_numeric(candidates["real_breakout_score"], errors="coerce").fillna(0.0)
    return candidates.sort_values(
        ["strategy_rank_score", "stage_rank", "real_breakout_current_buffer_pct", "real_breakout_breakout_volume_ratio", "real_breakout_platform_vol_ratio"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)


def build_price_strategy_registry() -> dict[str, StrategyPluginSpec]:
    return {
        "limitup_l1l2": StrategyPluginSpec(
            strategy_id="limitup_l1l2",
            strategy_name="龙门双阶强势臻选",
            history_bars=100,
            build_candidates=build_limitup_candidates,
            recommended_history_years=2.5,
        ),
        "platform_breakout": StrategyPluginSpec(
            strategy_id="platform_breakout",
            strategy_name="天衡回踩转强臻选",
            history_bars=60,
            build_candidates=build_platform_candidates,
            recommended_history_years=2.5,
        ),
        "double_bottom": StrategyPluginSpec(
            strategy_id="double_bottom",
            strategy_name="玄枢双底反转臻选",
            history_bars=170,
            build_candidates=build_double_bottom_candidates,
            recommended_history_years=3.0,
        ),
        "real_breakout": StrategyPluginSpec(
            strategy_id="real_breakout",
            strategy_name="真实资金突破臻选",
            history_bars=120,
            build_candidates=build_real_breakout_candidates,
            recommended_history_years=2.5,
        ),
    }


def supported_price_strategy_ids() -> list[str]:
    return sorted(build_price_strategy_registry().keys())
