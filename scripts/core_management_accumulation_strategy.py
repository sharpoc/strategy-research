from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from holder_strategy_core import (
    build_qfq_daily,
    clip_score,
    ensure_columns,
    filter_frame_as_of,
    json_safe,
    normalize_trade_day,
    rank_pct,
    to_bool,
    to_float,
    to_number,
)


STRATEGY_ID = "core_management_accumulation"
STRATEGY_NAME = "核心高管连增臻选"


@dataclass(frozen=True)
class CoreManagementAccumulationConfig:
    ann_start_date: str
    end_date: str
    event_chunk_days: int = 5
    price_lookback_days: int = 250
    moneyflow_lookback_days: int = 5
    recent_wave_trade_days: int = 15
    wave_split_max_gap_trade_days: int = 5
    min_wave_trade_days: int = 2
    min_dense_wave_event_count: int = 3
    min_dense_core_holder_count: int = 2
    min_wave_total_amount: float = 10_000_000.0
    min_dense_wave_total_amount: float = 5_000_000.0
    min_list_days: int = 120
    min_price: float = 3.0
    min_avg_amount_20d_yuan: float = 100_000_000.0
    min_current_to_cost: float = 0.92
    max_current_to_cost: float = 1.10
    max_deep_dive_stocks: int = 120
    post_wave_lookback_bars: int = 10
    post_wave_recent_signal_bars: int = 3
    min_total_score: float = 60.0
    repeat_signal_cooldown_trade_days: int = 3
    max_fresh_wave_age_trade_days: int = 5
    min_retrigger_structure_score: float = 16.0
    min_final_confirmation_score: float = 6.0
    include_star: bool = False
    include_gem: bool = True

    @classmethod
    def for_end_date(
        cls,
        end_date: str,
        ann_start_date: str = "",
        **overrides: Any,
    ) -> "CoreManagementAccumulationConfig":
        end_str = normalize_trade_day(end_date)
        ann_str = normalize_trade_day(ann_start_date) if ann_start_date else ""
        if not ann_str:
            ann_str = (pd.Timestamp(end_str) - pd.Timedelta(days=45)).strftime("%Y%m%d")
        return cls(ann_start_date=ann_str, end_date=end_str, **overrides)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoreManagementAccumulationConfig":
        return cls(
            ann_start_date=normalize_trade_day(data.get("ann_start_date", "")),
            end_date=normalize_trade_day(data.get("end_date", "")),
            event_chunk_days=int(data.get("event_chunk_days", 5)),
            price_lookback_days=int(data.get("price_lookback_days", 250)),
            moneyflow_lookback_days=int(data.get("moneyflow_lookback_days", 5)),
            recent_wave_trade_days=int(data.get("recent_wave_trade_days", 15)),
            wave_split_max_gap_trade_days=int(data.get("wave_split_max_gap_trade_days", 5)),
            min_wave_trade_days=int(data.get("min_wave_trade_days", 2)),
            min_dense_wave_event_count=int(data.get("min_dense_wave_event_count", 3)),
            min_dense_core_holder_count=int(data.get("min_dense_core_holder_count", 2)),
            min_wave_total_amount=float(data.get("min_wave_total_amount", 10_000_000.0)),
            min_dense_wave_total_amount=float(data.get("min_dense_wave_total_amount", 5_000_000.0)),
            min_list_days=int(data.get("min_list_days", 120)),
            min_price=float(data.get("min_price", 3.0)),
            min_avg_amount_20d_yuan=float(data.get("min_avg_amount_20d_yuan", 100_000_000.0)),
            min_current_to_cost=float(data.get("min_current_to_cost", 0.92)),
            max_current_to_cost=float(data.get("max_current_to_cost", 1.10)),
            max_deep_dive_stocks=int(data.get("max_deep_dive_stocks", 120)),
            post_wave_lookback_bars=int(data.get("post_wave_lookback_bars", 10)),
            post_wave_recent_signal_bars=int(data.get("post_wave_recent_signal_bars", 3)),
            min_total_score=float(data.get("min_total_score", 60.0)),
            repeat_signal_cooldown_trade_days=int(data.get("repeat_signal_cooldown_trade_days", 3)),
            max_fresh_wave_age_trade_days=int(data.get("max_fresh_wave_age_trade_days", 5)),
            min_retrigger_structure_score=float(data.get("min_retrigger_structure_score", 16.0)),
            min_final_confirmation_score=float(data.get("min_final_confirmation_score", 6.0)),
            include_star=to_bool(data.get("include_star", False)),
            include_gem=to_bool(data.get("include_gem", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_margin_summary(margin_detail_df: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
    if margin_detail_df.empty or "ts_code" not in margin_detail_df.columns:
        return pd.DataFrame()
    work = margin_detail_df.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work["rzmre"] = pd.to_numeric(work.get("rzmre"), errors="coerce").fillna(0.0)
    work["rzche"] = pd.to_numeric(work.get("rzche"), errors="coerce").fillna(0.0)
    work["rzye"] = pd.to_numeric(work.get("rzye"), errors="coerce").fillna(0.0)
    work["margin_net_buy"] = work["rzmre"] - work["rzche"]
    recent_dates = sorted({str(value) for value in trade_dates if str(value)})
    if recent_dates:
        work = work[work["trade_date"].isin(recent_dates)].copy()
    if work.empty:
        return pd.DataFrame()

    latest_trade_date = max(recent_dates) if recent_dates else work["trade_date"].max()
    rows: list[dict[str, Any]] = []
    for ts_code, sub in work.groupby("ts_code", dropna=False):
        ordered = sub.sort_values("trade_date").reset_index(drop=True)
        latest = ordered.iloc[-1]
        positive_days = int((ordered["margin_net_buy"] > 0).sum())
        rows.append(
            {
                "ts_code": ts_code,
                "margin_net_buy_3d": to_number(ordered["margin_net_buy"].sum(), 0),
                "margin_positive_days_3d": positive_days,
                "margin_balance_latest": to_number(latest.get("rzye"), 0) if str(latest.get("trade_date")) == latest_trade_date else None,
            }
        )
    return pd.DataFrame(rows)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    raw = str(text or "")
    return any(keyword in raw for keyword in keywords)


def classify_holder_identity(holder_type: Any, holder_name: Any) -> dict[str, Any]:
    holder_type_text = str(holder_type or "").strip().upper()
    holder_name_text = str(holder_name or "").strip()
    if _contains_any(holder_name_text, ("董事长", "实际控制人", "实控人", "控股股东")):
        return {
            "identity_label": "核心控制层",
            "identity_bucket": "core_control",
            "identity_weight": 1.00,
            "core_management_flag": True,
        }
    if _contains_any(holder_name_text, ("总经理", "总裁", "董事")):
        return {
            "identity_label": "核心经营层",
            "identity_bucket": "core_exec",
            "identity_weight": 0.85,
            "core_management_flag": True,
        }
    if _contains_any(holder_name_text, ("副总经理", "副总", "财务总监", "董秘", "高管")):
        return {
            "identity_label": "高级管理层",
            "identity_bucket": "senior_exec",
            "identity_weight": 0.70,
            "core_management_flag": True,
        }
    if holder_type_text == "G":
        return {
            "identity_label": "高管(G)",
            "identity_bucket": "exec_default",
            "identity_weight": 0.75,
            "core_management_flag": True,
        }
    if holder_type_text == "C":
        return {
            "identity_label": "公司股东",
            "identity_bucket": "corporate_holder",
            "identity_weight": 0.35,
            "core_management_flag": False,
        }
    if holder_type_text == "P":
        return {
            "identity_label": "自然人股东",
            "identity_bucket": "person_holder",
            "identity_weight": 0.20,
            "core_management_flag": False,
        }
    return {
        "identity_label": "其他",
        "identity_bucket": "other",
        "identity_weight": 0.0,
        "core_management_flag": False,
    }


def _trade_day_position(trade_dates: list[str], date_str: str) -> int:
    ordered = sorted(trade_dates)
    if not ordered:
        return 0
    normalized = normalize_trade_day(date_str)
    for idx, value in enumerate(ordered):
        if value >= normalized:
            return idx
    return len(ordered) - 1


def _board_allowed(ts_code: str, market: Any, include_star: bool, include_gem: bool) -> bool:
    market_text = str(market or "")
    code = str(ts_code or "")
    if code.endswith(".BJ") or "北交所" in market_text:
        return False
    if (market_text == "科创板" or code.startswith(("688", "689"))) and not include_star:
        return False
    if (market_text == "创业板" or code.startswith(("300", "301"))) and not include_gem:
        return False
    return market_text in {"主板", "创业板"} or code.startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605"))


def _is_st_name(name: Any) -> bool:
    text = str(name or "").upper()
    return "ST" in text


def _build_identity_strength_score(row: dict[str, Any]) -> float:
    role_component = (to_float(row.get("wave_best_identity_weight")) or 0.0) * 12.0
    amount = to_float(row.get("wave_total_amount")) or 0.0
    ratio = to_float(row.get("wave_total_change_ratio")) or 0.0
    if amount >= 50_000_000:
        amount_component = 9.0
    elif amount >= 20_000_000:
        amount_component = 7.0
    elif amount >= 10_000_000:
        amount_component = 5.0
    elif amount >= 5_000_000:
        amount_component = 2.0
    else:
        amount_component = 0.0
    if ratio >= 1.0:
        ratio_component = 4.0
    elif ratio >= 0.3:
        ratio_component = 3.0
    elif ratio > 0:
        ratio_component = 1.5
    else:
        ratio_component = 0.0
    return round(clip_score(role_component + amount_component + ratio_component, 0.0, 25.0), 2)


def _build_continuity_score(row: dict[str, Any]) -> float:
    trade_days = int(to_float(row.get("wave_trade_days")) or 0)
    event_count = int(to_float(row.get("wave_event_count")) or 0)
    core_event_count = int(to_float(row.get("wave_core_management_event_count")) or 0)
    core_holder_count = int(to_float(row.get("wave_core_holder_count")) or 0)
    span_days = int(to_float(row.get("wave_span_trade_days")) or 0)
    score = 0.0
    if trade_days >= 4:
        score += 14.0
    elif trade_days == 3:
        score += 12.0
    elif trade_days == 2:
        score += 8.0
    elif event_count >= 6 and core_holder_count >= 2:
        score += 8.0
    elif event_count >= 4 and core_holder_count >= 2:
        score += 6.0
    elif event_count >= 3 and core_event_count >= 2:
        score += 4.0
    if span_days <= 5:
        score += 4.0
    elif span_days <= 8:
        score += 3.0
    elif span_days <= 12:
        score += 2.0
    if core_holder_count >= 3:
        score += 2.0
    elif event_count >= 5:
        score += 2.0
    elif event_count >= 3:
        score += 1.0
    return round(clip_score(score, 0.0, 20.0), 2)


def _build_cost_zone_score(row: dict[str, Any]) -> float:
    ratio = to_float(row.get("current_to_cost_ratio"))
    if ratio is None:
        return 0.0
    if 0.98 <= ratio <= 1.03:
        score = 25.0
    elif 0.95 <= ratio <= 1.05:
        score = 23.0
    elif 0.95 <= ratio <= 1.08:
        score = 20.0
    elif 0.92 <= ratio <= 1.10:
        score = 14.0
    else:
        score = 0.0
    return round(score, 2)


def _build_aux_flow_health_score(row: dict[str, Any]) -> float:
    score = 5.0
    margin_net_buy_3d = to_float(row.get("margin_net_buy_3d"))
    margin_positive_days_3d = to_float(row.get("margin_positive_days_3d"))
    main_net_amount_3d = to_float(row.get("main_net_amount_3d"))
    main_net_amount_5d = to_float(row.get("main_net_amount_5d"))
    turnover_rate_f = to_float(row.get("turnover_rate_f")) or to_float(row.get("turnover_rate"))
    volume_ratio = to_float(row.get("volume_ratio"))

    if margin_net_buy_3d is not None:
        if margin_net_buy_3d > 0 and (margin_positive_days_3d or 0) >= 2:
            score += 1.5
        elif margin_net_buy_3d > 0:
            score += 0.8
        elif margin_net_buy_3d < 0:
            score -= 1.2

    if main_net_amount_3d is not None or main_net_amount_5d is not None:
        if (main_net_amount_3d or 0) > 0 and (main_net_amount_5d or 0) > 0:
            score += 2.0
        elif (main_net_amount_3d or 0) > 0 or (main_net_amount_5d or 0) > 0:
            score += 1.0
        elif (main_net_amount_3d or 0) < 0 and (main_net_amount_5d or 0) < 0:
            score -= 2.0

    if turnover_rate_f is not None:
        if 1.0 <= turnover_rate_f <= 8.0 and ((volume_ratio or 0) >= 0.9):
            score += 1.5
        elif 0.5 <= turnover_rate_f <= 15.0:
            score += 0.5
        elif turnover_rate_f < 0.3 or turnover_rate_f > 25.0:
            score -= 1.5

    return round(clip_score(score, 0.0, 10.0), 2)


def build_wave_signature(row: dict[str, Any]) -> str:
    weighted_cost = to_float(row.get("wave_buy_avg_price_weighted"))
    weighted_cost_rounded = round(weighted_cost, 2) if weighted_cost is not None else "NA"
    parts = [
        str(row.get("ts_code") or ""),
        str(row.get("wave_first_date") or ""),
        str(row.get("wave_last_date") or ""),
        str(int(to_float(row.get("wave_event_count")) or 0)),
        str(weighted_cost_rounded),
    ]
    return "|".join(parts)


def _trade_day_distance_for_signal(signal_date: Any, latest_trade_date: Any) -> int | None:
    signal = normalize_trade_day(signal_date)
    latest = normalize_trade_day(latest_trade_date)
    if not signal or not latest:
        return None
    try:
        signal_ts = pd.Timestamp(signal)
        latest_ts = pd.Timestamp(latest)
    except Exception:
        return None
    if latest_ts < signal_ts:
        return 0
    days = pd.bdate_range(signal_ts, latest_ts)
    return max(int(len(days) - 1), 0)


def _build_freshness_penalty_score(row: dict[str, Any], config: CoreManagementAccumulationConfig) -> float:
    age = to_float(row.get("wave_age_trade_days"))
    if age is None:
        return 0.0
    strong_hold = bool(
        to_bool(row.get("above_ma20"))
        and to_bool(row.get("ma10_slope_up"))
        and (to_float(row.get("post_wave_structure_score")) or 0.0) >= config.min_retrigger_structure_score
    )
    if age <= 1:
        return 0.0
    if age <= 3:
        return 1.5
    if age <= 5:
        return 3.5
    if strong_hold:
        return 4.5
    if to_bool(row.get("recent_restrengthen_flag")):
        return 6.0
    return 8.0


def _build_retrigger_quality_score(row: dict[str, Any], config: CoreManagementAccumulationConfig) -> float:
    age = to_float(row.get("wave_age_trade_days"))
    structure_score = to_float(row.get("post_wave_structure_score")) or 0.0
    recent_restrengthen = to_bool(row.get("recent_restrengthen_flag"))
    above_ma10 = to_bool(row.get("above_ma10"))
    above_ma20 = to_bool(row.get("above_ma20"))
    ma10_slope_up = to_bool(row.get("ma10_slope_up"))

    score = 0.0
    if age is not None and age <= 1:
        score += 1.5
    elif age is not None and age <= 3:
        score += 0.8
    if above_ma10:
        score += 2.0
    if ma10_slope_up:
        score += 1.5
    if above_ma20:
        score += 1.5
    if recent_restrengthen:
        score += 3.0
    if recent_restrengthen and above_ma20:
        score += 1.5
    if structure_score >= config.min_retrigger_structure_score:
        score += 1.5
    return round(clip_score(score, 0.0, 12.0), 2)


def build_repeat_signal_state(
    row: dict[str, Any],
    config: CoreManagementAccumulationConfig,
    recent_final_signals: pd.DataFrame | None = None,
) -> dict[str, Any]:
    default = {
        "wave_signature": build_wave_signature(row),
        "repeat_penalty_score": 0.0,
        "repeat_recent_signal_hit": False,
        "repeat_signal_distance_trade_days": None,
        "repeat_allowed_override": False,
        "repeat_signal_blocked": False,
    }
    if recent_final_signals is None or recent_final_signals.empty:
        return default

    signature = default["wave_signature"]
    work = recent_final_signals.copy()
    if "wave_signature" not in work.columns:
        return default
    matched = work[work["wave_signature"].astype(str) == signature].copy()
    if matched.empty:
        return default
    if "signal_date" not in matched.columns:
        return default
    matched["signal_date"] = matched["signal_date"].astype(str)
    matched = matched.sort_values("signal_date").reset_index(drop=True)
    latest_repeat = matched.iloc[-1].to_dict()
    distance = _trade_day_distance_for_signal(latest_repeat.get("signal_date"), row.get("latest_trade_date"))

    recent_restrengthen = to_bool(row.get("recent_restrengthen_flag"))
    above_ma10 = to_bool(row.get("above_ma10"))
    above_ma20 = to_bool(row.get("above_ma20"))
    ma10_slope_up = to_bool(row.get("ma10_slope_up"))
    structure_score = to_float(row.get("post_wave_structure_score")) or 0.0
    base_total_score = to_float(row.get("base_total_score"))
    wave_age_trade_days = to_float(row.get("wave_age_trade_days"))
    has_new_event = normalize_trade_day(row.get("wave_last_date")) != normalize_trade_day(latest_repeat.get("wave_last_date"))
    last_structure_score = to_float(latest_repeat.get("post_wave_structure_score")) or 0.0
    last_base_total_score = to_float(latest_repeat.get("base_total_score"))
    if last_base_total_score is None:
        last_base_total_score = to_float(latest_repeat.get("adjusted_total_score"))
    retrigger_override = bool(
        recent_restrengthen
        and above_ma10
        and above_ma20
        and ma10_slope_up
        and wave_age_trade_days is not None
        and wave_age_trade_days <= config.max_fresh_wave_age_trade_days
        and structure_score >= (config.min_retrigger_structure_score + 2.0)
        and structure_score >= (last_structure_score + 1.0)
        and (
            last_base_total_score is None
            or (base_total_score is not None and base_total_score >= (last_base_total_score + 2.0))
        )
    )
    override = bool(has_new_event or retrigger_override)
    if distance is None:
        return default | {
            "repeat_recent_signal_hit": True,
            "repeat_allowed_override": override,
        }
    recent_hit = distance <= config.repeat_signal_cooldown_trade_days
    if not override:
        penalty = 7.0 if recent_hit else 4.0
        blocked = True
    else:
        penalty = 2.5 if recent_hit else 1.5
        blocked = False
    return {
        "wave_signature": signature,
        "repeat_penalty_score": penalty,
        "repeat_recent_signal_hit": recent_hit,
        "repeat_signal_distance_trade_days": distance,
        "repeat_allowed_override": override,
        "repeat_signal_blocked": blocked,
    }


def build_event_wave_details(
    holdertrade_df: pd.DataFrame,
    stock_basic_df: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    trade_dates: list[str],
    config: CoreManagementAccumulationConfig,
    latest_trade_date: str,
) -> pd.DataFrame:
    work = filter_frame_as_of(holdertrade_df, latest_trade_date, ("ann_date",))
    if work.empty:
        return pd.DataFrame()
    work = work[work.get("in_de", "").fillna("").astype(str).str.upper() == "IN"].copy()
    if work.empty:
        return pd.DataFrame()

    recent_trade_dates = sorted(set(trade_dates))[-config.recent_wave_trade_days :]
    if not recent_trade_dates:
        return pd.DataFrame()
    recent_start = recent_trade_dates[0]
    work["ann_date"] = work["ann_date"].astype(str)
    work = work[(work["ann_date"] >= recent_start) & (work["ann_date"] <= latest_trade_date)].copy()
    if work.empty:
        return pd.DataFrame()

    for col in ["change_vol", "change_ratio", "avg_price", "after_ratio"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["holder_type"] = work.get("holder_type", "").fillna("").astype(str)
    work["holder_name"] = work.get("holder_name", "").fillna("").astype(str)
    work["ann_date_dt"] = pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce")
    work = work.dropna(subset=["ann_date_dt"]).copy()

    identity_rows = work.apply(lambda row: classify_holder_identity(row.get("holder_type"), row.get("holder_name")), axis=1, result_type="expand")
    work = pd.concat([work, identity_rows], axis=1)
    work["event_amount"] = pd.to_numeric(work["change_vol"], errors="coerce").fillna(0.0) * pd.to_numeric(work["avg_price"], errors="coerce").fillna(0.0)

    basic_cols = [c for c in ["ts_code", "symbol", "name", "industry", "market", "list_date"] if c in stock_basic_df.columns]
    work = work.merge(stock_basic_df[basic_cols].drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")
    market_cols = [
        c
        for c in [
            "ts_code",
            "trade_date",
            "close",
            "close_qfq",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "main_net_amount_3d",
            "main_net_amount_5d",
            "main_net_positive_days_3d",
            "main_net_positive_days_5d",
            "main_net_consecutive_days",
            "industry_pb_pct_rank",
            "margin_net_buy_3d",
            "margin_positive_days_3d",
            "margin_balance_latest",
        ]
        if c in market_snapshot.columns
    ]
    work = work.merge(market_snapshot[market_cols].drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")

    rows: list[dict[str, Any]] = []
    ordered_trade_dates = sorted(set(trade_dates))
    for ts_code, sub in work.groupby("ts_code", dropna=False):
        ordered = sub.sort_values(["ann_date_dt", "event_amount", "change_vol"], ascending=[True, False, False]).reset_index(drop=True)
        if ordered.empty:
            continue

        wave_groups: list[list[int]] = []
        current_wave: list[int] = []
        previous_pos: int | None = None
        for idx, row in ordered.iterrows():
            position = _trade_day_position(ordered_trade_dates, str(row.get("ann_date", "")))
            if previous_pos is None or position - previous_pos <= config.wave_split_max_gap_trade_days:
                current_wave.append(idx)
            else:
                wave_groups.append(current_wave)
                current_wave = [idx]
            previous_pos = position
        if current_wave:
            wave_groups.append(current_wave)

        latest_close = to_float(ordered.iloc[-1].get("close"))
        name = ordered.iloc[-1].get("name")
        market = ordered.iloc[-1].get("market")
        list_date = normalize_trade_day(ordered.iloc[-1].get("list_date", ""))
        listing_days = None
        if list_date:
            listing_days = int((pd.Timestamp(latest_trade_date) - pd.Timestamp(list_date)).days)

        for wave_idx, index_group in enumerate(wave_groups, start=1):
            wave_df = ordered.iloc[index_group].copy().sort_values(["ann_date_dt", "event_amount"], ascending=[True, False]).reset_index(drop=True)
            wave_dates = wave_df["ann_date"].dropna().astype(str).tolist()
            wave_trade_positions = sorted({_trade_day_position(ordered_trade_dates, value) for value in wave_dates})
            valid_amount_mask = pd.to_numeric(wave_df["event_amount"], errors="coerce").fillna(0.0) > 0
            total_amount = pd.to_numeric(wave_df.loc[valid_amount_mask, "event_amount"], errors="coerce").fillna(0.0).sum()
            total_vol = pd.to_numeric(wave_df.loc[valid_amount_mask, "change_vol"], errors="coerce").fillna(0.0).sum()
            weighted_cost = None
            if total_amount > 0 and total_vol > 0:
                weighted_cost = total_amount / total_vol
            else:
                weighted_cost = pd.to_numeric(wave_df["avg_price"], errors="coerce").dropna().mean()
            best_identity_idx = wave_df["identity_weight"].astype(float).idxmax()
            best_identity = wave_df.loc[best_identity_idx]
            current_to_cost_ratio = None
            if latest_close and weighted_cost and weighted_cost > 0:
                current_to_cost_ratio = latest_close / weighted_cost

            row = {
                "ts_code": ts_code,
                "name": name,
                "industry": wave_df.iloc[-1].get("industry"),
                "market": market,
                "list_date": list_date,
                "listing_days": listing_days,
                "is_st": _is_st_name(name),
                "board_allowed": _board_allowed(str(ts_code), market, config.include_star, config.include_gem),
                "wave_id": f"{ts_code}_{wave_idx}",
                "wave_first_date": wave_dates[0],
                "wave_last_date": wave_dates[-1],
                "wave_event_count": int(len(wave_df)),
                "wave_trade_days": int(len(wave_trade_positions)),
                "wave_span_trade_days": int((wave_trade_positions[-1] - wave_trade_positions[0] + 1) if wave_trade_positions else 1),
                "wave_holder_count": int(wave_df["holder_name"].nunique()),
                "wave_core_holder_count": int(wave_df.loc[wave_df["core_management_flag"].astype(bool), "holder_name"].nunique()),
                "wave_core_management_event_count": int(pd.to_numeric(wave_df["core_management_flag"], errors="coerce").fillna(False).astype(bool).sum()),
                "wave_total_change_vol": pd.to_numeric(wave_df["change_vol"], errors="coerce").fillna(0.0).sum(),
                "wave_total_change_ratio": pd.to_numeric(wave_df["change_ratio"], errors="coerce").fillna(0.0).sum(),
                "wave_total_amount": total_amount,
                "wave_buy_avg_price_weighted": weighted_cost,
                "wave_best_identity_label": best_identity.get("identity_label"),
                "wave_best_identity_bucket": best_identity.get("identity_bucket"),
                "wave_best_identity_weight": to_float(best_identity.get("identity_weight")),
                "holder_preview": " / ".join([value for value in wave_df["holder_name"].astype(str).tolist()[:3] if value]),
                "current_price": latest_close,
                "current_to_cost_ratio": current_to_cost_ratio,
                "current_to_cost_pct": to_number((current_to_cost_ratio - 1.0) * 100 if current_to_cost_ratio is not None else None),
                "trade_date": wave_df.iloc[-1].get("trade_date"),
                "turnover_rate": to_float(wave_df.iloc[-1].get("turnover_rate")),
                "turnover_rate_f": to_float(wave_df.iloc[-1].get("turnover_rate_f")),
                "volume_ratio": to_float(wave_df.iloc[-1].get("volume_ratio")),
                "main_net_amount_3d": to_float(wave_df.iloc[-1].get("main_net_amount_3d")),
                "main_net_amount_5d": to_float(wave_df.iloc[-1].get("main_net_amount_5d")),
                "main_net_positive_days_3d": to_float(wave_df.iloc[-1].get("main_net_positive_days_3d")),
                "main_net_positive_days_5d": to_float(wave_df.iloc[-1].get("main_net_positive_days_5d")),
                "main_net_consecutive_days": to_float(wave_df.iloc[-1].get("main_net_consecutive_days")),
                "margin_net_buy_3d": to_float(wave_df.iloc[-1].get("margin_net_buy_3d")),
                "margin_positive_days_3d": to_float(wave_df.iloc[-1].get("margin_positive_days_3d")),
            }
            row["identity_strength_score"] = _build_identity_strength_score(row)
            row["continuity_score"] = _build_continuity_score(row)
            row["cost_zone_score"] = _build_cost_zone_score(row)
            row["aux_flow_health_score"] = _build_aux_flow_health_score(row)
            row["preliminary_score"] = round(
                row["identity_strength_score"] + row["continuity_score"] + row["cost_zone_score"] + row["aux_flow_health_score"],
                2,
            )
            rows.append(row)
    details = pd.DataFrame(rows)
    if details.empty:
        return details
    details = details.sort_values(
        ["wave_last_date", "preliminary_score", "wave_total_amount", "wave_trade_days"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return details


def select_best_wave_per_stock(wave_details: pd.DataFrame) -> pd.DataFrame:
    if wave_details.empty:
        return wave_details.copy()
    ordered = wave_details.sort_values(
        ["ts_code", "wave_last_date", "preliminary_score", "wave_total_amount"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return ordered.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)


def build_preliminary_candidate_flags(row: dict[str, Any], config: CoreManagementAccumulationConfig) -> dict[str, Any]:
    current_price = to_float(row.get("current_price"))
    listing_days = to_float(row.get("listing_days"))
    current_to_cost_ratio = to_float(row.get("current_to_cost_ratio"))
    wave_trade_days = int(to_float(row.get("wave_trade_days")) or 0)
    wave_event_count = int(to_float(row.get("wave_event_count")) or 0)
    wave_core_holder_count = int(to_float(row.get("wave_core_holder_count")) or 0)
    dense_core_wave = wave_event_count >= config.min_dense_wave_event_count and wave_core_holder_count >= config.min_dense_core_holder_count
    continuity_ok = wave_trade_days >= config.min_wave_trade_days or dense_core_wave
    total_amount = to_float(row.get("wave_total_amount")) or 0.0
    amount_ok = total_amount >= config.min_wave_total_amount or (dense_core_wave and total_amount >= config.min_dense_wave_total_amount)
    return {
        "board_ok": to_bool(row.get("board_allowed")),
        "st_ok": not to_bool(row.get("is_st")),
        "listing_ok": listing_days is not None and listing_days >= config.min_list_days,
        "price_ok": current_price is not None and current_price >= config.min_price,
        "wave_days_ok": continuity_ok,
        "core_event_ok": (to_float(row.get("wave_core_management_event_count")) or 0.0) >= 1.0,
        "wave_amount_ok": amount_ok,
        "cost_zone_ok": current_to_cost_ratio is not None and config.min_current_to_cost <= current_to_cost_ratio <= config.max_current_to_cost,
    }


def build_post_wave_structure_metrics(
    daily_df: pd.DataFrame,
    adj_df: pd.DataFrame,
    end_date: str,
    wave_first_date: str,
    wave_last_date: str,
    weighted_cost: float | None,
    config: CoreManagementAccumulationConfig,
) -> dict[str, Any]:
    daily = filter_frame_as_of(daily_df, end_date, ("trade_date",))
    if daily.empty:
        return {
            "avg_amount_20d_yuan": None,
            "latest_close_raw": None,
            "latest_trade_date": None,
            "above_ma5": False,
            "above_ma10": False,
            "above_ma20": False,
            "ma10_slope_up": False,
            "recent_restrengthen_flag": False,
            "post_wave_breakdown_flag": True,
            "post_wave_low_to_cost_pct": None,
            "wave_age_trade_days": None,
            "post_wave_structure_score": 0.0,
        }

    work = daily.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work["amount"] = pd.to_numeric(work.get("amount"), errors="coerce")
    work["open"] = pd.to_numeric(work.get("open"), errors="coerce")
    work["high"] = pd.to_numeric(work.get("high"), errors="coerce")
    work["low"] = pd.to_numeric(work.get("low"), errors="coerce")
    work["close"] = pd.to_numeric(work.get("close"), errors="coerce")
    work = work.sort_values("trade_date").reset_index(drop=True)

    qfq = build_qfq_daily(work, adj_df)
    qfq["trade_date"] = qfq["trade_date"].astype(str)
    qfq = qfq.sort_values("trade_date").reset_index(drop=True)
    qfq["ma5"] = qfq["close_qfq_calc"].rolling(window=5, min_periods=3).mean()
    qfq["ma10"] = qfq["close_qfq_calc"].rolling(window=10, min_periods=5).mean()
    qfq["ma20"] = qfq["close_qfq_calc"].rolling(window=20, min_periods=10).mean()
    qfq["prev_close_raw"] = qfq["close"].shift(1)
    qfq["pct_chg_raw"] = np.where(
        pd.to_numeric(qfq["prev_close_raw"], errors="coerce").fillna(0.0) != 0,
        (pd.to_numeric(qfq["close"], errors="coerce") / pd.to_numeric(qfq["prev_close_raw"], errors="coerce") - 1.0) * 100.0,
        np.nan,
    )
    qfq["avg_amount_5"] = qfq["amount"].shift(1).rolling(window=5, min_periods=3).mean()

    latest = qfq.iloc[-1]
    recent_20 = qfq.tail(20)
    avg_amount_20d_yuan = None
    if not recent_20["amount"].dropna().empty:
        avg_amount_20d_yuan = float(recent_20["amount"].dropna().mean()) * 1000.0

    post_wave = qfq[qfq["trade_date"] >= normalize_trade_day(wave_first_date)].copy()
    if post_wave.empty:
        post_wave = qfq.tail(config.post_wave_lookback_bars).copy()
    else:
        post_wave = post_wave.tail(config.post_wave_lookback_bars).copy()

    above_ma5 = bool(to_float(latest.get("close_qfq_calc")) is not None and to_float(latest.get("ma5")) is not None and to_float(latest.get("close_qfq_calc")) > to_float(latest.get("ma5")))
    above_ma10 = bool(to_float(latest.get("close_qfq_calc")) is not None and to_float(latest.get("ma10")) is not None and to_float(latest.get("close_qfq_calc")) > to_float(latest.get("ma10")))
    above_ma20 = bool(to_float(latest.get("close_qfq_calc")) is not None and to_float(latest.get("ma20")) is not None and to_float(latest.get("close_qfq_calc")) > to_float(latest.get("ma20")))
    prev_ma10 = to_float(qfq.iloc[-2].get("ma10")) if len(qfq) >= 2 else None
    ma10_now = to_float(latest.get("ma10"))
    ma10_slope_up = bool(ma10_now is not None and prev_ma10 is not None and ma10_now >= prev_ma10)

    recent_signal = post_wave.tail(config.post_wave_recent_signal_bars).copy()
    recent_restrengthen_flag = False
    if len(recent_signal) >= 2:
        latest_close = to_float(recent_signal.iloc[-1].get("close_qfq_calc"))
        previous_close = to_float(recent_signal.iloc[-2].get("close_qfq_calc"))
        prev_high_raw = to_float(recent_signal.iloc[-2].get("high"))
        latest_close_raw = to_float(recent_signal.iloc[-1].get("close"))
        recent_restrengthen_flag = bool(
            (latest_close is not None and previous_close is not None and latest_close > previous_close and above_ma5)
            or (latest_close_raw is not None and prev_high_raw is not None and latest_close_raw > prev_high_raw)
        )

    breakdown_days = post_wave[
        (pd.to_numeric(post_wave["pct_chg_raw"], errors="coerce").fillna(0.0) <= -6.0)
        & (
            pd.to_numeric(post_wave["amount"], errors="coerce").fillna(0.0)
            >= pd.to_numeric(post_wave["avg_amount_5"], errors="coerce").fillna(np.inf) * 1.5
        )
    ]
    latest_close_qfq = to_float(latest.get("close_qfq_calc"))
    latest_ma20 = to_float(latest.get("ma20"))
    latest_close_raw = to_float(latest.get("close"))
    low_since_wave = to_float(pd.to_numeric(post_wave["low"], errors="coerce").min()) if not post_wave.empty else None
    low_to_cost_pct = None
    if low_since_wave is not None and weighted_cost not in (None, 0):
        low_to_cost_pct = (low_since_wave / weighted_cost - 1.0) * 100.0
    wave_age_trade_days = int(
        (
            (qfq["trade_date"].astype(str) > normalize_trade_day(wave_last_date))
            & (qfq["trade_date"].astype(str) <= normalize_trade_day(end_date))
        ).sum()
    )

    post_wave_breakdown_flag = bool(
        (not breakdown_days.empty)
        or (latest_close_qfq is not None and latest_ma20 is not None and latest_close_qfq < latest_ma20 * 0.97)
        or (latest_close_raw is not None and weighted_cost not in (None, 0) and latest_close_raw < weighted_cost * 0.92)
    )

    score = 0.0
    if above_ma5:
        score += 4.0
    if above_ma10:
        score += 4.0
    if above_ma20:
        score += 3.0
    if ma10_slope_up:
        score += 2.0
    if not post_wave_breakdown_flag:
        score += 3.0
    if recent_restrengthen_flag:
        score += 4.0
    post_wave_structure_score = round(clip_score(score, 0.0, 20.0), 2)

    return {
        "avg_amount_20d_yuan": to_number(avg_amount_20d_yuan, 0),
        "latest_close_raw": to_number(latest_close_raw),
        "latest_trade_date": latest.get("trade_date"),
        "above_ma5": above_ma5,
        "above_ma10": above_ma10,
        "above_ma20": above_ma20,
        "ma10_slope_up": ma10_slope_up,
        "recent_restrengthen_flag": recent_restrengthen_flag,
        "post_wave_breakdown_flag": post_wave_breakdown_flag,
        "post_wave_low_to_cost_pct": to_number(low_to_cost_pct),
        "wave_age_trade_days": wave_age_trade_days,
        "post_wave_structure_score": post_wave_structure_score,
    }


def build_final_candidate_flags(row: dict[str, Any], config: CoreManagementAccumulationConfig) -> dict[str, Any]:
    base_flags = build_preliminary_candidate_flags(row, config)
    avg_amount_20d_yuan = to_float(row.get("avg_amount_20d_yuan"))
    wave_age_trade_days = to_float(row.get("wave_age_trade_days"))
    final_confirmation_score = to_float(row.get("final_confirmation_score")) or 0.0
    strong_hold = bool(
        to_bool(row.get("above_ma20"))
        and to_bool(row.get("ma10_slope_up"))
        and (to_float(row.get("post_wave_structure_score")) or 0.0) >= config.min_retrigger_structure_score
    )
    freshness_ok = wave_age_trade_days is not None and wave_age_trade_days <= config.max_fresh_wave_age_trade_days
    if wave_age_trade_days is not None and wave_age_trade_days > config.max_fresh_wave_age_trade_days and (
        to_bool(row.get("recent_restrengthen_flag")) or strong_hold
    ):
        freshness_ok = True
    base_flags.update(
        {
            "avg_amount_ok": avg_amount_20d_yuan is not None and avg_amount_20d_yuan >= config.min_avg_amount_20d_yuan,
            "structure_ok": (
                not to_bool(row.get("post_wave_breakdown_flag"))
                and (
                    to_bool(row.get("recent_restrengthen_flag"))
                    or (to_bool(row.get("above_ma10")) and to_bool(row.get("ma10_slope_up")))
                )
            ),
            "freshness_ok": freshness_ok,
            "repeat_ok": not to_bool(row.get("repeat_signal_blocked")),
            "confirmation_ok": final_confirmation_score >= config.min_final_confirmation_score,
        }
    )
    base_flags["all_hard_filters_ok"] = all(base_flags.values())
    return base_flags


def score_final_candidates(
    candidate_df: pd.DataFrame,
    config: CoreManagementAccumulationConfig,
    recent_final_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = candidate_df.copy()
    if work.empty:
        return work
    work["identity_strength_score"] = work.apply(lambda row: _build_identity_strength_score(row.to_dict()), axis=1)
    work["continuity_score"] = work.apply(lambda row: _build_continuity_score(row.to_dict()), axis=1)
    work["cost_zone_score"] = work.apply(lambda row: _build_cost_zone_score(row.to_dict()), axis=1)
    work["aux_flow_health_score"] = work.apply(lambda row: _build_aux_flow_health_score(row.to_dict()), axis=1)
    work["post_wave_structure_score"] = pd.to_numeric(work.get("post_wave_structure_score"), errors="coerce").fillna(0.0).round(2)
    repeat_state = work.apply(lambda row: build_repeat_signal_state(row.to_dict(), config, recent_final_signals=recent_final_signals), axis=1, result_type="expand")
    work = pd.concat([work, repeat_state], axis=1)
    work = work.loc[:, ~work.columns.duplicated()].copy()
    work["freshness_penalty_score"] = work.apply(lambda row: _build_freshness_penalty_score(row.to_dict(), config), axis=1)
    work["retrigger_quality_score"] = work.apply(lambda row: _build_retrigger_quality_score(row.to_dict(), config), axis=1)
    work["final_confirmation_score"] = (
        pd.to_numeric(work["retrigger_quality_score"], errors="coerce").fillna(0.0)
        + np.where(pd.to_numeric(work["post_wave_structure_score"], errors="coerce").fillna(0.0) >= config.min_retrigger_structure_score, 1.5, 0.0)
    ).round(2)
    work["base_total_score"] = (
        pd.to_numeric(work["identity_strength_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["continuity_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["aux_flow_health_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["cost_zone_score"], errors="coerce").fillna(0.0) * 0.4
        + pd.to_numeric(work["post_wave_structure_score"], errors="coerce").fillna(0.0) * 1.1
        + pd.to_numeric(work["final_confirmation_score"], errors="coerce").fillna(0.0) * 1.2
    ).round(2)
    work["adjusted_total_score"] = (
        pd.to_numeric(work["base_total_score"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["freshness_penalty_score"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["repeat_penalty_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["retrigger_quality_score"], errors="coerce").fillna(0.0)
    ).round(2)
    work["total_score"] = work["adjusted_total_score"]
    flags = work.apply(lambda row: build_final_candidate_flags(row.to_dict(), config), axis=1, result_type="expand")
    work = pd.concat([work, flags], axis=1)
    work["strategy_name"] = STRATEGY_NAME
    work["strategy_rank_score"] = work["adjusted_total_score"]
    filtered = work[
        work["all_hard_filters_ok"].fillna(False)
        & (pd.to_numeric(work["adjusted_total_score"], errors="coerce").fillna(0.0) >= config.min_total_score)
    ].copy()
    if filtered.empty:
        return filtered
    filtered = filtered.sort_values(
        ["adjusted_total_score", "final_confirmation_score", "post_wave_structure_score", "recent_restrengthen_flag", "wave_last_date", "wave_total_amount"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    return filtered


def display_columns() -> list[str]:
    return [
        "ts_code",
        "name",
        "industry",
        "strategy_name",
        "strategy_rank_score",
        "total_score",
        "base_total_score",
        "adjusted_total_score",
        "identity_strength_score",
        "continuity_score",
        "cost_zone_score",
        "post_wave_structure_score",
        "aux_flow_health_score",
        "freshness_penalty_score",
        "repeat_penalty_score",
        "retrigger_quality_score",
        "final_confirmation_score",
        "wave_age_trade_days",
        "wave_signature",
        "repeat_recent_signal_hit",
        "repeat_signal_distance_trade_days",
        "repeat_allowed_override",
        "repeat_signal_blocked",
        "wave_best_identity_label",
        "wave_trade_days",
        "wave_event_count",
        "wave_total_amount",
        "wave_buy_avg_price_weighted",
        "current_price",
        "current_to_cost_pct",
        "avg_amount_20d_yuan",
        "recent_restrengthen_flag",
        "post_wave_breakdown_flag",
        "wave_first_date",
        "wave_last_date",
        "trade_date",
    ]


def build_screen_summary(
    config: CoreManagementAccumulationConfig,
    export_dir: str,
    latest_trade_date: str,
    market_moneyflow_dates: list[str],
    wave_details: pd.DataFrame,
    stage1_candidates: pd.DataFrame,
    final_candidates: pd.DataFrame,
    best_pick_candidate: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "strategy_id": STRATEGY_ID,
        "strategy_name": STRATEGY_NAME,
        "requested_end_date": config.end_date,
        "latest_trade_date": latest_trade_date,
        "market_moneyflow_dates": market_moneyflow_dates,
        "raw_wave_rows": int(len(wave_details)),
        "stage1_candidate_rows": int(len(stage1_candidates)),
        "final_candidate_rows": int(len(final_candidates)),
        "best_pick_ts_code": best_pick_candidate.iloc[0]["ts_code"] if not best_pick_candidate.empty else None,
        "best_pick_name": best_pick_candidate.iloc[0]["name"] if not best_pick_candidate.empty else None,
        "config_snapshot": json_safe(config.to_dict()),
        "export_dir": export_dir,
    }
