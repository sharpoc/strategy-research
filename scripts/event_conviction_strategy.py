from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from core_management_accumulation_strategy import (
    CoreManagementAccumulationConfig,
    build_margin_summary,
    build_post_wave_structure_metrics,
    classify_holder_identity,
)
from holder_strategy_core import clip_score, ensure_columns, normalize_trade_day, to_bool, to_float


EVENT_STRATEGY_ID = "event_conviction_signal"
EVENT_STRATEGY_NAME = "事件信念臻选"


@dataclass(frozen=True)
class EventConvictionConfig:
    ann_start_date: str
    end_date: str
    event_chunk_days: int = 5
    price_lookback_days: int = 250
    moneyflow_lookback_days: int = 5
    recent_wave_trade_days: int = 15
    min_list_days: int = 120
    min_price: float = 3.0
    min_publish_score: float = 45.0
    max_candidates: int = 300
    include_star: bool = False
    include_gem: bool = True
    event_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "management_buy": 16.0,
            "important_shareholder_buy": 11.0,
            "company_buyback": 14.0,
        }
    )
    identity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "company": 20.0,
            "controller": 18.0,
            "core_management": 16.0,
            "industry_capital": 12.0,
            "important_shareholder": 10.0,
            "other": 4.0,
        }
    )
    conviction_weights: dict[str, float] = field(
        default_factory=lambda: {
            "amount_max": 14.0,
            "continuity_max": 8.0,
            "density_max": 5.0,
            "execution_max": 5.0,
        }
    )
    cost_zone_weights: dict[str, float] = field(
        default_factory=lambda: {
            "tight_band": 14.0,
            "mid_band": 10.0,
            "wide_band": 6.0,
            "far_penalty": 4.0,
        }
    )
    structure_weights: dict[str, float] = field(
        default_factory=lambda: {
            "base": 0.8,
            "restabilize_bonus": 3.0,
            "breakdown_penalty": 6.0,
        }
    )
    market_confirmation_weights: dict[str, float] = field(
        default_factory=lambda: {
            "main_flow_max": 6.0,
            "margin_flow_max": 4.0,
            "turnover_health_max": 4.0,
            "amount_health_max": 4.0,
        }
    )
    penalty_weights: dict[str, float] = field(
        default_factory=lambda: {
            "repeat_signal": 5.0,
            "old_event": 4.0,
            "execution_low": 3.0,
            "extreme_distance": 5.0,
        }
    )

    @classmethod
    def for_end_date(
        cls,
        end_date: str,
        ann_start_date: str = "",
        **overrides: Any,
    ) -> "EventConvictionConfig":
        end_str = normalize_trade_day(end_date)
        ann_str = normalize_trade_day(ann_start_date) if ann_start_date else ""
        if not ann_str:
            ann_str = (pd.Timestamp(end_str) - pd.Timedelta(days=45)).strftime("%Y%m%d")
        return cls(ann_start_date=ann_str, end_date=end_str, **overrides)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventConvictionConfig":
        merged = dict(data)
        return cls(
            ann_start_date=normalize_trade_day(merged.get("ann_start_date", "")),
            end_date=normalize_trade_day(merged.get("end_date", "")),
            event_chunk_days=int(merged.get("event_chunk_days", 5)),
            price_lookback_days=int(merged.get("price_lookback_days", 250)),
            moneyflow_lookback_days=int(merged.get("moneyflow_lookback_days", 5)),
            recent_wave_trade_days=int(merged.get("recent_wave_trade_days", 15)),
            min_list_days=int(merged.get("min_list_days", 120)),
            min_price=float(merged.get("min_price", 3.0)),
            min_publish_score=float(merged.get("min_publish_score", 45.0)),
            max_candidates=int(merged.get("max_candidates", 300)),
            include_star=to_bool(merged.get("include_star", False)),
            include_gem=to_bool(merged.get("include_gem", True)),
            event_type_weights=dict(merged.get("event_type_weights", {}))
            or cls("", "").event_type_weights,
            identity_weights=dict(merged.get("identity_weights", {}))
            or cls("", "").identity_weights,
            conviction_weights=dict(merged.get("conviction_weights", {}))
            or cls("", "").conviction_weights,
            cost_zone_weights=dict(merged.get("cost_zone_weights", {}))
            or cls("", "").cost_zone_weights,
            structure_weights=dict(merged.get("structure_weights", {}))
            or cls("", "").structure_weights,
            market_confirmation_weights=dict(merged.get("market_confirmation_weights", {}))
            or cls("", "").market_confirmation_weights,
            penalty_weights=dict(merged.get("penalty_weights", {}))
            or cls("", "").penalty_weights,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    return "ST" in str(name or "").upper()


def _identity_role_from_bucket(bucket: str) -> str:
    mapping = {
        "core_control": "controller",
        "core_exec": "core_management",
        "senior_exec": "core_management",
        "exec_default": "core_management",
        "corporate_holder": "important_shareholder",
        "person_holder": "important_shareholder",
        "other": "other",
    }
    return mapping.get(str(bucket or ""), "other")


def _to_number(value: Any, default: float = 0.0) -> float:
    number = to_float(value)
    return float(default if number is None else number)


def build_holdertrade_event_candidates(
    holdertrade_df: pd.DataFrame,
    stock_basic_df: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    latest_trade_date: str,
    config: EventConvictionConfig,
) -> pd.DataFrame:
    if holdertrade_df.empty:
        return pd.DataFrame()
    work = holdertrade_df.copy()
    work["ts_code"] = work.get("ts_code", "").fillna("").astype(str)
    work["ann_date"] = work.get("ann_date", "").fillna("").astype(str)
    work = work[(work["ann_date"] >= config.ann_start_date) & (work["ann_date"] <= latest_trade_date)].copy()
    if work.empty:
        return pd.DataFrame()
    work = work[work.get("in_de", "").fillna("").astype(str).str.upper() == "IN"].copy()
    if work.empty:
        return pd.DataFrame()
    for col in ["change_vol", "avg_price", "change_ratio", "after_ratio"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    identity_rows = work.apply(lambda row: classify_holder_identity(row.get("holder_type"), row.get("holder_name")), axis=1, result_type="expand")
    work = pd.concat([work, identity_rows], axis=1)
    work["event_amount"] = pd.to_numeric(work.get("change_vol"), errors="coerce").fillna(0.0) * pd.to_numeric(work.get("avg_price"), errors="coerce").fillna(0.0)

    basic_cols = [c for c in ["ts_code", "symbol", "name", "industry", "market", "list_date"] if c in stock_basic_df.columns]
    stock_basic_work = stock_basic_df[basic_cols].copy()
    stock_basic_work["ts_code"] = stock_basic_work["ts_code"].fillna("").astype(str)
    work = work.merge(stock_basic_work.drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")
    snapshot_cols = [
        c
        for c in market_snapshot.columns
        if c not in {"trade_date", "ts_code", "symbol", "name", "industry", "market", "list_date"}
    ]
    if snapshot_cols:
        market_snapshot_work = market_snapshot[["ts_code", *snapshot_cols]].copy()
        market_snapshot_work["ts_code"] = market_snapshot_work["ts_code"].fillna("").astype(str)
        work = work.merge(market_snapshot_work.drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")

    rows: list[dict[str, Any]] = []
    for ts_code, sub in work.groupby("ts_code", dropna=False):
        ordered = sub.sort_values(["ann_date", "event_amount"], ascending=[True, False]).reset_index(drop=True)
        if ordered.empty:
            continue
        latest = ordered.iloc[-1]
        latest_close = to_float(latest.get("close"))
        listing_days = None
        list_date = normalize_trade_day(latest.get("list_date", ""))
        if list_date:
            listing_days = int((pd.Timestamp(latest_trade_date) - pd.Timestamp(list_date)).days)
        identity_weight = float(pd.to_numeric(ordered["identity_weight"], errors="coerce").fillna(0.0).max())
        core_count = int(pd.to_numeric(ordered["core_management_flag"], errors="coerce").fillna(False).astype(bool).sum())
        total_amount = float(pd.to_numeric(ordered["event_amount"], errors="coerce").fillna(0.0).sum())
        total_vol = float(pd.to_numeric(ordered["change_vol"], errors="coerce").fillna(0.0).sum())
        weighted_cost = total_amount / total_vol if total_amount > 0 and total_vol > 0 else to_float(latest.get("avg_price"))
        current_to_cost_ratio = latest_close / weighted_cost if latest_close and weighted_cost else None
        role = _identity_role_from_bucket(str(ordered.sort_values("identity_weight", ascending=False).iloc[0].get("identity_bucket", "")))
        event_type = "management_buy" if role == "core_management" or role == "controller" else "important_shareholder_buy"
        execution_ratio = 1.0
        rows.append(
            {
                "ts_code": ts_code,
                "name": latest.get("name"),
                "industry": latest.get("industry"),
                "market": latest.get("market"),
                "list_date": list_date,
                "listing_days": listing_days,
                "is_st": _is_st_name(latest.get("name")),
                "board_allowed": _board_allowed(str(ts_code), latest.get("market"), config.include_star, config.include_gem),
                "event_type": event_type,
                "event_date": str(latest.get("ann_date") or ""),
                "event_direction": "positive",
                "identity_role": role,
                "identity_weight": identity_weight,
                "event_amount": round(total_amount, 2),
                "event_count": int(len(ordered)),
                "core_event_count": core_count,
                "trade_day_count": int(ordered["ann_date"].astype(str).nunique()),
                "density_score_raw": int(max(core_count, len(ordered))),
                "execution_ratio": execution_ratio,
                "cost_anchor": weighted_cost,
                "current_to_cost_ratio": current_to_cost_ratio,
                "source_strategy": "core_management_accumulation",
                "event_label": "高管/股东增持",
                "holder_name": latest.get("holder_name"),
                "holder_type": latest.get("holder_type"),
                "turnover_rate": latest.get("turnover_rate"),
                "turnover_rate_f": latest.get("turnover_rate_f"),
                "volume_ratio": latest.get("volume_ratio"),
                "main_net_amount_3d": latest.get("main_net_amount_3d"),
                "main_net_amount_5d": latest.get("main_net_amount_5d"),
                "margin_net_buy_3d": latest.get("margin_net_buy_3d"),
                "margin_positive_days_3d": latest.get("margin_positive_days_3d"),
                "close": latest_close,
            }
        )
    return pd.DataFrame(rows)


def build_buyback_event_candidates(
    buyback_df: pd.DataFrame,
    stock_basic_df: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    latest_trade_date: str,
    config: EventConvictionConfig,
) -> pd.DataFrame:
    if buyback_df.empty:
        return pd.DataFrame()
    work = buyback_df.copy()
    work["ts_code"] = work.get("ts_code", "").fillna("").astype(str)
    date_col = "ann_date" if "ann_date" in work.columns else ("end_date" if "end_date" in work.columns else "")
    if not date_col:
        return pd.DataFrame()
    work[date_col] = work[date_col].fillna("").astype(str).str.replace("-", "", regex=False)
    work = work[(work[date_col] >= config.ann_start_date) & (work[date_col] <= latest_trade_date)].copy()
    if work.empty:
        return pd.DataFrame()
    status_text = work.get("proc", "").fillna("").astype(str)
    if not status_text.empty:
        mask = ~status_text.str.contains("终止|失败|取消", regex=True)
        work = work[mask].copy()
    for col in ["amount", "vol", "high_limit", "low_limit"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    basic_cols = [c for c in ["ts_code", "symbol", "name", "industry", "market", "list_date"] if c in stock_basic_df.columns]
    stock_basic_work = stock_basic_df[basic_cols].copy()
    stock_basic_work["ts_code"] = stock_basic_work["ts_code"].fillna("").astype(str)
    work = work.merge(stock_basic_work.drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")
    snapshot_cols = [
        c
        for c in market_snapshot.columns
        if c not in {"trade_date", "ts_code", "symbol", "name", "industry", "market", "list_date"}
    ]
    if snapshot_cols:
        market_snapshot_work = market_snapshot[["ts_code", *snapshot_cols]].copy()
        market_snapshot_work["ts_code"] = market_snapshot_work["ts_code"].fillna("").astype(str)
        work = work.merge(market_snapshot_work.drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")

    rows: list[dict[str, Any]] = []
    for ts_code, sub in work.groupby("ts_code", dropna=False):
        ordered = sub.sort_values(date_col).reset_index(drop=True)
        latest = ordered.iloc[-1]
        latest_close = to_float(latest.get("close"))
        listing_days = None
        list_date = normalize_trade_day(latest.get("list_date", ""))
        if list_date:
            listing_days = int((pd.Timestamp(latest_trade_date) - pd.Timestamp(list_date)).days)
        amount = float(pd.to_numeric(ordered.get("amount"), errors="coerce").fillna(0.0).max()) if "amount" in ordered.columns else 0.0
        vol = float(pd.to_numeric(ordered.get("vol"), errors="coerce").fillna(0.0).max()) if "vol" in ordered.columns else 0.0
        cost_anchor = None
        if amount > 0 and vol > 0:
            cost_anchor = amount / vol
        else:
            high_limit = to_float(latest.get("high_limit"))
            low_limit = to_float(latest.get("low_limit"))
            if high_limit and low_limit:
                cost_anchor = (high_limit + low_limit) / 2.0
        current_to_cost_ratio = latest_close / cost_anchor if latest_close and cost_anchor else None
        execution_ratio = 1.0 if amount > 0 else 0.5
        rows.append(
            {
                "ts_code": ts_code,
                "name": latest.get("name"),
                "industry": latest.get("industry"),
                "market": latest.get("market"),
                "list_date": list_date,
                "listing_days": listing_days,
                "is_st": _is_st_name(latest.get("name")),
                "board_allowed": _board_allowed(str(ts_code), latest.get("market"), config.include_star, config.include_gem),
                "event_type": "company_buyback",
                "event_date": str(latest.get(date_col) or ""),
                "event_direction": "positive",
                "identity_role": "company",
                "identity_weight": 1.0,
                "event_amount": round(amount, 2),
                "event_count": int(len(ordered)),
                "core_event_count": int(len(ordered)),
                "trade_day_count": int(ordered[date_col].astype(str).nunique()),
                "density_score_raw": int(len(ordered)),
                "execution_ratio": execution_ratio,
                "cost_anchor": cost_anchor,
                "current_to_cost_ratio": current_to_cost_ratio,
                "source_strategy": "company_buyback",
                "event_label": "公司回购",
                "turnover_rate": latest.get("turnover_rate"),
                "turnover_rate_f": latest.get("turnover_rate_f"),
                "volume_ratio": latest.get("volume_ratio"),
                "main_net_amount_3d": latest.get("main_net_amount_3d"),
                "main_net_amount_5d": latest.get("main_net_amount_5d"),
                "margin_net_buy_3d": latest.get("margin_net_buy_3d"),
                "margin_positive_days_3d": latest.get("margin_positive_days_3d"),
                "close": latest_close,
            }
        )
    return pd.DataFrame(rows)


def apply_bottom_filters(candidate_df: pd.DataFrame, config: EventConvictionConfig) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    work = candidate_df.copy()
    work["listing_days"] = pd.to_numeric(work.get("listing_days"), errors="coerce")
    work["close"] = pd.to_numeric(work.get("close"), errors="coerce")
    work["board_allowed"] = work.get("board_allowed", False).fillna(False).astype(bool)
    work["event_direction"] = work.get("event_direction", "").fillna("").astype(str)
    keep = (
        work["board_allowed"]
        & (~work.get("is_st", False).fillna(False).astype(bool))
        & (work["listing_days"].fillna(0) >= config.min_list_days)
        & (work["close"].fillna(0) >= config.min_price)
        & work["event_direction"].str.lower().isin(["positive", "in", "buyback"])
    )
    return work[keep].copy().reset_index(drop=True)


def enrich_market_features(
    candidate_df: pd.DataFrame,
    price_bundle_map: dict[str, dict[str, pd.DataFrame]],
    end_date: str,
    config: EventConvictionConfig,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    core_cfg = CoreManagementAccumulationConfig.for_end_date(end_date)
    rows: list[dict[str, Any]] = []
    for _, row in candidate_df.iterrows():
        ts_code = str(row.get("ts_code") or "")
        bundle = price_bundle_map.get(ts_code, {})
        metrics = build_post_wave_structure_metrics(
            daily_df=bundle.get("daily_df", pd.DataFrame()),
            adj_df=bundle.get("adj_df", pd.DataFrame()),
            end_date=end_date,
            wave_first_date=str(row.get("event_date") or end_date),
            wave_last_date=str(row.get("event_date") or end_date),
            weighted_cost=row.get("cost_anchor"),
            config=core_cfg,
        )
        metrics["post_event_breakdown_flag"] = metrics.pop("post_wave_breakdown_flag", True)
        rows.append({"ts_code": ts_code, **metrics})
    return candidate_df.merge(pd.DataFrame(rows), on="ts_code", how="left")


def score_event_type(row: dict[str, Any], config: EventConvictionConfig) -> float:
    return round(float(config.event_type_weights.get(str(row.get("event_type") or ""), 0.0)), 2)


def score_identity(row: dict[str, Any], config: EventConvictionConfig) -> float:
    role = str(row.get("identity_role") or "other")
    base = float(config.identity_weights.get(role, config.identity_weights.get("other", 0.0)))
    return round(base * max(float(_to_number(row.get("identity_weight"), 0.0)), 0.2), 2)


def score_conviction(row: dict[str, Any], config: EventConvictionConfig) -> float:
    amount = _to_number(row.get("event_amount"))
    event_count = _to_number(row.get("event_count"))
    trade_day_count = _to_number(row.get("trade_day_count"))
    execution_ratio = _to_number(row.get("execution_ratio"), 0.0)
    amount_score = 0.0
    if amount >= 100_000_000:
        amount_score = float(config.conviction_weights.get("amount_max", 14.0))
    elif amount >= 30_000_000:
        amount_score = float(config.conviction_weights.get("amount_max", 14.0)) * 0.75
    elif amount >= 10_000_000:
        amount_score = float(config.conviction_weights.get("amount_max", 14.0)) * 0.45
    elif amount > 0:
        amount_score = float(config.conviction_weights.get("amount_max", 14.0)) * 0.2

    continuity_score = min(trade_day_count, 4.0) / 4.0 * float(config.conviction_weights.get("continuity_max", 8.0))
    density_score = min(event_count, 6.0) / 6.0 * float(config.conviction_weights.get("density_max", 5.0))
    execution_score = max(min(execution_ratio, 1.0), 0.0) * float(config.conviction_weights.get("execution_max", 5.0))
    return round(amount_score + continuity_score + density_score + execution_score, 2)


def score_cost_zone(row: dict[str, Any], config: EventConvictionConfig) -> float:
    ratio = to_float(row.get("current_to_cost_ratio"))
    if ratio is None:
        return 0.0
    weights = config.cost_zone_weights
    if 0.98 <= ratio <= 1.03:
        return round(float(weights.get("tight_band", 14.0)), 2)
    if 0.95 <= ratio <= 1.08:
        return round(float(weights.get("mid_band", 10.0)), 2)
    if 0.90 <= ratio <= 1.15:
        return round(float(weights.get("wide_band", 6.0)), 2)
    return round(-float(weights.get("far_penalty", 4.0)), 2)


def score_structure(row: dict[str, Any], config: EventConvictionConfig) -> float:
    weights = config.structure_weights
    event_type = str(row.get("event_type") or "")
    structure_raw = _to_number(row.get("post_wave_structure_score"), 0.0)
    low_to_cost_pct = _to_number(row.get("post_wave_low_to_cost_pct"), 0.0)
    above_ma5 = to_bool(row.get("above_ma5"))
    above_ma10 = to_bool(row.get("above_ma10"))
    above_ma20 = to_bool(row.get("above_ma20"))
    ma10_slope_up = to_bool(row.get("ma10_slope_up"))
    base = structure_raw * float(weights.get("base", 0.8))

    if to_bool(row.get("recent_restrengthen_flag")):
        base += float(weights.get("restabilize_bonus", 3.0))

    if structure_raw >= 24:
        base += float(weights.get("high_structure_bonus", 0.0))
    elif structure_raw <= 12:
        base -= float(weights.get("low_structure_penalty", 0.0))

    if low_to_cost_pct <= -8:
        base -= float(weights.get("deep_pullback_penalty", 0.0))
    elif low_to_cost_pct <= -4:
        base -= float(weights.get("mid_pullback_penalty", 0.0))

    if above_ma5 and above_ma10 and above_ma20 and ma10_slope_up:
        base += float(weights.get("trend_alignment_bonus", 0.0))
    elif above_ma10 and above_ma20:
        base += float(weights.get("mid_trend_bonus", 0.0))

    if not above_ma10:
        base -= float(weights.get("ma10_loss_penalty", 0.0))
    if not above_ma20:
        base -= float(weights.get("ma20_loss_penalty", 0.0))
    if not ma10_slope_up:
        base -= float(weights.get("ma10_slope_penalty", 0.0))

    if to_bool(row.get("post_event_breakdown_flag")):
        penalty = float(weights.get("breakdown_penalty", 6.0))
        if event_type == "management_buy":
            penalty += float(weights.get("management_breakdown_extra_penalty", 0.0))
        base -= penalty

    if event_type == "management_buy":
        if not to_bool(row.get("recent_restrengthen_flag")):
            base -= float(weights.get("management_no_restrengthen_penalty", 0.0))
        if structure_raw < float(weights.get("management_min_structure_gate", 0.0)):
            base -= float(weights.get("management_low_structure_penalty", 0.0))
        if low_to_cost_pct <= float(weights.get("management_pullback_gate", -5.0)):
            base -= float(weights.get("management_pullback_penalty", 0.0))
        if not above_ma10:
            base -= float(weights.get("management_ma10_loss_penalty", 0.0))
        if not above_ma20:
            base -= float(weights.get("management_ma20_loss_penalty", 0.0))
        if not ma10_slope_up:
            base -= float(weights.get("management_ma10_slope_penalty", 0.0))

    return round(base, 2)


def score_market_confirmation(row: dict[str, Any], config: EventConvictionConfig) -> float:
    weights = config.market_confirmation_weights
    score = 0.0
    main3 = _to_number(row.get("main_net_amount_3d"))
    main5 = _to_number(row.get("main_net_amount_5d"))
    margin3 = _to_number(row.get("margin_net_buy_3d"))
    turnover = to_float(row.get("turnover_rate_f")) or to_float(row.get("turnover_rate")) or 0.0
    amount20 = _to_number(row.get("avg_amount_20d_yuan"))
    if main3 > 0 and main5 > 0:
        score += float(weights.get("main_flow_max", 6.0))
    elif main3 > 0 or main5 > 0:
        score += float(weights.get("main_flow_max", 6.0)) * 0.6
    if margin3 > 0:
        score += float(weights.get("margin_flow_max", 4.0))
    if 0.8 <= turnover <= 12.0:
        score += float(weights.get("turnover_health_max", 4.0))
    elif 0.3 <= turnover <= 20.0:
        score += float(weights.get("turnover_health_max", 4.0)) * 0.5
    if amount20 >= 300_000_000:
        score += float(weights.get("amount_health_max", 4.0))
    elif amount20 >= 100_000_000:
        score += float(weights.get("amount_health_max", 4.0)) * 0.6
    return round(score, 2)


def score_penalty(row: dict[str, Any], config: EventConvictionConfig) -> float:
    weights = config.penalty_weights
    score = 0.0
    ratio = to_float(row.get("current_to_cost_ratio"))
    age = _to_number(row.get("wave_age_trade_days"), 0.0)
    execution_ratio = _to_number(row.get("execution_ratio"), 1.0)
    recent_top1_count = _to_number(row.get("recent_top1_count"), 0.0)
    event_type = str(row.get("event_type") or "")
    structure_raw = _to_number(row.get("post_wave_structure_score"), 0.0)
    main3 = _to_number(row.get("main_net_amount_3d"))
    main5 = _to_number(row.get("main_net_amount_5d"))
    if age > 8:
        score += float(weights.get("old_event", 4.0))
    elif age > 5:
        score += float(weights.get("old_event", 4.0)) * 0.5
    if execution_ratio < 0.4:
        score += float(weights.get("execution_low", 3.0))
    if ratio is not None and (ratio > 1.25 or ratio < 0.82):
        score += float(weights.get("extreme_distance", 5.0))
    if recent_top1_count > 0:
        score += float(weights.get("repeat_signal", 5.0)) * recent_top1_count
    if event_type == "company_buyback":
        if structure_raw < 8:
            score += float(weights.get("buyback_weak_structure", 6.0))
        elif structure_raw < 14:
            score += float(weights.get("buyback_weak_structure", 6.0)) * 0.5
        if not (main3 > 0 and main5 > 0):
            score += float(weights.get("buyback_flow_mismatch", 5.0))
    return round(score, 2)


def score_candidates(candidate_df: pd.DataFrame, config: EventConvictionConfig) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    work = candidate_df.copy()
    work["event_type_score"] = work.apply(lambda row: score_event_type(row.to_dict(), config), axis=1)
    work["identity_score"] = work.apply(lambda row: score_identity(row.to_dict(), config), axis=1)
    work["conviction_score"] = work.apply(lambda row: score_conviction(row.to_dict(), config), axis=1)
    work["cost_zone_score"] = work.apply(lambda row: score_cost_zone(row.to_dict(), config), axis=1)
    work["structure_score"] = work.apply(lambda row: score_structure(row.to_dict(), config), axis=1)
    work["market_confirmation_score"] = work.apply(lambda row: score_market_confirmation(row.to_dict(), config), axis=1)
    work["penalty_score"] = work.apply(lambda row: score_penalty(row.to_dict(), config), axis=1)
    work["total_score"] = (
        pd.to_numeric(work["event_type_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["identity_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["conviction_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["cost_zone_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["structure_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["market_confirmation_score"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["penalty_score"], errors="coerce").fillna(0.0)
    ).round(2)
    return work.sort_values(
        ["total_score", "structure_score", "conviction_score", "identity_score", "event_amount", "event_date"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def select_top1_candidate(scored_df: pd.DataFrame, config: EventConvictionConfig) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df.head(0)
    return scored_df.head(1).copy()


def build_screen_summary(
    config: EventConvictionConfig,
    export_dir: str,
    latest_trade_date: str,
    candidate_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    top1_df: pd.DataFrame,
) -> dict[str, Any]:
    event_counts = candidate_df["event_type"].astype(str).value_counts().to_dict() if not candidate_df.empty and "event_type" in candidate_df.columns else {}
    return {
        "strategy_id": EVENT_STRATEGY_ID,
        "strategy_name": EVENT_STRATEGY_NAME,
        "requested_end_date": config.end_date,
        "latest_trade_date": latest_trade_date,
        "candidate_rows": int(len(candidate_df)),
        "candidate_unique_stock_count": int(candidate_df["ts_code"].astype(str).nunique()) if not candidate_df.empty else 0,
        "event_type_counts": {str(k): int(v) for k, v in event_counts.items()},
        "selection_mode": "score_rank_top1",
        "min_publish_score": float(config.min_publish_score),
        "top1_rows": int(len(top1_df)),
        "best_pick_ts_code": str(top1_df.iloc[0]["ts_code"]) if not top1_df.empty else None,
        "best_pick_name": str(top1_df.iloc[0]["name"]) if not top1_df.empty else None,
        "best_pick_score": _to_number(top1_df.iloc[0]["total_score"], 0.0) if not top1_df.empty else None,
        "config_snapshot": config.to_dict(),
        "export_dir": export_dir,
    }


def display_columns() -> list[str]:
    return [
        "ts_code",
        "name",
        "event_type",
        "event_date",
        "identity_role",
        "event_amount",
        "event_count",
        "trade_day_count",
        "execution_ratio",
        "current_to_cost_ratio",
        "avg_amount_20d_yuan",
        "post_wave_structure_score",
        "event_type_score",
        "identity_score",
        "conviction_score",
        "cost_zone_score",
        "structure_score",
        "market_confirmation_score",
        "penalty_score",
        "total_score",
    ]
