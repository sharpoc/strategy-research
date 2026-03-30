from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_research_filter_metrics(window_history: pd.DataFrame) -> pd.DataFrame:
    if window_history is None or window_history.empty:
        return pd.DataFrame(columns=["ts_code", "close_qfq", "avg_amount_20", "listed_trade_days"])

    history = window_history.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    for column in ["close", "amount", "vol"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    if "amount" not in history.columns or history["amount"].dropna().empty:
        close_series = pd.to_numeric(history.get("close"), errors="coerce")
        vol_series = pd.to_numeric(history.get("vol"), errors="coerce")
        history["amount_proxy"] = close_series * vol_series
        amount_col = "amount_proxy"
    else:
        amount_col = "amount"

    history["avg_amount_20"] = history.groupby("ts_code")[amount_col].transform(lambda s: s.rolling(20, min_periods=5).mean())
    history["listed_trade_days"] = history.groupby("ts_code").cumcount() + 1

    latest = history.groupby("ts_code", as_index=False).tail(1).copy()
    latest = latest.rename(columns={"close": "close_qfq"})
    if "trade_date" in latest.columns:
        as_of_date = pd.to_datetime(latest["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        latest["as_of_date"] = as_of_date
    keep_cols = [column for column in ["ts_code", "close_qfq", "avg_amount_20", "listed_trade_days", "as_of_date"] if column in latest.columns]
    return latest[keep_cols].copy()


def apply_research_candidate_filters(
    candidates: pd.DataFrame,
    window_history: pd.DataFrame,
    filter_config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candidates is None or candidates.empty:
        return pd.DataFrame(), {"enabled": False, "before_count": 0, "after_count": 0}

    config = dict(filter_config or {})
    enabled = _as_bool(config.get("enabled"), default=False)
    before_count = int(len(candidates))
    if not enabled:
        return candidates.copy(), {"enabled": False, "before_count": before_count, "after_count": before_count}

    filtered = candidates.copy()
    has_embedded_metrics = {"avg_amount_20", "listed_trade_days"}.issubset(filtered.columns)
    metrics = pd.DataFrame()
    if not has_embedded_metrics:
        metrics = build_research_filter_metrics(window_history)
    if not metrics.empty:
        filtered = filtered.merge(metrics, on="ts_code", how="left", suffixes=("", "_research"))
        if "close_qfq_research" in filtered.columns:
            filtered["close_qfq"] = pd.to_numeric(filtered.get("close_qfq"), errors="coerce").fillna(
                pd.to_numeric(filtered["close_qfq_research"], errors="coerce")
            )
            filtered = filtered.drop(columns=["close_qfq_research"])
        if "as_of_date" in metrics.columns:
            filtered["as_of_date"] = filtered["ts_code"].map(metrics.set_index("ts_code")["as_of_date"])

    name_series = filtered.get("name", pd.Series("", index=filtered.index)).fillna("").astype(str)
    market_series = filtered.get("market", pd.Series("", index=filtered.index)).fillna("").astype(str)
    ts_code_series = filtered.get("ts_code", pd.Series("", index=filtered.index)).fillna("").astype(str)
    close_series = pd.to_numeric(filtered.get("close_qfq"), errors="coerce")
    listed_trade_days = pd.to_numeric(filtered.get("listed_trade_days"), errors="coerce")
    avg_amount_20 = pd.to_numeric(filtered.get("avg_amount_20"), errors="coerce")
    list_date_series = pd.to_datetime(filtered.get("list_date"), format="%Y%m%d", errors="coerce")
    as_of_date_series = pd.to_datetime(filtered.get("as_of_date"), errors="coerce")
    listed_calendar_days = (as_of_date_series - list_date_series).dt.days

    mask = pd.Series(True, index=filtered.index)
    drop_reasons: dict[str, int] = {}

    def apply_rule(rule_name: str, rule_mask: pd.Series) -> None:
        nonlocal mask
        dropped = int((mask & ~rule_mask).sum())
        if dropped > 0:
            drop_reasons[rule_name] = dropped
        mask = mask & rule_mask

    if _as_bool(config.get("exclude_st"), default=True):
        st_mask = ~name_series.str.contains("ST", case=False, na=False)
        apply_rule("exclude_st", st_mask)

    if _as_bool(config.get("exclude_delisting"), default=True):
        delist_mask = ~name_series.str.contains("退", case=False, na=False)
        apply_rule("exclude_delisting", delist_mask)

    if _as_bool(config.get("exclude_bj"), default=True):
        bj_mask = ~ts_code_series.str.endswith(".BJ") & ~market_series.str.contains("北交", na=False)
        apply_rule("exclude_bj", bj_mask)

    if _as_bool(config.get("exclude_kcb"), default=False):
        kcb_mask = ~ts_code_series.str.startswith(("688", "689")) & ~market_series.str.contains("科创", na=False)
        apply_rule("exclude_kcb", kcb_mask)

    min_listed_trade_days = int(config.get("min_listed_trade_days") or 0)
    if min_listed_trade_days > 0:
        calendar_proxy_days = int(np.ceil(min_listed_trade_days * 1.45))
        listed_mask = pd.Series(False, index=filtered.index)
        if not listed_calendar_days.dropna().empty:
            listed_mask = listed_mask | (listed_calendar_days >= calendar_proxy_days)
        listed_mask = listed_mask | (listed_trade_days >= float(min_listed_trade_days))
        apply_rule("min_listed_trade_days", listed_mask.fillna(False))

    min_close = float(config.get("min_close") or 0.0)
    if min_close > 0:
        close_mask = close_series >= min_close
        apply_rule("min_close", close_mask.fillna(False))

    min_avg_amount_20 = float(config.get("min_avg_amount_20") or 0.0)
    if min_avg_amount_20 > 0:
        amount_mask = avg_amount_20 >= min_avg_amount_20
        apply_rule("min_avg_amount_20", amount_mask.fillna(False))

    filtered = filtered[mask].copy().reset_index(drop=True)
    meta = {
        "enabled": True,
        "before_count": before_count,
        "after_count": int(len(filtered)),
        "dropped_count": int(before_count - len(filtered)),
        "drop_reasons": drop_reasons,
    }
    return filtered, meta


def apply_research_universe_filters(
    stock_basic_df: pd.DataFrame,
    window_history: pd.DataFrame,
    filter_config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stock_basic_df is None or stock_basic_df.empty:
        return pd.DataFrame(), window_history.copy(), {"enabled": False, "before_count": 0, "after_count": 0}

    config = dict(filter_config or {})
    enabled = _as_bool(config.get("enabled"), default=False)
    before_count = int(stock_basic_df["ts_code"].nunique()) if "ts_code" in stock_basic_df.columns else int(len(stock_basic_df))
    if not enabled:
        return stock_basic_df.copy(), window_history.copy(), {"enabled": False, "before_count": before_count, "after_count": before_count}

    metrics = build_research_filter_metrics(window_history)
    base = stock_basic_df.copy()
    if not metrics.empty:
        base = base.merge(metrics, on="ts_code", how="left")

    filtered_base, meta = apply_research_candidate_filters(base, window_history, filter_config=config)
    if filtered_base.empty:
        filtered_history = window_history.iloc[0:0].copy()
    else:
        keep_codes = set(filtered_base["ts_code"].astype(str))
        filtered_history = window_history[window_history["ts_code"].astype(str).isin(keep_codes)].copy()
    return filtered_base, filtered_history, meta
