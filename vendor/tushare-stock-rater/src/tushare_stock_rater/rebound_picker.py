from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .client import TushareClient, TushareClientError
from .config import load_config
from .data_loader import choose_latest_trade_date, date_after, date_before
from .models import TableMeta
from .utils import clip, compact_join, score_range, to_float, write_json


REALTIME_TS_PATTERN = "0*.SZ,3*.SZ,6*.SH"
REALTIME_FIELDS = "ts_code,name,pre_close,high,open,low,close,vol,amount,trade_time"


@dataclass
class ReboundCandidate:
    ts_code: str
    name: str
    trade_date: str
    mode: str
    score: float
    pct_chg: float | None
    industry: str
    industry_pct_chg: float | None
    excess_drop_pct: float | None
    close: float | None
    pre_close: float | None
    rebound_from_low_pct: float | None
    close_vs_open_pct: float | None
    avg_amount_20d: float | None
    latest_amount: float | None
    amount_ratio_20d: float | None
    circ_mv: float | None
    turnover_rate: float | None
    volume_ratio: float | None
    ma20: float | None
    ma60: float | None
    dist_to_ma20_pct: float | None
    dist_to_ma60_pct: float | None
    ma20_slope_5d_pct: float | None
    holder_decrease_ratio_60d: float | None
    unlock_ratio_30d: float | None
    entry_low: float | None
    entry_high: float | None
    stop_loss: float | None
    next_day_target: float | None
    reasons: list[str]
    warnings: list[str]


@dataclass
class ReboundOutputPaths:
    output_dir: Path
    summary_md: Path
    summary_json: Path
    candidates_csv: Path


@dataclass
class ReboundBacktestTrade:
    trade_date: str
    ts_code: str
    name: str
    score: float
    buy_close: float | None
    next_trade_date: str
    next_open: float | None
    next_high: float | None
    next_close: float | None
    open_return_pct: float | None
    high_return_pct: float | None
    close_return_pct: float | None
    exit_rule: str
    exit_return_pct: float | None
    exit_price: float | None
    hit_open: bool
    hit_close: bool
    reasons: list[str]
    warnings: list[str]


@dataclass
class ReboundBacktestSummary:
    start_date: str
    end_date: str
    exit_rule: str
    selection_count: int
    skipped_days: int
    avg_open_return_pct: float | None
    avg_high_return_pct: float | None
    avg_close_return_pct: float | None
    avg_exit_return_pct: float | None
    win_rate_open_pct: float | None
    win_rate_close_pct: float | None
    win_rate_exit_pct: float | None
    best_trade_pct: float | None
    worst_trade_pct: float | None
    trades: list[ReboundBacktestTrade]


@dataclass
class ReboundBacktestPaths:
    output_dir: Path
    summary_md: Path
    summary_json: Path
    trades_csv: Path


def _strategy_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config.get("rebound_strategy", {}))
    settings["markets"] = [str(item) for item in settings.get("markets", ["主板", "创业板"])]
    settings["realtime_ts_code"] = str(settings.get("realtime_ts_code", REALTIME_TS_PATTERN))
    settings["realtime_fields"] = str(settings.get("realtime_fields", REALTIME_FIELDS))
    return settings


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def _triangular_score(value: Any, low: float, peak: float, high: float) -> float:
    number = to_float(value)
    if number is None or low >= peak or peak >= high:
        return 0.0
    if number <= low or number >= high:
        return 0.0
    if number == peak:
        return 1.0
    if number < peak:
        return (number - low) / (peak - low)
    return (high - number) / (high - peak)


def _human_amount_yi(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "未知"
    return f"{number / 100000:.2f}亿"


def _human_mv_yi(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "未知"
    return f"{number / 10000:.1f}亿"


def _limit_pct(ts_code: str) -> float:
    code = str(ts_code or "")
    if code.endswith(".BJ"):
        return 30.0
    if code.startswith("300") or code.startswith("688"):
        return 20.0
    return 10.0


def _build_history_metrics(history: pd.DataFrame, current_in_history: bool) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    work = _numeric(history, ["close", "amount"])
    work["trade_date"] = work["trade_date"].astype(str)
    work = work.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for ts_code, group in work.groupby("ts_code", sort=False):
        prices = group["close"].dropna()
        if prices.empty:
            continue
        g = group.reset_index(drop=True)
        ref = g.iloc[:-1].copy() if current_in_history and len(g) > 1 else g.copy()
        ma20 = prices.tail(20).mean() if len(prices) >= 20 else None
        ma60 = prices.tail(60).mean() if len(prices) >= 60 else None
        ma20_prev = prices.iloc[:-5].tail(20).mean() if len(prices) >= 25 else None
        amount_avg = ref["amount"].dropna().tail(20).mean() if len(ref["amount"].dropna()) >= 20 else None
        rows.append(
            {
                "ts_code": ts_code,
                "ma20": ma20,
                "ma60": ma60,
                "ma20_slope_5d_pct": ((ma20 / ma20_prev - 1) * 100) if ma20 and ma20_prev else None,
                "avg_amount_20d": amount_avg,
            }
        )
    return pd.DataFrame(rows)


def _latest_express_metrics(express_df: pd.DataFrame, negative_yoy_threshold: float) -> pd.DataFrame:
    if express_df is None or express_df.empty:
        return pd.DataFrame(columns=["ts_code", "express_yoy_net_profit", "express_negative"])
    work = _numeric(express_df, ["yoy_net_profit", "n_income"])
    for column in ["ann_date", "end_date"]:
        if column in work.columns:
            work[column] = work[column].astype(str)
    sort_cols = [col for col in ["ann_date", "end_date"] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    work = work.drop_duplicates(subset=["ts_code"], keep="first")
    work["express_negative"] = (work.get("n_income") < 0) | (work.get("yoy_net_profit") <= negative_yoy_threshold)
    return work[["ts_code", "yoy_net_profit", "express_negative"]].rename(columns={"yoy_net_profit": "express_yoy_net_profit"})


def _holdertrade_metrics(holdertrade_df: pd.DataFrame) -> pd.DataFrame:
    if holdertrade_df is None or holdertrade_df.empty:
        return pd.DataFrame(columns=["ts_code", "holder_decrease_ratio_60d", "holder_decrease_events_60d"])
    work = _numeric(holdertrade_df, ["change_ratio"])
    work["in_de"] = work.get("in_de", "").astype(str).str.upper()
    work = work[work["in_de"] == "DE"].copy()
    if work.empty:
        return pd.DataFrame(columns=["ts_code", "holder_decrease_ratio_60d", "holder_decrease_events_60d"])
    grouped = (
        work.groupby("ts_code", as_index=False)
        .agg(
            holder_decrease_ratio_60d=("change_ratio", "sum"),
            holder_decrease_events_60d=("change_ratio", "count"),
        )
    )
    return grouped


def _share_float_metrics(share_float_df: pd.DataFrame) -> pd.DataFrame:
    if share_float_df is None or share_float_df.empty:
        return pd.DataFrame(columns=["ts_code", "unlock_ratio_30d", "unlock_events_30d"])
    work = _numeric(share_float_df, ["float_ratio"])
    grouped = (
        work.groupby("ts_code", as_index=False)
        .agg(
            unlock_ratio_30d=("float_ratio", "sum"),
            unlock_events_30d=("float_ratio", "count"),
        )
    )
    return grouped


def _base_frame(
    stock_basic: pd.DataFrame,
    history: pd.DataFrame,
    snapshot: pd.DataFrame,
    trade_date: str,
    mode: str,
    daily_basic: pd.DataFrame | None = None,
    st_df: pd.DataFrame | None = None,
    express_df: pd.DataFrame | None = None,
    holdertrade_df: pd.DataFrame | None = None,
    share_float_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if stock_basic is None or stock_basic.empty or history is None or history.empty or snapshot is None or snapshot.empty:
        return pd.DataFrame()
    settings = _strategy_settings(config or load_config(None))
    stock = stock_basic.copy()
    for column in ["ts_code", "name", "industry", "market", "list_date", "list_status"]:
        if column in stock.columns:
            stock[column] = stock[column].astype(str)
    metrics = _build_history_metrics(history, current_in_history=(mode != "live"))
    if metrics.empty:
        return pd.DataFrame()
    snap = snapshot.copy()
    snap = _numeric(snap, ["pre_close", "high", "open", "low", "close", "vol", "amount", "pct_chg"])
    snap["ts_code"] = snap["ts_code"].astype(str)
    if "pct_chg" not in snap.columns:
        snap["pct_chg"] = ((snap["close"] / snap["pre_close"]) - 1) * 100
    snap["trade_date"] = trade_date
    merged = stock.merge(metrics, on="ts_code", how="inner").merge(
        snap[["ts_code", "trade_date", "pre_close", "open", "high", "low", "close", "vol", "amount", "pct_chg"]],
        on="ts_code",
        how="inner",
    )
    if daily_basic is not None and not daily_basic.empty:
        base = _numeric(daily_basic, ["turnover_rate", "volume_ratio", "circ_mv"])
        merged = merged.merge(base[["ts_code", "turnover_rate", "volume_ratio", "circ_mv"]], on="ts_code", how="left")
    if express_df is not None and not express_df.empty:
        merged = merged.merge(
            _latest_express_metrics(express_df, settings["express_negative_yoy_threshold"]),
            on="ts_code",
            how="left",
        )
    if holdertrade_df is not None and not holdertrade_df.empty:
        merged = merged.merge(_holdertrade_metrics(holdertrade_df), on="ts_code", how="left")
    if share_float_df is not None and not share_float_df.empty:
        merged = merged.merge(_share_float_metrics(share_float_df), on="ts_code", how="left")
    st_codes: set[str] = set()
    if st_df is not None and not st_df.empty and "ts_code" in st_df.columns:
        st_codes = {str(code) for code in st_df["ts_code"].astype(str)}
    trade_ts = pd.to_datetime(trade_date, format="%Y%m%d")
    list_ts = pd.to_datetime(merged["list_date"], format="%Y%m%d", errors="coerce")
    merged["list_days"] = (trade_ts - list_ts).dt.days
    merged["industry"] = merged["industry"].replace({"": "其他"}).fillna("其他")
    merged["industry_pct_chg"] = merged.groupby("industry")["pct_chg"].transform("mean")
    merged["excess_drop_pct"] = merged["pct_chg"] - merged["industry_pct_chg"]
    merged["rebound_from_low_pct"] = ((merged["close"] / merged["low"]) - 1) * 100
    merged["close_vs_open_pct"] = ((merged["close"] / merged["open"]) - 1) * 100
    merged["dist_to_ma20_pct"] = ((merged["close"] / merged["ma20"]) - 1) * 100
    merged["dist_to_ma60_pct"] = ((merged["close"] / merged["ma60"]) - 1) * 100
    merged["amount_ratio_20d"] = merged["amount"] / merged["avg_amount_20d"]
    merged["is_st"] = (
        merged["name"].str.contains("ST", case=False, na=False)
        | merged["name"].str.startswith("退", na=False)
        | merged["ts_code"].isin(st_codes)
    )
    merged["limit_pct"] = merged["ts_code"].map(_limit_pct)
    merged["near_limit_down"] = merged["pct_chg"] <= (-(merged["limit_pct"] - settings["limit_down_buffer_pct"]))
    for column in [
        "turnover_rate",
        "volume_ratio",
        "circ_mv",
        "express_yoy_net_profit",
        "holder_decrease_ratio_60d",
        "holder_decrease_events_60d",
        "unlock_ratio_30d",
        "unlock_events_30d",
    ]:
        if column not in merged.columns:
            merged[column] = None
    if "express_negative" in merged.columns:
        merged["express_negative"] = merged["express_negative"].astype("boolean").fillna(False).astype(bool)
    else:
        merged["express_negative"] = False
    return merged


def _score_frame(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    settings = _strategy_settings(config or load_config(None))
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    circ_mv = pd.to_numeric(work.get("circ_mv"), errors="coerce")
    unlock_ratio = pd.to_numeric(work.get("unlock_ratio_30d"), errors="coerce")
    holder_ratio = pd.to_numeric(work.get("holder_decrease_ratio_60d"), errors="coerce")
    work["eligible"] = True
    work["eligible"] &= work["list_status"].fillna("L") == "L"
    work["eligible"] &= work["market"].isin(settings["markets"])
    work["eligible"] &= ~work["ts_code"].str.endswith(".BJ")
    work["eligible"] &= ~work["is_st"].fillna(False)
    work["eligible"] &= work["list_days"].fillna(0) >= settings["min_list_days"]
    work["eligible"] &= work["close"].fillna(0) >= settings["min_price"]
    work["eligible"] &= work["pct_chg"].between(settings["min_drop_pct"], settings["max_drop_pct"], inclusive="both")
    work["eligible"] &= work["excess_drop_pct"].fillna(0) <= settings["min_excess_drop_pct"]
    work["eligible"] &= work["avg_amount_20d"].fillna(0) >= settings["min_avg_amount_20d"]
    work["eligible"] &= work["amount"].fillna(0) >= settings["min_latest_amount"]
    work["eligible"] &= work["dist_to_ma20_pct"].fillna(-999) >= -settings["max_distance_below_ma20_pct"]
    work["eligible"] &= work["dist_to_ma60_pct"].fillna(-999) >= -settings["max_distance_below_ma60_pct"]
    work["eligible"] &= work["ma20_slope_5d_pct"].fillna(-999) >= settings["min_ma20_slope_5d_pct"]
    work["eligible"] &= work["rebound_from_low_pct"].fillna(0) >= settings["min_rebound_from_low_pct"]
    work["eligible"] &= work["close_vs_open_pct"].fillna(-999) >= settings["min_close_vs_open_pct"]
    work["eligible"] &= ~work["near_limit_down"].fillna(False)
    if "circ_mv" in work.columns:
        work["eligible"] &= circ_mv.fillna(settings["min_circ_mv"]) >= settings["min_circ_mv"]
    work["eligible"] &= ~work["express_negative"].fillna(False)
    work["eligible"] &= holder_ratio.fillna(0) <= settings["max_holder_decrease_ratio_60d"]
    work["eligible"] &= unlock_ratio.fillna(0) <= settings["max_unlock_ratio_30d"]

    drop_score = work["pct_chg"].apply(
        lambda value: _triangular_score(value, settings["min_drop_pct"], settings["target_drop_pct"], settings["max_drop_pct"]) * 25
    )
    excess_score = work["excess_drop_pct"].apply(
        lambda value: score_range(value, good=settings["strong_excess_drop_pct"], weak=settings["min_excess_drop_pct"], max_score=20, reverse=True)
    )
    liquidity_score = (
        work["avg_amount_20d"].apply(
            lambda value: score_range(value, good=settings["strong_avg_amount_20d"], weak=settings["min_avg_amount_20d"], max_score=8)
        )
        + work["amount_ratio_20d"].apply(
            lambda value: _triangular_score(value, settings["min_amount_ratio_20d"], settings["target_amount_ratio_20d"], settings["max_amount_ratio_20d"]) * 6
        )
        + circ_mv.apply(
            lambda value: score_range(value, good=settings["strong_circ_mv"], weak=settings["min_circ_mv"], max_score=4)
        )
        + work["turnover_rate"].apply(lambda value: score_range(value, good=3.0, weak=0.8, max_score=2))
    )
    trend_score = (
        work["dist_to_ma20_pct"].apply(lambda value: score_range(value, good=1.0, weak=-settings["max_distance_below_ma20_pct"], max_score=8))
        + work["dist_to_ma60_pct"].apply(lambda value: score_range(value, good=3.0, weak=-settings["max_distance_below_ma60_pct"], max_score=6))
        + work["ma20_slope_5d_pct"].apply(lambda value: score_range(value, good=1.0, weak=settings["min_ma20_slope_5d_pct"], max_score=6))
    )
    support_score = (
        work["rebound_from_low_pct"].apply(
            lambda value: score_range(value, good=settings["strong_rebound_from_low_pct"], weak=settings["min_rebound_from_low_pct"], max_score=9)
        )
        + work["close_vs_open_pct"].apply(lambda value: score_range(value, good=-0.1, weak=settings["min_close_vs_open_pct"], max_score=4))
        + work["volume_ratio"].apply(lambda value: score_range(value, good=1.2, weak=0.5, max_score=2))
    )
    risk_bonus = (
        unlock_ratio.apply(lambda value: score_range(value, good=0.0, weak=settings["max_unlock_ratio_30d"], max_score=2, reverse=True))
        + holder_ratio.apply(
            lambda value: score_range(value, good=0.0, weak=settings["max_holder_decrease_ratio_60d"], max_score=2, reverse=True)
        )
    )
    work["score"] = (drop_score + excess_score + liquidity_score + trend_score + support_score + risk_bonus).round(2)
    return work.sort_values(["eligible", "score", "pct_chg"], ascending=[False, False, True]).reset_index(drop=True)


def build_rebound_candidates(
    stock_basic: pd.DataFrame,
    history: pd.DataFrame,
    snapshot: pd.DataFrame,
    trade_date: str,
    mode: str = "historical",
    daily_basic: pd.DataFrame | None = None,
    st_df: pd.DataFrame | None = None,
    express_df: pd.DataFrame | None = None,
    holdertrade_df: pd.DataFrame | None = None,
    share_float_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> list[ReboundCandidate]:
    base = _base_frame(
        stock_basic=stock_basic,
        history=history,
        snapshot=snapshot,
        trade_date=trade_date,
        mode=mode,
        daily_basic=daily_basic,
        st_df=st_df,
        express_df=express_df,
        holdertrade_df=holdertrade_df,
        share_float_df=share_float_df,
        config=config,
    )
    ranked = _score_frame(base, config=config)
    if ranked.empty:
        return []
    settings = _strategy_settings(config or load_config(None))
    candidates: list[ReboundCandidate] = []
    for row in ranked.itertuples(index=False):
        if not getattr(row, "eligible", False):
            continue
        close = to_float(row.close)
        low = to_float(row.low)
        pre_close = to_float(row.pre_close)
        entry_low = round(close * (1 - settings["entry_discount_pct"] / 100), 2) if close else None
        entry_high = round(close * (1 + settings["entry_premium_pct"] / 100), 2) if close else None
        stop_loss = None
        if close and low:
            stop_loss = round(min(close * (1 - settings["stop_loss_pct"] / 100), low * 0.995), 2)
        next_day_target = round(pre_close, 2) if pre_close else None
        reasons = [
            f"当日跌幅 {to_float(row.pct_chg):.2f}%，比行业多跌 {abs(to_float(row.excess_drop_pct) or 0):.2f}%",
            f"20日均成交额 {_human_amount_yi(row.avg_amount_20d)}，当前成交 {_human_amount_yi(row.amount)}",
            f"离日内低点回升 {to_float(row.rebound_from_low_pct):.2f}%，距 MA20 {to_float(row.dist_to_ma20_pct):.2f}%",
        ]
        warnings: list[str] = []
        if to_float(row.unlock_ratio_30d):
            warnings.append(f"未来30日解禁比 {to_float(row.unlock_ratio_30d):.2f}%")
        if to_float(row.holder_decrease_ratio_60d):
            warnings.append(f"近60日股东减持 {to_float(row.holder_decrease_ratio_60d):.2f}%")
        if to_float(row.express_yoy_net_profit) is not None:
            reasons.append(f"最近业绩快报净利同比 {to_float(row.express_yoy_net_profit):.2f}%")
        if to_float(row.circ_mv) is not None:
            reasons.append(f"流通市值 {_human_mv_yi(row.circ_mv)}")
        candidates.append(
            ReboundCandidate(
                ts_code=row.ts_code,
                name=row.name,
                trade_date=trade_date,
                mode=mode,
                score=round(to_float(row.score) or 0.0, 2),
                pct_chg=to_float(row.pct_chg),
                industry=row.industry,
                industry_pct_chg=to_float(row.industry_pct_chg),
                excess_drop_pct=to_float(row.excess_drop_pct),
                close=close,
                pre_close=pre_close,
                rebound_from_low_pct=to_float(row.rebound_from_low_pct),
                close_vs_open_pct=to_float(row.close_vs_open_pct),
                avg_amount_20d=to_float(row.avg_amount_20d),
                latest_amount=to_float(row.amount),
                amount_ratio_20d=to_float(row.amount_ratio_20d),
                circ_mv=to_float(row.circ_mv),
                turnover_rate=to_float(row.turnover_rate),
                volume_ratio=to_float(row.volume_ratio),
                ma20=to_float(row.ma20),
                ma60=to_float(row.ma60),
                dist_to_ma20_pct=to_float(row.dist_to_ma20_pct),
                dist_to_ma60_pct=to_float(row.dist_to_ma60_pct),
                ma20_slope_5d_pct=to_float(row.ma20_slope_5d_pct),
                holder_decrease_ratio_60d=to_float(row.holder_decrease_ratio_60d),
                unlock_ratio_30d=to_float(row.unlock_ratio_30d),
                entry_low=entry_low,
                entry_high=entry_high,
                stop_loss=stop_loss,
                next_day_target=next_day_target,
                reasons=reasons,
                warnings=warnings,
            )
        )
    return candidates


def _fetch_trade_dates(client: TushareClient, requested_date: str, count: int) -> tuple[str, list[str], TableMeta]:
    trade_cal, meta = client.call(
        "trade_cal",
        exchange="",
        start_date=date_before(requested_date, count * 3),
        end_date=requested_date,
    )
    actual_as_of = choose_latest_trade_date(trade_cal, requested_date)
    work = trade_cal.copy()
    work["cal_date"] = work["cal_date"].astype(str)
    open_days = work[(work["is_open"].astype(str) == "1") & (work["cal_date"] <= actual_as_of)]["cal_date"].tolist()
    open_days = sorted(open_days)
    return actual_as_of, open_days, meta


def _fetch_market_daily_history(client: TushareClient, trade_dates: list[str]) -> tuple[pd.DataFrame, list[TableMeta]]:
    frames: list[pd.DataFrame] = []
    metas: list[TableMeta] = []
    for trade_date in trade_dates:
        df, meta = client.call("daily", label=f"daily_{trade_date}", trade_date=trade_date)
        metas.append(meta)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), metas
    return pd.concat(frames, ignore_index=True), metas


def _write_markdown(candidates: list[ReboundCandidate], warnings: list[str], trade_date: str, mode: str) -> str:
    lines = [
        f"# 尾盘反抽候选 {trade_date}",
        "",
        f"- 模式: {mode}",
        f"- 候选数: {len(candidates)}",
        f"- 警告: {compact_join(warnings, limit=6)}",
        "",
    ]
    if not candidates:
        lines.append("无符合条件的候选。")
        return "\n".join(lines)
    for index, candidate in enumerate(candidates, start=1):
        lines.extend(
            [
                f"## {index}. {candidate.name} ({candidate.ts_code})",
                "",
                f"- 得分: {candidate.score:.2f}",
                f"- 跌幅: {candidate.pct_chg:.2f}% / 行业超跌: {candidate.excess_drop_pct:.2f}%",
                f"- 观察区间: {candidate.entry_low} ~ {candidate.entry_high}",
                f"- 风控位: {candidate.stop_loss}",
                f"- 次日先看: {candidate.next_day_target}",
                f"- 入选原因: {compact_join(candidate.reasons, limit=4)}",
                f"- 风险提示: {compact_join(candidate.warnings, limit=4)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_rebound_outputs(
    candidates: list[ReboundCandidate],
    out_dir: Path | str,
    trade_date: str,
    mode: str,
    warnings: list[str] | None = None,
) -> ReboundOutputPaths:
    output_dir = Path(out_dir) / f"rebound_{trade_date}_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_md = output_dir / "summary.md"
    summary_json = output_dir / "summary.json"
    candidates_csv = output_dir / "candidates.csv"
    warning_list = warnings or []
    pd.DataFrame([asdict(item) for item in candidates]).to_csv(candidates_csv, index=False)
    markdown = _write_markdown(candidates, warning_list, trade_date, mode)
    summary_md.write_text(markdown, encoding="utf-8")
    write_json(
        summary_json,
        {
            "trade_date": trade_date,
            "mode": mode,
            "warnings": warning_list,
            "candidate_count": len(candidates),
            "candidates": [asdict(item) for item in candidates],
        },
    )
    return ReboundOutputPaths(
        output_dir=output_dir,
        summary_md=summary_md,
        summary_json=summary_json,
        candidates_csv=candidates_csv,
    )


def _date_to_ts_code_map(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    work = df.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    return {trade_date: group.copy() for trade_date, group in work.groupby("trade_date", sort=True)}


def _express_window_by_trade_date(express_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if express_df is None or express_df.empty or "ann_date" not in express_df.columns:
        return pd.DataFrame()
    work = express_df.copy()
    work["ann_date"] = work["ann_date"].astype(str)
    return work[work["ann_date"] <= trade_date].copy()


def _holdertrade_window_by_trade_date(holdertrade_df: pd.DataFrame, trade_date: str, lookback_days: int) -> pd.DataFrame:
    if holdertrade_df is None or holdertrade_df.empty or "ann_date" not in holdertrade_df.columns:
        return pd.DataFrame()
    start_date = date_before(trade_date, lookback_days)
    work = holdertrade_df.copy()
    work["ann_date"] = work["ann_date"].astype(str)
    return work[(work["ann_date"] >= start_date) & (work["ann_date"] <= trade_date)].copy()


def _share_float_window_by_trade_date(share_float_df: pd.DataFrame, trade_date: str, lookahead_days: int) -> pd.DataFrame:
    if share_float_df is None or share_float_df.empty or "float_date" not in share_float_df.columns:
        return pd.DataFrame()
    end_date = date_after(trade_date, lookahead_days)
    work = share_float_df.copy()
    work["float_date"] = work["float_date"].astype(str)
    return work[(work["float_date"] > trade_date) & (work["float_date"] <= end_date)].copy()


def _safe_return(base: float | None, target: float | None) -> float | None:
    if base is None or target is None or base == 0:
        return None
    return round((target / base - 1) * 100, 2)


def _exit_outcome(
    buy_close: float | None,
    next_open: float | None,
    next_high: float | None,
    next_close: float | None,
    settings: dict[str, Any],
) -> tuple[str, float | None, float | None]:
    if buy_close is None:
        return "unknown", None, None
    gap_stop = float(settings.get("backtest_gap_stop_pct", -4.0))
    intraday_stop = float(settings.get("backtest_intraday_stop_pct", -3.0))
    take_profit = float(settings.get("backtest_take_profit_pct", 2.0))

    open_ret = _safe_return(buy_close, next_open)
    if open_ret is not None and open_ret <= gap_stop:
        return "gap_stop_open", open_ret, next_open
    if open_ret is not None and open_ret >= take_profit:
        exit_price = round(buy_close * (1 + take_profit / 100), 4)
        return "take_profit_open", round(take_profit, 2), exit_price

    high_ret = _safe_return(buy_close, next_high)
    if high_ret is not None and high_ret >= take_profit:
        exit_price = round(buy_close * (1 + take_profit / 100), 4)
        return "take_profit_intraday", round(take_profit, 2), exit_price

    stop_price = round(buy_close * (1 + intraday_stop / 100), 4)
    if next_open is not None and next_open <= stop_price:
        return "intraday_stop_open", _safe_return(buy_close, next_open), next_open
    return "close_exit", _safe_return(buy_close, next_close), next_close


def _mean_or_none(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 2)


def _write_backtest_markdown(summary: ReboundBacktestSummary) -> str:
    lines = [
        f"# 尾盘反抽回测 {summary.start_date} - {summary.end_date}",
        "",
        f"- 出场规则: {summary.exit_rule}",
        f"- 有效交易日: {summary.selection_count}",
        f"- 空仓日: {summary.skipped_days}",
        f"- 次日开盘平均收益: {summary.avg_open_return_pct}%",
        f"- 次日最高平均收益: {summary.avg_high_return_pct}%",
        f"- 次日收盘平均收益: {summary.avg_close_return_pct}%",
        f"- 策略实际平均收益: {summary.avg_exit_return_pct}%",
        f"- 次日开盘胜率: {summary.win_rate_open_pct}%",
        f"- 次日收盘胜率: {summary.win_rate_close_pct}%",
        f"- 策略实际胜率: {summary.win_rate_exit_pct}%",
        f"- 单笔最好收盘收益: {summary.best_trade_pct}%",
        f"- 单笔最差收盘收益: {summary.worst_trade_pct}%",
        "",
        "## 最近交易",
        "",
    ]
    if not summary.trades:
        lines.append("无交易。")
        return "\n".join(lines)
    for trade in summary.trades[-10:]:
        lines.extend(
            [
                f"- {trade.trade_date} {trade.name}({trade.ts_code}) "
                f"买入收盘 {trade.buy_close} -> 次日收盘 {trade.next_close} "
                f"收盘收益 {trade.close_return_pct}%，策略收益 {trade.exit_return_pct}%，次日最高 {trade.high_return_pct}%",
            ]
        )
    return "\n".join(lines)


def write_rebound_backtest_outputs(
    summary: ReboundBacktestSummary,
    out_dir: Path | str,
) -> ReboundBacktestPaths:
    output_dir = Path(out_dir) / f"rebound_backtest_{summary.start_date}_{summary.end_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_md = output_dir / "summary.md"
    summary_json = output_dir / "summary.json"
    trades_csv = output_dir / "trades.csv"
    pd.DataFrame([asdict(item) for item in summary.trades]).to_csv(trades_csv, index=False)
    summary_md.write_text(_write_backtest_markdown(summary), encoding="utf-8")
    write_json(summary_json, asdict(summary))
    return ReboundBacktestPaths(
        output_dir=output_dir,
        summary_md=summary_md,
        summary_json=summary_json,
        trades_csv=trades_csv,
    )


def pick_rebound_stocks(
    as_of_date: str = "",
    top_n: int = 5,
    out_dir: Path | str = "reports",
    config_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    no_cache: bool = False,
    retries: int = 2,
    sleep_sec: float = 0.2,
    historical: bool = False,
) -> tuple[list[ReboundCandidate], ReboundOutputPaths, list[str]]:
    config = load_config(config_path)
    settings = _strategy_settings(config)
    client = TushareClient(cache_dir=cache_dir, retries=retries, sleep_sec=sleep_sec, use_cache=not no_cache)
    warnings: list[str] = []

    requested_date = pd.to_datetime(as_of_date, format="%Y%m%d").strftime("%Y%m%d") if as_of_date else pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
    stock_basic, stock_meta = client.call(
        "stock_basic",
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,list_status",
    )
    if stock_basic.empty:
        raise TushareClientError(f"stock_basic 返回为空: {stock_meta.error or '未知错误'}")
    actual_as_of, open_days, cal_meta = _fetch_trade_dates(client, requested_date, settings["history_trade_days"] + 10)
    live_mode = (not historical) and (actual_as_of == pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d"))
    if live_mode:
        history_dates = open_days[:-1][-settings["history_trade_days"] :]
        if not history_dates:
            raise TushareClientError("实时模式缺少上一交易日历史数据。")
        snapshot, rt_meta = client.call(
            "rt_k",
            label=f"rt_k_{actual_as_of}",
            ts_code=settings["realtime_ts_code"],
            fields=settings["realtime_fields"],
        )
        if snapshot.empty:
            if rt_meta.error and "频率超限" in rt_meta.error:
                raise TushareClientError("rt_k 触发官方限频（1次/分钟），请稍后 1 分钟再试，或改用 --historical 回放。")
            raise TushareClientError(f"rt_k 返回为空: {rt_meta.error or '可能受频率限制或权限限制'}")
        basic_date = history_dates[-1]
        now_cn = pd.Timestamp.now(tz="Asia/Shanghai")
        if now_cn.strftime("%H:%M") < settings["tail_buy_start_time"]:
            warnings.append(f"当前时间 {now_cn.strftime('%H:%M')} 早于建议执行时间 {settings['tail_buy_start_time']}")
    else:
        history_dates = open_days[-settings["history_trade_days"] :]
        basic_date = actual_as_of

    history, history_metas = _fetch_market_daily_history(client, history_dates if live_mode else history_dates)
    if history.empty:
        raise TushareClientError("daily 历史行情为空，无法构建尾盘反抽候选。")
    if not live_mode:
        snapshot = history[history["trade_date"].astype(str) == actual_as_of].copy()
        if snapshot.empty:
            raise TushareClientError(f"未找到 {actual_as_of} 的日线快照。")

    daily_basic, basic_meta = client.call(
        "daily_basic",
        label=f"daily_basic_{basic_date}",
        trade_date=basic_date,
    )
    st_df, st_meta = client.call("stock_st", label=f"stock_st_{actual_as_of}", date=actual_as_of)
    express_df, express_meta = client.call(
        "express",
        start_date=date_before(actual_as_of, settings["express_lookback_days"]),
        end_date=actual_as_of,
    )
    holdertrade_df, holder_meta = client.call(
        "stk_holdertrade",
        start_date=date_before(actual_as_of, settings["holdertrade_lookback_days"]),
        end_date=actual_as_of,
    )
    share_float_df, share_meta = client.call(
        "share_float",
        start_date=actual_as_of,
        end_date=date_after(actual_as_of, settings["unlock_lookahead_days"]),
    )

    for meta in [stock_meta, cal_meta, basic_meta, st_meta, express_meta, holder_meta, share_meta, *history_metas]:
        if meta.error:
            warnings.append(f"{meta.endpoint}: {meta.error}")

    candidates = build_rebound_candidates(
        stock_basic=stock_basic,
        history=history,
        snapshot=snapshot,
        trade_date=actual_as_of,
        mode="live" if live_mode else "historical",
        daily_basic=daily_basic,
        st_df=st_df,
        express_df=express_df,
        holdertrade_df=holdertrade_df,
        share_float_df=share_float_df,
        config=config,
    )[:top_n]
    paths = write_rebound_outputs(candidates, out_dir=out_dir, trade_date=actual_as_of, mode="live" if live_mode else "historical", warnings=warnings)
    return candidates, paths, warnings


def backtest_rebound_strategy(
    start_date: str,
    end_date: str,
    top_n: int = 1,
    out_dir: Path | str = "reports",
    config_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    no_cache: bool = False,
    retries: int = 2,
    sleep_sec: float = 0.2,
) -> tuple[ReboundBacktestSummary, ReboundBacktestPaths]:
    config = load_config(config_path)
    settings = _strategy_settings(config)
    client = TushareClient(cache_dir=cache_dir, retries=retries, sleep_sec=sleep_sec, use_cache=not no_cache)

    stock_basic, _ = client.call(
        "stock_basic",
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,list_status",
    )
    trade_cal, _ = client.call("trade_cal", exchange="", start_date=date_before(start_date, 260), end_date=end_date)
    work = trade_cal.copy()
    work["cal_date"] = work["cal_date"].astype(str)
    open_days = sorted(work[(work["is_open"].astype(str) == "1") & (work["cal_date"] >= start_date) & (work["cal_date"] <= end_date)]["cal_date"].tolist())
    if len(open_days) < 2:
        raise TushareClientError("回测区间内有效交易日不足 2 天。")

    history_start = date_before(open_days[0], 140)
    history_dates = sorted(work[(work["is_open"].astype(str) == "1") & (work["cal_date"] >= history_start) & (work["cal_date"] <= end_date)]["cal_date"].tolist())
    daily_frames: list[pd.DataFrame] = []
    daily_basic_by_date: dict[str, pd.DataFrame] = {}
    st_by_date: dict[str, pd.DataFrame] = {}
    for trade_date in history_dates:
        daily_df, _ = client.call("daily", label=f"daily_{trade_date}", trade_date=trade_date)
        if not daily_df.empty:
            daily_frames.append(daily_df)
        if trade_date >= start_date:
            daily_basic_df, _ = client.call("daily_basic", label=f"daily_basic_{trade_date}", trade_date=trade_date)
            daily_basic_by_date[trade_date] = daily_basic_df
            st_df, _ = client.call("stock_st", label=f"stock_st_{trade_date}", date=trade_date)
            st_by_date[trade_date] = st_df
    if not daily_frames:
        raise TushareClientError("daily 历史行情为空，无法回测。")
    all_daily = pd.concat(daily_frames, ignore_index=True)
    daily_by_date = _date_to_ts_code_map(all_daily)

    express_df, _ = client.call("express", start_date=date_before(start_date, settings["express_lookback_days"]), end_date=end_date)
    holdertrade_df, _ = client.call("stk_holdertrade", start_date=date_before(start_date, settings["holdertrade_lookback_days"]), end_date=end_date)
    share_float_df, _ = client.call("share_float", start_date=start_date, end_date=date_after(end_date, settings["unlock_lookahead_days"]))

    trades: list[ReboundBacktestTrade] = []
    skipped_days = 0
    for idx in range(settings["history_trade_days"], len(open_days) - 1):
        trade_date = open_days[idx]
        next_trade_date = open_days[idx + 1]
        snapshot = daily_by_date.get(trade_date)
        next_snapshot = daily_by_date.get(next_trade_date)
        if snapshot is None or snapshot.empty or next_snapshot is None or next_snapshot.empty:
            skipped_days += 1
            continue
        history_slice_dates = open_days[idx - settings["history_trade_days"] : idx + 1]
        history_slice = all_daily[all_daily["trade_date"].astype(str).isin(history_slice_dates)].copy()
        candidates = build_rebound_candidates(
            stock_basic=stock_basic,
            history=history_slice,
            snapshot=snapshot,
            trade_date=trade_date,
            mode="historical",
            daily_basic=daily_basic_by_date.get(trade_date),
            st_df=st_by_date.get(trade_date),
            express_df=_express_window_by_trade_date(express_df, trade_date),
            holdertrade_df=_holdertrade_window_by_trade_date(holdertrade_df, trade_date, settings["holdertrade_lookback_days"]),
            share_float_df=_share_float_window_by_trade_date(share_float_df, trade_date, settings["unlock_lookahead_days"]),
            config=config,
        )
        if not candidates:
            skipped_days += 1
            continue
        for candidate in candidates[:top_n]:
            next_row = next_snapshot[next_snapshot["ts_code"].astype(str) == candidate.ts_code]
            if next_row.empty:
                continue
            row = next_row.iloc[0]
            buy_close = candidate.close
            next_open = to_float(row.get("open"))
            next_high = to_float(row.get("high"))
            next_close = to_float(row.get("close"))
            exit_rule, exit_return_pct, exit_price = _exit_outcome(buy_close, next_open, next_high, next_close, settings)
            trades.append(
                ReboundBacktestTrade(
                    trade_date=trade_date,
                    ts_code=candidate.ts_code,
                    name=candidate.name,
                    score=candidate.score,
                    buy_close=buy_close,
                    next_trade_date=next_trade_date,
                    next_open=next_open,
                    next_high=next_high,
                    next_close=next_close,
                    open_return_pct=_safe_return(buy_close, next_open),
                    high_return_pct=_safe_return(buy_close, next_high),
                    close_return_pct=_safe_return(buy_close, next_close),
                    exit_rule=exit_rule,
                    exit_return_pct=exit_return_pct,
                    exit_price=exit_price,
                    hit_open=(_safe_return(buy_close, next_open) or -999) > 0,
                    hit_close=(_safe_return(buy_close, next_close) or -999) > 0,
                    reasons=candidate.reasons,
                    warnings=candidate.warnings,
                )
            )

    close_returns = [trade.close_return_pct for trade in trades]
    open_returns = [trade.open_return_pct for trade in trades]
    high_returns = [trade.high_return_pct for trade in trades]
    exit_returns = [trade.exit_return_pct for trade in trades]
    close_clean = [value for value in close_returns if value is not None]
    exit_clean = [value for value in exit_returns if value is not None]
    summary = ReboundBacktestSummary(
        start_date=start_date,
        end_date=end_date,
        exit_rule=f"次日冲高达到 {settings['backtest_take_profit_pct']}% 止盈，否则按止损/收盘退出",
        selection_count=len(trades),
        skipped_days=skipped_days,
        avg_open_return_pct=_mean_or_none(open_returns),
        avg_high_return_pct=_mean_or_none(high_returns),
        avg_close_return_pct=_mean_or_none(close_returns),
        avg_exit_return_pct=_mean_or_none(exit_returns),
        win_rate_open_pct=round(sum(1 for value in open_returns if value is not None and value > 0) / len([v for v in open_returns if v is not None]) * 100, 2) if [v for v in open_returns if v is not None] else None,
        win_rate_close_pct=round(sum(1 for value in close_returns if value is not None and value > 0) / len(close_clean) * 100, 2) if close_clean else None,
        win_rate_exit_pct=round(sum(1 for value in exit_returns if value is not None and value > 0) / len(exit_clean) * 100, 2) if exit_clean else None,
        best_trade_pct=round(max(exit_clean), 2) if exit_clean else None,
        worst_trade_pct=round(min(exit_clean), 2) if exit_clean else None,
        trades=trades,
    )
    paths = write_rebound_backtest_outputs(summary, out_dir=out_dir)
    return summary, paths
