from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


NEGATIVE_FORECAST_TYPES = {"预减", "略减", "首亏", "续亏", "续亏", "增亏", "减亏"}
POSITIVE_FORECAST_TYPES = {"预增", "略增", "扭亏", "续盈"}


@dataclass(frozen=True)
class FinancialReportCatalystConfig:
    screen_end_date: str
    report_lookahead_days: int = 45
    recent_confirmed_days: int = 7
    forecast_lookback_days: int = 180
    history_trade_days: int = 60
    price_min: float = 3.0
    price_max: float = 35.0
    min_amount: float = 40000.0
    min_turnover_rate: float = 0.45
    max_pe_ttm: float = 60.0
    max_pb: float = 6.0
    max_ps_ttm: float = 8.0
    max_position_60d: float = 0.82
    min_return_20d: float = -12.0
    min_profit_yoy: float = 12.0
    min_roe: float = 5.0
    max_fina_candidates: int = 350
    api_sleep_sec: float = 0.05

    @classmethod
    def for_end_date(cls, screen_end_date: str, **overrides: Any) -> "FinancialReportCatalystConfig":
        clean_date = normalize_trade_day(screen_end_date)
        return cls(screen_end_date=clean_date, **overrides)


def normalize_trade_day(value: str) -> str:
    trade_day = str(value or "").strip().replace("-", "")
    if len(trade_day) != 8 or not trade_day.isdigit():
        raise ValueError(f"invalid trade day: {value}")
    return trade_day


def to_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def number_or(value: Any, default: float = 0.0) -> float:
    number = to_number(value)
    return default if number is None else number


def parse_yyyymmdd(value: Any) -> pd.Timestamp | None:
    text = str(value or "").strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        return None
    ts = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def yyyymmdd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def recent_report_periods(screen_end_date: str, count: int = 4) -> list[str]:
    screen_ts = pd.Timestamp(normalize_trade_day(screen_end_date))
    quarter_months = (3, 6, 9, 12)
    periods: list[pd.Timestamp] = []
    year = screen_ts.year
    for candidate_year in range(year, year - 3, -1):
        for month in reversed(quarter_months):
            period = pd.Timestamp(year=candidate_year, month=month, day=31 if month == 3 or month == 12 else 30)
            if period <= screen_ts:
                periods.append(period)
    return [yyyymmdd(period) for period in sorted(periods, reverse=True)[:count]]


def _copy_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _valid_stock_basic(stock_basic: pd.DataFrame) -> pd.DataFrame:
    df = _copy_frame(stock_basic)
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "name", "industry"])
    df["ts_code"] = df["ts_code"].astype(str)
    if "list_status" in df.columns:
        df = df[df["list_status"].astype(str).str.upper().eq("L")].copy()
    name = df.get("name", pd.Series("", index=df.index)).astype(str)
    df = df[~name.str.contains("ST|退", regex=True, na=False)].copy()
    keep = [col for col in ("ts_code", "name", "industry", "market", "list_date") if col in df.columns]
    return df[keep].drop_duplicates("ts_code")


def _latest_prefixed(df: pd.DataFrame, prefix: str, screen_end_date: str) -> pd.DataFrame:
    work = _copy_frame(df)
    if work.empty or "ts_code" not in work.columns:
        return pd.DataFrame(columns=["ts_code"])
    work["ts_code"] = work["ts_code"].astype(str)
    if "ann_date" in work.columns:
        work["ann_date"] = work["ann_date"].astype(str).str.replace("-", "", regex=False)
        work = work[work["ann_date"].le(screen_end_date)].copy()
    sort_cols = [col for col in ("ann_date", "end_date") if col in work.columns]
    if sort_cols:
        work = work.sort_values(["ts_code", *sort_cols])
    work = work.groupby("ts_code", as_index=False).tail(1)
    rename = {col: f"{prefix}_{col}" for col in work.columns if col != "ts_code"}
    return work.rename(columns=rename)


def _prepare_disclosure(disclosure: pd.DataFrame, config: FinancialReportCatalystConfig) -> pd.DataFrame:
    work = _copy_frame(disclosure)
    if work.empty or "ts_code" not in work.columns:
        return pd.DataFrame(columns=["ts_code"])
    screen_ts = pd.Timestamp(config.screen_end_date)
    lookahead_end = screen_ts + pd.Timedelta(days=config.report_lookahead_days)
    confirmed_start = screen_ts - pd.Timedelta(days=config.recent_confirmed_days)
    rows: list[dict[str, Any]] = []

    for row in work.to_dict(orient="records"):
        ann_ts = parse_yyyymmdd(row.get("ann_date"))
        if ann_ts is None or ann_ts > screen_ts:
            continue
        pre_ts = parse_yyyymmdd(row.get("pre_date"))
        actual_ts = parse_yyyymmdd(row.get("actual_date"))
        catalyst_ts: pd.Timestamp | None = None
        catalyst_type = ""
        if actual_ts is not None and confirmed_start <= actual_ts <= screen_ts:
            catalyst_ts = actual_ts
            catalyst_type = "confirmed"
        elif pre_ts is not None and screen_ts <= pre_ts <= lookahead_end:
            catalyst_ts = pre_ts
            catalyst_type = "upcoming"
        if catalyst_ts is None:
            continue
        distance = abs((catalyst_ts - screen_ts).days)
        rows.append(
            {
                "ts_code": str(row.get("ts_code") or "").strip(),
                "report_period": str(row.get("end_date") or "").strip(),
                "disclosure_ann_date": yyyymmdd(ann_ts),
                "report_catalyst_date": yyyymmdd(catalyst_ts),
                "report_catalyst_type": catalyst_type,
                "report_catalyst_distance_days": int(distance),
                "report_catalyst_score": catalyst_score(distance, catalyst_type, config),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["ts_code"])
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["ts_code", "report_catalyst_score", "report_catalyst_distance_days"],
        ascending=[True, False, True],
    )
    return result.groupby("ts_code", as_index=False).head(1).reset_index(drop=True)


def catalyst_score(distance_days: int, catalyst_type: str, config: FinancialReportCatalystConfig) -> float:
    if catalyst_type == "confirmed":
        base = 22.0
        span = max(config.recent_confirmed_days, 1)
    else:
        base = 26.0
        span = max(config.report_lookahead_days, 1)
    decay = max(0.0, 1.0 - (float(distance_days) / float(span)))
    return round(base * (0.35 + 0.65 * decay), 4)


def _history_features(history: pd.DataFrame) -> pd.DataFrame:
    work = _copy_frame(history)
    if work.empty or "ts_code" not in work.columns or "close" not in work.columns:
        return pd.DataFrame(columns=["ts_code", "return_20d", "return_60d", "position_60d"])
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work.get("trade_date", "").astype(str)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna(subset=["close"]).sort_values(["ts_code", "trade_date"])
    rows: list[dict[str, Any]] = []
    for ts_code, sub in work.groupby("ts_code"):
        closes = sub["close"].tolist()
        latest = closes[-1]
        first = closes[0]
        close_20 = closes[-20] if len(closes) >= 20 else first
        low = min(closes)
        high = max(closes)
        span = high - low
        position = 0.5 if span <= 0 else (latest - low) / span
        rows.append(
            {
                "ts_code": ts_code,
                "return_20d": round((latest / close_20 - 1.0) * 100.0, 4) if close_20 else None,
                "return_60d": round((latest / first - 1.0) * 100.0, 4) if first else None,
                "position_60d": round(position, 4),
            }
        )
    return pd.DataFrame(rows)


def _forecast_negative(row: pd.Series) -> bool:
    forecast_type = str(row.get("f_type") or "").strip()
    min_change = to_number(row.get("f_p_change_min"))
    max_change = to_number(row.get("f_p_change_max"))
    if forecast_type in NEGATIVE_FORECAST_TYPES:
        return True
    if "亏" in forecast_type and forecast_type != "扭亏":
        return True
    if max_change is not None and max_change < 0:
        return True
    if min_change is not None and min_change < -20:
        return True
    return False


def _growth_score(value: float | None, scale: float, cap: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value) / scale, 1.0) * cap)


def _score_row(row: pd.Series, config: FinancialReportCatalystConfig) -> dict[str, float | bool]:
    forecast_type = str(row.get("f_type") or "").strip()
    f_min = to_number(row.get("f_p_change_min"))
    f_max = to_number(row.get("f_p_change_max"))
    forecast_avg = None
    if f_min is not None and f_max is not None:
        forecast_avg = (f_min + f_max) / 2.0
    elif f_min is not None:
        forecast_avg = f_min
    elif f_max is not None:
        forecast_avg = f_max
    forecast_score = _growth_score(forecast_avg, 80.0, 24.0)
    if forecast_type in POSITIVE_FORECAST_TYPES:
        forecast_score += 4.0
    if f_min is not None and f_min >= 20:
        forecast_score += 4.0

    express_yoy = to_number(row.get("e_yoy_net_profit"))
    express_roe = to_number(row.get("e_diluted_roe"))
    express_income = to_number(row.get("e_n_income"))
    express_score = _growth_score(express_yoy, 80.0, 22.0) + _growth_score(express_roe, 15.0, 8.0)
    if express_income is not None and express_income <= 0:
        express_score = 0.0

    fina_yoy = max(
        to_number(row.get("fi_dt_netprofit_yoy")) or -999.0,
        to_number(row.get("fi_netprofit_yoy")) or -999.0,
    )
    revenue_yoy = max(
        to_number(row.get("fi_or_yoy")) or -999.0,
        to_number(row.get("fi_q_sales_yoy")) or -999.0,
        to_number(row.get("fi_tr_yoy")) or -999.0,
    )
    roe = to_number(row.get("fi_roe"))
    gross_margin = max(
        to_number(row.get("fi_gross_margin")) or -999.0,
        to_number(row.get("fi_grossprofit_margin")) or -999.0,
    )
    ocfps = to_number(row.get("fi_ocfps"))
    fina_score = _growth_score(fina_yoy, 80.0, 18.0) + _growth_score(revenue_yoy, 50.0, 8.0)
    fina_score += _growth_score(roe, 15.0, 8.0) + _growth_score(gross_margin, 45.0, 4.0)
    if ocfps is not None and ocfps > 0:
        fina_score += 3.0

    quality_score = forecast_score + express_score + fina_score
    has_quality = (
        (forecast_avg is not None and forecast_avg >= config.min_profit_yoy)
        or (express_yoy is not None and express_yoy >= config.min_profit_yoy and (express_income is None or express_income > 0))
        or (fina_yoy >= config.min_profit_yoy and (roe is None or roe >= config.min_roe))
    )

    pe_ttm = to_number(row.get("pe_ttm"))
    pb = to_number(row.get("pb"))
    ps_ttm = to_number(row.get("ps_ttm"))
    close = to_number(row.get("close"))
    valuation_score = 0.0
    if pe_ttm is not None and 0 < pe_ttm <= config.max_pe_ttm:
        valuation_score += max(0.0, 10.0 * (1.0 - pe_ttm / config.max_pe_ttm))
    if pb is not None and pb <= config.max_pb:
        valuation_score += max(0.0, 6.0 * (1.0 - pb / config.max_pb))
    if ps_ttm is not None and ps_ttm <= config.max_ps_ttm:
        valuation_score += max(0.0, 4.0 * (1.0 - ps_ttm / config.max_ps_ttm))

    price_score = 0.0
    if close is not None:
        midpoint = (config.price_min + config.price_max) / 2.0
        half_span = max((config.price_max - config.price_min) / 2.0, 1.0)
        price_score = max(0.0, 10.0 * (1.0 - abs(close - midpoint) / half_span))

    position = to_number(row.get("position_60d"))
    return_20d = to_number(row.get("return_20d"))
    turnover = to_number(row.get("turnover_rate"))
    volume_ratio = to_number(row.get("volume_ratio"))
    market_score = 0.0
    if position is not None:
        market_score += max(0.0, 10.0 * (1.0 - abs(position - 0.45) / 0.45))
    if return_20d is not None:
        market_score += max(0.0, min((return_20d + 8.0) / 20.0, 1.0) * 7.0)
    if turnover is not None:
        market_score += max(0.0, min(turnover / 3.0, 1.0) * 4.0)
    if volume_ratio is not None:
        market_score += max(0.0, min(volume_ratio / 1.5, 1.0) * 3.0)

    report_score = number_or(row.get("report_catalyst_score"))
    total = quality_score + valuation_score + price_score + market_score + report_score
    return {
        "forecast_score": round(forecast_score, 4),
        "express_score": round(express_score, 4),
        "fina_score": round(fina_score, 4),
        "quality_score": round(quality_score, 4),
        "valuation_score": round(valuation_score, 4),
        "price_score": round(price_score, 4),
        "market_score": round(market_score, 4),
        "financial_report_score": round(total, 4),
        "has_quality_evidence": bool(has_quality),
        "forecast_negative": bool(_forecast_negative(row)),
    }


def _coerce_market_frames(daily_basic: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    basic = _copy_frame(daily_basic)
    day = _copy_frame(daily)
    if not basic.empty:
        basic["ts_code"] = basic["ts_code"].astype(str)
    if not day.empty:
        day["ts_code"] = day["ts_code"].astype(str)
    if basic.empty:
        return day
    if day.empty:
        return basic
    duplicate_cols = [col for col in day.columns if col in basic.columns and col != "ts_code"]
    day = day.drop(columns=duplicate_cols)
    return basic.merge(day, on="ts_code", how="left")


def build_financial_report_candidates(
    *,
    config: FinancialReportCatalystConfig,
    stock_basic: pd.DataFrame,
    disclosure: pd.DataFrame,
    forecast: pd.DataFrame,
    express: pd.DataFrame,
    fina_indicator: pd.DataFrame,
    daily_basic: pd.DataFrame,
    daily: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    stocks = _valid_stock_basic(stock_basic)
    disclosure_window = _prepare_disclosure(disclosure, config)
    if stocks.empty or disclosure_window.empty:
        return pd.DataFrame()

    market = _coerce_market_frames(daily_basic, daily)
    history_features = _history_features(history)
    rows = (
        disclosure_window.merge(stocks, on="ts_code", how="inner")
        .merge(market, on="ts_code", how="inner")
        .merge(_latest_prefixed(forecast, "f", config.screen_end_date), on="ts_code", how="left")
        .merge(_latest_prefixed(express, "e", config.screen_end_date), on="ts_code", how="left")
        .merge(_latest_prefixed(fina_indicator, "fi", config.screen_end_date), on="ts_code", how="left")
        .merge(history_features, on="ts_code", how="left")
    )
    if rows.empty:
        return rows

    for col in ("close", "amount", "turnover_rate", "pe_ttm", "pb", "ps_ttm", "position_60d", "return_20d"):
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")

    rows = rows[
        rows["close"].between(config.price_min, config.price_max, inclusive="both")
        & rows["amount"].ge(config.min_amount)
        & rows["turnover_rate"].ge(config.min_turnover_rate)
        & rows["pe_ttm"].gt(0)
        & rows["pe_ttm"].le(config.max_pe_ttm)
        & rows["pb"].le(config.max_pb)
        & rows["ps_ttm"].le(config.max_ps_ttm)
    ].copy()
    if "position_60d" in rows.columns:
        rows = rows[rows["position_60d"].fillna(0.5).le(config.max_position_60d)].copy()
    if "return_20d" in rows.columns:
        rows = rows[rows["return_20d"].fillna(0.0).ge(config.min_return_20d)].copy()
    if rows.empty:
        return rows

    score_rows = rows.apply(lambda row: pd.Series(_score_row(row, config)), axis=1)
    rows = pd.concat([rows.reset_index(drop=True), score_rows.reset_index(drop=True)], axis=1)
    rows = rows[rows["has_quality_evidence"] & ~rows["forecast_negative"]].copy()
    if rows.empty:
        return rows

    rows["trade_date"] = config.screen_end_date
    rows["latest_change_date"] = rows["report_catalyst_date"]
    rows["preferred_pool"] = rows["report_catalyst_type"].map(
        {"upcoming": "earnings_upcoming", "confirmed": "earnings_confirmed"}
    ).fillna("earnings_catalyst")
    rows["priority_score"] = rows["financial_report_score"]
    rows["final_score"] = rows["financial_report_score"]
    rows = rows.sort_values(
        ["financial_report_score", "quality_score", "report_catalyst_score", "amount"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return rows


def configure_tushare_client(token: str, custom_http_url: str = ""):
    import tushare as ts

    ts.set_token(token)
    pro = ts.pro_api(token)
    if custom_http_url:
        pro._DataApi__token = token
        pro._DataApi__http_url = custom_http_url
    return pro


def ensure_token(token: str) -> None:
    if not str(token or "").strip():
        raise SystemExit("TUSHARE_TOKEN is required.")


def _safe_query(name: str, func: Any, **kwargs: Any) -> pd.DataFrame:
    if func is None:
        return pd.DataFrame()
    try:
        return func(**kwargs)
    except Exception as exc:
        print(f"[warn] {name} failed: {exc}")
        return pd.DataFrame()


def _date_range(start_date: str, end_date: str) -> Iterable[str]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    for ts in pd.date_range(start=start, end=end, freq="D"):
        yield yyyymmdd(ts)


def fetch_forecast_range(pro: Any, start_date: str, end_date: str, sleep_sec: float = 0.05) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ann_date in _date_range(start_date, end_date):
        df = _safe_query(f"forecast_{ann_date}", getattr(pro, "forecast", None), ann_date=ann_date)
        if not df.empty:
            frames.append(df)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    frames = [frame.dropna(axis=1, how="all") for frame in frames if not frame.dropna(axis=1, how="all").empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def concat_non_empty(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    return pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()


def fetch_recent_trade_dates(pro: Any, screen_end_date: str, count: int) -> list[str]:
    start = yyyymmdd(pd.Timestamp(screen_end_date) - pd.Timedelta(days=max(90, count * 3)))
    cal = _safe_query("trade_cal", getattr(pro, "trade_cal", None), start_date=start, end_date=screen_end_date)
    if cal.empty or "cal_date" not in cal.columns:
        return list(pd.bdate_range(end=pd.Timestamp(screen_end_date), periods=count).strftime("%Y%m%d"))
    cal = cal[cal.get("is_open", 1).astype(str).eq("1")].copy()
    dates = sorted(cal["cal_date"].astype(str).tolist())
    return dates[-count:]


def fetch_history_by_trade_dates(pro: Any, trade_dates: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        df = _safe_query(f"daily_{trade_date}", getattr(pro, "daily", None), trade_date=trade_date)
        if not df.empty:
            frames.append(df[["ts_code", "trade_date", "close"]].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_fina_indicator_for_codes(
    pro: Any,
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    *,
    sleep_sec: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ts_code in ts_codes:
        df = _safe_query(
            f"fina_indicator_{ts_code}",
            getattr(pro, "fina_indicator", None),
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if not df.empty:
            frames.append(df)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    frames = [frame.dropna(axis=1, how="all") for frame in frames if not frame.dropna(axis=1, how="all").empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _shortlist_for_fina(
    config: FinancialReportCatalystConfig,
    stock_basic: pd.DataFrame,
    disclosure: pd.DataFrame,
    daily_basic: pd.DataFrame,
    daily: pd.DataFrame,
) -> list[str]:
    stocks = _valid_stock_basic(stock_basic)
    window = _prepare_disclosure(disclosure, config)
    market = _coerce_market_frames(daily_basic, daily)
    if stocks.empty or window.empty or market.empty:
        return []
    rows = window.merge(stocks, on="ts_code", how="inner").merge(market, on="ts_code", how="inner")
    for col in ("close", "amount", "turnover_rate", "pe_ttm", "pb", "ps_ttm"):
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows[
        rows["close"].between(config.price_min, config.price_max, inclusive="both")
        & rows["amount"].ge(config.min_amount)
        & rows["turnover_rate"].ge(config.min_turnover_rate)
        & rows["pe_ttm"].gt(0)
        & rows["pe_ttm"].le(config.max_pe_ttm)
        & rows["pb"].le(config.max_pb)
        & rows["ps_ttm"].le(config.max_ps_ttm)
    ].copy()
    rows = rows.sort_values(["report_catalyst_score", "amount"], ascending=[False, False])
    return rows["ts_code"].astype(str).drop_duplicates().head(config.max_fina_candidates).tolist()


def export_financial_report_result(
    *,
    config: FinancialReportCatalystConfig,
    candidates: pd.DataFrame,
    export_root: Path,
    export_prefix: str,
    report_periods: list[str],
) -> Path:
    run_tag = f"{export_prefix}{config.screen_end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = export_root / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "strategy_id": "holder_increase_screening",
        "strategy_name": "财报催化臻选",
        "requested_end_date": config.screen_end_date,
        "screen_end_date": config.screen_end_date,
        "latest_trade_date": config.screen_end_date,
        "report_periods": report_periods,
        "candidate_count": int(len(candidates)),
        "selected_count": int(0 if candidates.empty else 1),
        "config": asdict(config),
        "export_dir": str(export_dir),
    }
    (export_dir / "screen_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates.to_csv(export_dir / "final_candidates.csv", index=False)
    candidates.to_csv(export_dir / "ranked_candidates_stage1.csv", index=False)
    candidates.head(5).to_csv(export_dir / "stable_candidates.csv", index=False)
    pd.DataFrame().to_csv(export_dir / "aggressive_candidates.csv", index=False)
    candidates.head(1).to_csv(export_dir / "best_pick_candidate.csv", index=False)
    return export_dir


def run_financial_report_catalyst_screening(
    config: FinancialReportCatalystConfig,
    *,
    pro: Any,
    export_results: bool = True,
    export_root: Path | None = None,
    export_prefix: str = "financial_report_catalyst_",
) -> dict[str, Any]:
    report_periods = recent_report_periods(config.screen_end_date, count=4)
    disclosure_frames = [
        _safe_query(f"disclosure_date_{period}", getattr(pro, "disclosure_date", None), end_date=period)
        for period in report_periods
    ]
    disclosure = concat_non_empty(disclosure_frames)
    stock_basic = _safe_query(
        "stock_basic",
        getattr(pro, "stock_basic", None),
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,list_status",
    )
    daily_basic = _safe_query("daily_basic", getattr(pro, "daily_basic", None), trade_date=config.screen_end_date)
    daily = _safe_query("daily", getattr(pro, "daily", None), trade_date=config.screen_end_date)
    forecast_start = yyyymmdd(pd.Timestamp(config.screen_end_date) - pd.Timedelta(days=config.forecast_lookback_days))
    forecast = fetch_forecast_range(pro, forecast_start, config.screen_end_date, sleep_sec=config.api_sleep_sec)
    express = _safe_query("express", getattr(pro, "express", None), start_date=forecast_start, end_date=config.screen_end_date)
    trade_dates = fetch_recent_trade_dates(pro, config.screen_end_date, config.history_trade_days)
    history = fetch_history_by_trade_dates(pro, trade_dates)
    fina_start = yyyymmdd(pd.Timestamp(config.screen_end_date) - pd.Timedelta(days=760))
    fina_codes = _shortlist_for_fina(config, stock_basic, disclosure, daily_basic, daily)
    fina_indicator = fetch_fina_indicator_for_codes(
        pro,
        fina_codes,
        fina_start,
        config.screen_end_date,
        sleep_sec=config.api_sleep_sec,
    )
    candidates = build_financial_report_candidates(
        config=config,
        stock_basic=stock_basic,
        disclosure=disclosure,
        forecast=forecast,
        express=express,
        fina_indicator=fina_indicator,
        daily_basic=daily_basic,
        daily=daily,
        history=history,
    )
    export_dir = None
    if export_results:
        root = export_root or Path("output/jupyter-notebook/financial_report_catalyst_exports").resolve()
        export_dir = export_financial_report_result(
            config=config,
            candidates=candidates,
            export_root=root,
            export_prefix=export_prefix,
            report_periods=report_periods,
        )
    summary = {
        "screen_end_date": config.screen_end_date,
        "report_periods": report_periods,
        "candidate_count": int(len(candidates)),
        "selected_count": int(0 if candidates.empty else 1),
    }
    return {
        "screen_summary": summary,
        "candidates": candidates,
        "best_pick_candidate": candidates.head(1).copy(),
        "export_dir": export_dir,
    }
