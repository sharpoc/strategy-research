from __future__ import annotations

import json
import textwrap
from pathlib import Path


def lines(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def build_notebook() -> dict:
    return {
        "cells": [
            markdown_cell(
                """
                # Experiment: Tushare A股全量信息抓取与GPT分析输入

                Objective:
                - 输入一个股票代码、股票简称或 `ts_code`，自动拉取行情、估值、资金流、财务、业绩、股东变化和同业对比。
                - 输出一份适合直接交给 GPT 做二次分析的结构化 JSON 摘要，并把原始表落盘成 CSV。
                """
            ),
            markdown_cell(
                """
                ## 使用方式

                1. 最方便的方式：只改下面配置单元格里的 `STOCK_INPUT`，然后 `Run -> Run All Cells`。
                2. 如果把 `STOCK_INPUT = ""` 留空，运行时会弹出 `input()` 让你手动输入。
                3. `TOKEN` 建议放环境变量 `TUSHARE_TOKEN`；如果没配，也可以直接在单元格里填。
                4. 横向对比默认按 `stock_basic.industry` 自动挑选同业公司，也可以手工指定 `PEER_TS_CODES`。
                5. 结果会写到 `output/jupyter-notebook/tushare_exports/<ts_code>/`。
                """
            ),
            code_cell(
                """
                from __future__ import annotations

                import json
                import os
                import warnings
                from pathlib import Path

                try:
                    from urllib3.exceptions import NotOpenSSLWarning
                    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
                except Exception:
                    warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

                import numpy as np
                import pandas as pd
                import tushare as ts

                warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

                pd.set_option("display.max_columns", 100)
                pd.set_option("display.width", 220)

                CWD = Path.cwd().resolve()
                if CWD.name == "jupyter-notebook" and CWD.parent.name == "output":
                    REPO_ROOT = CWD.parent.parent
                else:
                    REPO_ROOT = CWD

                TODAY = pd.Timestamp.today().normalize()
                TODAY_STR = TODAY.strftime("%Y%m%d")
                DEFAULT_START_DATE = (TODAY - pd.Timedelta(days=540)).strftime("%Y%m%d")
                OUTPUT_DIR = REPO_ROOT / "output" / "jupyter-notebook" / "tushare_exports"
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                print(
                    {
                        "today": TODAY_STR,
                        "default_start_date": DEFAULT_START_DATE,
                        "repo_root": str(REPO_ROOT),
                        "output_dir": str(OUTPUT_DIR.resolve()),
                    }
                )
                """
            ),
            code_cell(
                """
                # 只需要改这一格。
                TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()
                if not TOKEN:
                    TOKEN = ""

                STOCK_INPUT = "002517"  # 支持 600519 / 600519.SH / 贵州茅台
                START_DATE = DEFAULT_START_DATE
                END_DATE = TODAY_STR
                KLINE_LIMIT = 240
                FINANCIAL_LIMIT = 12
                HOLDER_HISTORY_LIMIT = 80
                HK_HOLD_LIMIT = 120
                HOLDER_TRADE_LIMIT = 80
                FUND_PORTFOLIO_QUARTERS = 8
                PEER_LIMIT = 8
                PEER_TS_CODES: list[str] = []  # 例如 ["002555.SZ", "603444.SH"]

                if not STOCK_INPUT.strip():
                    STOCK_INPUT = input("输入股票代码、ts_code 或股票简称: ").strip()
                if not TOKEN.strip():
                    TOKEN = input("输入 Tushare Token: ").strip()

                config_snapshot = {
                    "stock_input": STOCK_INPUT,
                    "start_date": START_DATE,
                    "end_date": END_DATE,
                    "kline_limit": KLINE_LIMIT,
                    "financial_limit": FINANCIAL_LIMIT,
                    "holder_history_limit": HOLDER_HISTORY_LIMIT,
                    "hk_hold_limit": HK_HOLD_LIMIT,
                    "holder_trade_limit": HOLDER_TRADE_LIMIT,
                    "fund_portfolio_quarters": FUND_PORTFOLIO_QUARTERS,
                    "peer_limit": PEER_LIMIT,
                    "peer_ts_codes": PEER_TS_CODES,
                }
                config_snapshot
                """
            ),
            code_cell(
                '''
                def ensure_token(token: str) -> None:
                    if not token or token.startswith("PASTE_"):
                        raise ValueError("缺少 Tushare Token。请设置环境变量 TUSHARE_TOKEN 或直接在配置单元格里填写。")


                def safe_call(label: str, fn, **kwargs) -> pd.DataFrame:
                    try:
                        df = fn(**kwargs)
                    except Exception as exc:
                        print(f"[{label}] 调用失败: {exc}")
                        return pd.DataFrame()
                    if df is None:
                        return pd.DataFrame()
                    return df.copy()


                def sort_desc(df: pd.DataFrame) -> pd.DataFrame:
                    if df.empty:
                        return df
                    for col in ["trade_date", "end_date", "ann_date", "f_ann_date"]:
                        if col in df.columns:
                            return df.sort_values(col, ascending=False).reset_index(drop=True)
                    return df.reset_index(drop=True)


                def latest_row(df: pd.DataFrame) -> dict:
                    ordered = sort_desc(df)
                    if ordered.empty:
                        return {}
                    row = ordered.iloc[0]
                    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


                def latest_period_rows(df: pd.DataFrame, date_col: str = "end_date", limit=None) -> pd.DataFrame:
                    if df.empty or date_col not in df.columns:
                        return pd.DataFrame()
                    ordered = sort_desc(df)
                    latest_value = ordered[date_col].iloc[0]
                    result = ordered[ordered[date_col] == latest_value].reset_index(drop=True)
                    if limit is not None:
                        result = result.head(limit).reset_index(drop=True)
                    return result


                def pick_value(row: dict, *keys: str):
                    for key in keys:
                        if key in row and pd.notna(row[key]):
                            return row[key]
                    return None


                def to_number(value, digits: int = 2):
                    if value is None:
                        return None
                    try:
                        if pd.isna(value):
                            return None
                    except Exception:
                        pass
                    try:
                        return round(float(value), digits)
                    except Exception:
                        return value


                def json_safe(obj):
                    if isinstance(obj, dict):
                        return {str(k): json_safe(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [json_safe(v) for v in obj]
                    if isinstance(obj, tuple):
                        return [json_safe(v) for v in obj]
                    if isinstance(obj, Path):
                        return str(obj)
                    if isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return None if np.isnan(obj) else float(obj)
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    if obj is None:
                        return None
                    try:
                        if pd.isna(obj):
                            return None
                    except Exception:
                        pass
                    return obj


                def frame_tail(df: pd.DataFrame, limit: int = 5) -> list[dict]:
                    if df.empty:
                        return []
                    sample = sort_desc(df).head(limit).copy()
                    sample = sample.replace({np.nan: None})
                    return json_safe(sample.to_dict(orient="records"))


                def resolve_ts_code(stock_basic_all: pd.DataFrame, stock_input: str) -> dict:
                    stock_input = stock_input.strip()
                    basics = stock_basic_all.fillna("")

                    if "." in stock_input:
                        exact = basics[basics["ts_code"].str.upper() == stock_input.upper()]
                        if not exact.empty:
                            return exact.iloc[0].to_dict()

                    symbol_match = basics[basics["symbol"] == stock_input]
                    if not symbol_match.empty:
                        return symbol_match.iloc[0].to_dict()

                    name_match = basics[basics["name"] == stock_input]
                    if not name_match.empty:
                        return name_match.iloc[0].to_dict()

                    fuzzy = basics[
                        basics["name"].str.contains(stock_input, case=False, regex=False)
                        | basics["symbol"].str.contains(stock_input, case=False, regex=False)
                        | basics["ts_code"].str.contains(stock_input, case=False, regex=False)
                    ]
                    if not fuzzy.empty:
                        return fuzzy.iloc[0].to_dict()

                    raise ValueError(f"未找到股票: {stock_input}")


                def exchange_from_ts_code(ts_code: str) -> str:
                    suffix = ts_code.split(".")[-1].upper()
                    mapping = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}
                    return mapping.get(suffix, "")


                def compute_main_net_amount(df: pd.DataFrame) -> pd.Series:
                    if df.empty:
                        return pd.Series(dtype=float)
                    if "net_mf_amount" in df.columns:
                        return pd.to_numeric(df["net_mf_amount"], errors="coerce")
                    buys = pd.Series(0.0, index=df.index)
                    sells = pd.Series(0.0, index=df.index)
                    for col in ["buy_lg_amount", "buy_elg_amount"]:
                        if col in df.columns:
                            buys = buys.add(pd.to_numeric(df[col], errors="coerce").fillna(0.0), fill_value=0.0)
                    for col in ["sell_lg_amount", "sell_elg_amount"]:
                        if col in df.columns:
                            sells = sells.add(pd.to_numeric(df[col], errors="coerce").fillna(0.0), fill_value=0.0)
                    return buys - sells


                def compute_return(daily_df: pd.DataFrame, window: int):
                    ordered = sort_desc(daily_df)
                    if ordered.empty or len(ordered) <= window:
                        return None
                    latest_close = pd.to_numeric(ordered.iloc[0]["close"], errors="coerce")
                    old_close = pd.to_numeric(ordered.iloc[window]["close"], errors="coerce")
                    if pd.isna(latest_close) or pd.isna(old_close) or old_close == 0:
                        return None
                    return (latest_close / old_close - 1.0) * 100


                def average_column(df: pd.DataFrame, column: str, window: int):
                    ordered = sort_desc(df)
                    if ordered.empty or column not in ordered.columns:
                        return None
                    return pd.to_numeric(ordered.head(window)[column], errors="coerce").mean()


                def latest_completed_quarter(end_date_str: str) -> pd.Timestamp:
                    ts_value = pd.Timestamp(end_date_str)
                    quarter_end = (ts_value + pd.offsets.QuarterEnd(0)).normalize()
                    if quarter_end > ts_value.normalize():
                        quarter_end = (ts_value - pd.offsets.QuarterEnd()).normalize()
                    return quarter_end


                def recent_quarter_periods(end_date_str: str, count: int = 8) -> list[str]:
                    periods: list[str] = []
                    current = latest_completed_quarter(end_date_str)
                    for _ in range(count):
                        periods.append(current.strftime("%Y%m%d"))
                        current = (current - pd.offsets.QuarterEnd()).normalize()
                    return periods


                def summarize_value_change(df: pd.DataFrame, date_col: str, value_col: str, days: int) -> dict:
                    if df.empty or date_col not in df.columns or value_col not in df.columns:
                        return {"days": days, "latest": None, "past": None, "change": None, "latest_date": None}
                    work = df.copy()
                    work[date_col] = pd.to_datetime(work[date_col], format="%Y%m%d", errors="coerce")
                    work = work.dropna(subset=[date_col]).sort_values(date_col, ascending=False).reset_index(drop=True)
                    if work.empty:
                        return {"days": days, "latest": None, "past": None, "change": None, "latest_date": None}
                    latest_date = work.loc[0, date_col]
                    latest_value = pd.to_numeric(work.loc[0, value_col], errors="coerce")
                    past_candidates = work[work[date_col] <= latest_date - pd.Timedelta(days=days)]
                    past_value = pd.to_numeric(past_candidates.iloc[0][value_col], errors="coerce") if not past_candidates.empty else np.nan
                    change = latest_value - past_value if not pd.isna(latest_value) and not pd.isna(past_value) else np.nan
                    return {
                        "days": days,
                        "latest_date": latest_date.strftime("%Y%m%d"),
                        "latest": to_number(latest_value),
                        "past": to_number(past_value),
                        "change": to_number(change),
                    }


                def trend_label(change):
                    if change is None:
                        return "无数据"
                    try:
                        if pd.isna(change):
                            return "无数据"
                    except Exception:
                        pass
                    if change > 0:
                        return "增加"
                    if change < 0:
                        return "减少"
                    return "持平"


                def summarize_hk_hold(df: pd.DataFrame) -> dict:
                    if df.empty:
                        return {
                            "available": False,
                            "latest_trade_date": None,
                            "latest_ratio": None,
                            "latest_vol": None,
                            "trend_1m": "无数据",
                            "trend_3m": "无数据",
                            "recent_points": [],
                        }
                    ordered = sort_desc(df)
                    latest = latest_row(ordered)
                    vol_30 = summarize_value_change(ordered, "trade_date", "vol", 30)
                    vol_90 = summarize_value_change(ordered, "trade_date", "vol", 90)
                    ratio_30 = summarize_value_change(ordered, "trade_date", "ratio", 30)
                    ratio_90 = summarize_value_change(ordered, "trade_date", "ratio", 90)
                    return {
                        "available": True,
                        "latest_trade_date": pick_value(latest, "trade_date"),
                        "latest_ratio": to_number(pick_value(latest, "ratio")),
                        "latest_vol": to_number(pick_value(latest, "vol"), 0),
                        "trend_1m": trend_label(vol_30["change"]),
                        "trend_3m": trend_label(vol_90["change"]),
                        "vol_change_1m": vol_30["change"],
                        "vol_change_3m": vol_90["change"],
                        "ratio_change_1m": ratio_30["change"],
                        "ratio_change_3m": ratio_90["change"],
                        "recent_points": frame_tail(ordered, 5),
                    }


                def summarize_holder_category(df: pd.DataFrame, keywords: list[str], label: str, limit_periods: int = 4) -> dict:
                    if df.empty or "holder_name" not in df.columns or "end_date" not in df.columns:
                        return {"label": label, "latest_period": None, "trend": "无数据", "periods": []}
                    pattern = "|".join(keywords)
                    work = df.copy()
                    work["holder_name"] = work["holder_name"].fillna("").astype(str)
                    filtered = work[work["holder_name"].str.contains(pattern, case=False, regex=True)]
                    if filtered.empty:
                        return {"label": label, "latest_period": None, "trend": "无数据", "periods": []}
                    periods: list[dict] = []
                    end_dates = filtered["end_date"].dropna().astype(str).drop_duplicates().tolist()
                    for end_date in end_dates[:limit_periods]:
                        sub = filtered[filtered["end_date"] == end_date].copy()
                        hold_ratio_total = pd.to_numeric(sub["hold_ratio"], errors="coerce").sum() if "hold_ratio" in sub.columns else np.nan
                        hold_amount_total = pd.to_numeric(sub["hold_amount"], errors="coerce").sum() if "hold_amount" in sub.columns else np.nan
                        periods.append(
                            {
                                "end_date": end_date,
                                "holders": sub["holder_name"].head(10).tolist(),
                                "hold_ratio_total": to_number(hold_ratio_total),
                                "hold_amount_total": to_number(hold_amount_total, 0),
                            }
                        )
                    latest_ratio = periods[0]["hold_ratio_total"] if periods else None
                    prev_ratio = periods[1]["hold_ratio_total"] if len(periods) > 1 else None
                    ratio_change = None if latest_ratio is None or prev_ratio is None else to_number(latest_ratio - prev_ratio)
                    return {
                        "label": label,
                        "latest_period": periods[0]["end_date"] if periods else None,
                        "latest_hold_ratio_total": latest_ratio,
                        "prev_hold_ratio_total": prev_ratio,
                        "ratio_change_vs_prev_period": ratio_change,
                        "trend": trend_label(ratio_change),
                        "periods": periods,
                    }


                def summarize_holder_trade(df: pd.DataFrame, recent_days: int = 365) -> dict:
                    if df.empty:
                        return {
                            "window_days": recent_days,
                            "increase_events": 0,
                            "decrease_events": 0,
                            "net_change_vol": None,
                            "latest_events": [],
                        }
                    work = sort_desc(df)
                    work["ann_date_dt"] = pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce")
                    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=recent_days)
                    recent = work[work["ann_date_dt"] >= cutoff].copy()
                    if recent.empty:
                        recent = work.head(20).copy()
                    increase = recent[recent["in_de"] == "IN"].copy()
                    decrease = recent[recent["in_de"] == "DE"].copy()
                    increase_vol = pd.to_numeric(increase["change_vol"], errors="coerce").sum() if not increase.empty else 0.0
                    decrease_vol = pd.to_numeric(decrease["change_vol"], errors="coerce").sum() if not decrease.empty else 0.0
                    preview_cols = [c for c in ["ann_date", "holder_name", "holder_type", "in_de", "change_vol", "change_ratio", "avg_price"] if c in recent.columns]
                    return {
                        "window_days": recent_days,
                        "increase_events": int(len(increase)),
                        "decrease_events": int(len(decrease)),
                        "net_change_vol": to_number(increase_vol - decrease_vol, 0),
                        "trend": trend_label(increase_vol - decrease_vol),
                        "latest_events": frame_tail(recent[preview_cols], 10) if preview_cols else frame_tail(recent, 10),
                    }


                def fetch_fund_portfolio_history(pro, ts_code: str, end_date_str: str, quarter_count: int = 8):
                    periods = recent_quarter_periods(end_date_str, quarter_count)
                    frames: list[pd.DataFrame] = []
                    summaries: list[dict] = []
                    for period in periods:
                        df = safe_call(f"fund_portfolio_{period}", pro.fund_portfolio, ts_code=ts_code, period=period)
                        if not df.empty:
                            df = sort_desc(df)
                            df = df.copy()
                            df["query_period"] = period
                            frames.append(df)
                            summaries.append(
                                {
                                    "period": period,
                                    "funds_count": int(df["symbol"].nunique()) if "symbol" in df.columns else int(len(df)),
                                    "hold_amount_total": to_number(pd.to_numeric(df["amount"], errors="coerce").sum(), 0),
                                    "hold_mkv_total": to_number(pd.to_numeric(df["mkv"], errors="coerce").sum(), 0),
                                    "float_ratio_total": to_number(pd.to_numeric(df["stk_float_ratio"], errors="coerce").sum()),
                                }
                            )
                        else:
                            summaries.append(
                                {
                                    "period": period,
                                    "funds_count": 0,
                                    "hold_amount_total": 0.0,
                                    "hold_mkv_total": 0.0,
                                    "float_ratio_total": 0.0,
                                }
                            )
                    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                    return combined, summaries


                def summarize_periodic_trend(period_summaries: list[dict], value_key: str) -> dict:
                    if not period_summaries:
                        return {"latest_period": None, "trend": "无数据", "history": []}
                    latest = period_summaries[0]
                    previous = period_summaries[1] if len(period_summaries) > 1 else {}
                    latest_value = latest.get(value_key)
                    previous_value = previous.get(value_key)
                    change = None if latest_value is None or previous_value is None else to_number(latest_value - previous_value)
                    return {
                        "latest_period": latest.get("period"),
                        "latest_value": latest_value,
                        "previous_period": previous.get("period"),
                        "previous_value": previous_value,
                        "change_vs_previous": change,
                        "trend": trend_label(change),
                        "history": period_summaries[: min(4, len(period_summaries))],
                    }


                def extract_overseas_revenue_ratio(mainbz_df: pd.DataFrame):
                    latest = latest_period_rows(mainbz_df, "end_date")
                    if latest.empty or "bz_item" not in latest.columns or "bz_sales" not in latest.columns:
                        return None
                    work = latest.copy()
                    work["bz_sales_num"] = pd.to_numeric(work["bz_sales"], errors="coerce").fillna(0.0)
                    total_sales = work["bz_sales_num"].sum()
                    if total_sales <= 0:
                        return None
                    mask = work["bz_item"].fillna("").astype(str).str.contains(
                        "海外|境外|国际|国外|overseas|international|other",
                        case=False,
                        regex=True,
                    )
                    overseas_sales = work.loc[mask, "bz_sales_num"].sum()
                    if overseas_sales <= 0:
                        return None
                    return to_number(overseas_sales / total_sales * 100)


                def extract_mainbz_tags(mainbz_df: pd.DataFrame, limit: int = 3) -> list[str]:
                    latest = latest_period_rows(mainbz_df, "end_date")
                    if latest.empty or "bz_item" not in latest.columns:
                        return []
                    work = latest.copy()
                    if "bz_sales" in work.columns:
                        work["bz_sales_num"] = pd.to_numeric(work["bz_sales"], errors="coerce").fillna(0.0)
                        work = work.sort_values("bz_sales_num", ascending=False)
                    return work["bz_item"].dropna().astype(str).head(limit).tolist()


                def build_peer_candidates(
                    stock_basic_all: pd.DataFrame,
                    resolved: dict,
                    market_daily_basic: pd.DataFrame,
                    manual_peer_ts_codes: list[str],
                    peer_limit: int,
                ):
                    if manual_peer_ts_codes:
                        candidates = stock_basic_all[stock_basic_all["ts_code"].isin(manual_peer_ts_codes)].copy()
                        selection_method = "manual_peer_ts_codes"
                    else:
                        candidates = stock_basic_all[
                            (stock_basic_all["industry"] == resolved.get("industry"))
                            & (stock_basic_all["ts_code"] != resolved.get("ts_code"))
                        ].copy()
                        selection_method = f"same_industry:{resolved.get('industry')}"
                        if candidates.empty:
                            candidates = stock_basic_all[
                                (stock_basic_all["market"] == resolved.get("market"))
                                & (stock_basic_all["ts_code"] != resolved.get("ts_code"))
                            ].copy()
                            selection_method = f"same_market_fallback:{resolved.get('market')}"

                    if not market_daily_basic.empty and "ts_code" in market_daily_basic.columns:
                        market_cols = [
                            c
                            for c in ["ts_code", "trade_date", "close", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv"]
                            if c in market_daily_basic.columns
                        ]
                        candidates = candidates.merge(market_daily_basic[market_cols], on="ts_code", how="left")
                        if "total_mv" in candidates.columns:
                            candidates["total_mv"] = pd.to_numeric(candidates["total_mv"], errors="coerce")
                            candidates = candidates.sort_values(["total_mv", "ts_code"], ascending=[False, True])
                    return candidates.head(peer_limit).reset_index(drop=True), selection_method


                def build_company_metric_snapshot(
                    basic_row: dict,
                    market_row: dict,
                    indicator_row: dict,
                    income_row: dict,
                    balance_row: dict,
                    mainbz_df: pd.DataFrame,
                    target_ts_code: str,
                ) -> dict:
                    revenue = pick_value(income_row, "total_revenue", "revenue")
                    net_profit = pick_value(income_row, "n_income_attr_p", "n_income")
                    total_liab = pick_value(balance_row, "total_liab")
                    money_cap = pick_value(balance_row, "money_cap")
                    netprofit_margin = None
                    cash_to_liab = None
                    peg_ttm = None

                    try:
                        if revenue not in (None, 0) and net_profit is not None:
                            netprofit_margin = float(net_profit) / float(revenue) * 100
                    except Exception:
                        netprofit_margin = None

                    try:
                        if total_liab not in (None, 0) and money_cap is not None:
                            cash_to_liab = float(money_cap) / float(total_liab) * 100
                    except Exception:
                        cash_to_liab = None

                    pe_ttm = pick_value(market_row, "pe_ttm", "pe")
                    netprofit_yoy = pick_value(indicator_row, "netprofit_yoy", "dt_netprofit_yoy")
                    try:
                        if pe_ttm is not None and netprofit_yoy not in (None, 0) and float(netprofit_yoy) > 0:
                            peg_ttm = float(pe_ttm) / float(netprofit_yoy)
                    except Exception:
                        peg_ttm = None

                    return {
                        "ts_code": basic_row.get("ts_code"),
                        "name": basic_row.get("name"),
                        "industry": basic_row.get("industry"),
                        "market": basic_row.get("market"),
                        "is_target": basic_row.get("ts_code") == target_ts_code,
                        "trade_date": pick_value(market_row, "trade_date"),
                        "report_period": pick_value(indicator_row, "end_date") or pick_value(income_row, "end_date") or pick_value(balance_row, "end_date"),
                        "close": to_number(pick_value(market_row, "close")),
                        "total_mv": to_number(pick_value(market_row, "total_mv")),
                        "pe_ttm": to_number(pe_ttm),
                        "pb": to_number(pick_value(market_row, "pb")),
                        "ps_ttm": to_number(pick_value(market_row, "ps_ttm", "ps")),
                        "dv_ttm": to_number(pick_value(market_row, "dv_ttm", "dv_ratio")),
                        "roe": to_number(pick_value(indicator_row, "roe", "q_roe")),
                        "gross_margin": to_number(pick_value(indicator_row, "gross_margin", "grossprofit_margin")),
                        "netprofit_margin": to_number(netprofit_margin),
                        "revenue_yoy": to_number(pick_value(indicator_row, "tr_yoy", "or_yoy", "q_sales_yoy")),
                        "netprofit_yoy": to_number(netprofit_yoy),
                        "peg_ttm": to_number(peg_ttm),
                        "money_cap": to_number(money_cap),
                        "cash_to_liab": to_number(cash_to_liab),
                        "overseas_revenue_ratio": to_number(extract_overseas_revenue_ratio(mainbz_df)),
                        "main_business_tags": ", ".join(extract_mainbz_tags(mainbz_df, 3)),
                    }


                def build_peer_comparison(
                    pro,
                    resolved: dict,
                    peer_candidates_df: pd.DataFrame,
                    market_daily_basic: pd.DataFrame,
                    datasets: dict[str, pd.DataFrame],
                ) -> pd.DataFrame:
                    market_lookup: dict[str, dict] = {}
                    if not market_daily_basic.empty and "ts_code" in market_daily_basic.columns:
                        market_lookup = market_daily_basic.set_index("ts_code").to_dict(orient="index")

                    rows: list[dict] = []
                    rows.append(
                        build_company_metric_snapshot(
                            resolved,
                            market_lookup.get(resolved["ts_code"], latest_row(datasets["daily_basic"])),
                            latest_row(datasets["fina_indicator"]),
                            latest_row(datasets["income"]),
                            latest_row(datasets["balancesheet"]),
                            datasets["fina_mainbz"],
                            resolved["ts_code"],
                        )
                    )

                    for _, basic in peer_candidates_df.iterrows():
                        basic_row = {k: (None if pd.isna(v) else v) for k, v in basic.items()}
                        peer_ts_code = basic_row["ts_code"]
                        indicator_df = sort_desc(safe_call(f"peer_fina_indicator_{peer_ts_code}", pro.fina_indicator, ts_code=peer_ts_code)).head(FINANCIAL_LIMIT)
                        income_df = sort_desc(safe_call(f"peer_income_{peer_ts_code}", pro.income, ts_code=peer_ts_code)).head(FINANCIAL_LIMIT)
                        balance_df = sort_desc(safe_call(f"peer_balancesheet_{peer_ts_code}", pro.balancesheet, ts_code=peer_ts_code)).head(FINANCIAL_LIMIT)
                        mainbz_df = sort_desc(safe_call(f"peer_fina_mainbz_{peer_ts_code}", pro.fina_mainbz, ts_code=peer_ts_code)).head(FINANCIAL_LIMIT * 8)
                        rows.append(
                            build_company_metric_snapshot(
                                basic_row,
                                market_lookup.get(peer_ts_code, basic_row),
                                latest_row(indicator_df),
                                latest_row(income_df),
                                latest_row(balance_df),
                                mainbz_df,
                                resolved["ts_code"],
                            )
                        )

                    peer_df = pd.DataFrame(rows)
                    if peer_df.empty:
                        return peer_df
                    if "total_mv" in peer_df.columns:
                        peer_df["_sort_mv"] = pd.to_numeric(peer_df["total_mv"], errors="coerce")
                    else:
                        peer_df["_sort_mv"] = np.nan
                    peer_df["_sort_target"] = peer_df["is_target"].astype(int)
                    peer_df = peer_df.sort_values(["_sort_target", "_sort_mv"], ascending=[False, False]).drop(columns=["_sort_target", "_sort_mv"])
                    return peer_df.reset_index(drop=True)


                def peer_metric_context(peer_df: pd.DataFrame, target_ts_code: str, metrics: list[str]) -> dict:
                    if peer_df.empty or "ts_code" not in peer_df.columns:
                        return {}
                    target_row = peer_df[peer_df["ts_code"] == target_ts_code]
                    if target_row.empty:
                        return {}
                    result: dict[str, dict] = {}
                    lower_is_better = {"pe_ttm", "pb", "ps_ttm", "peg_ttm"}
                    for metric in metrics:
                        if metric not in peer_df.columns:
                            continue
                        series = pd.to_numeric(peer_df[metric], errors="coerce")
                        series_valid = series.dropna()
                        target_value = pd.to_numeric(target_row.iloc[0][metric], errors="coerce")
                        if series_valid.empty or pd.isna(target_value):
                            continue
                        peer_median = series_valid.median()
                        delta_pct = None
                        if peer_median not in (None, 0) and not pd.isna(peer_median):
                            delta_pct = (float(target_value) / float(peer_median) - 1.0) * 100
                        if metric in lower_is_better:
                            rank = int((series < target_value).sum() + 1)
                        else:
                            rank = int((series > target_value).sum() + 1)
                        result[metric] = {
                            "target": to_number(target_value),
                            "peer_median": to_number(peer_median),
                            "delta_vs_peer_median_pct": to_number(delta_pct),
                            "rank": rank,
                            "peer_count": int(series_valid.shape[0]),
                        }
                    return result
                '''
            ),
            code_cell(
                """
                ensure_token(TOKEN)
                pro = ts.pro_api(TOKEN)

                stock_basic_all = safe_call(
                    "stock_basic_all",
                    pro.stock_basic,
                    exchange="",
                    list_status="L",
                    fields="ts_code,symbol,name,area,industry,market,list_date",
                )
                if stock_basic_all.empty:
                    raise ValueError("stock_basic 接口未返回数据，无法继续。")
                stock_basic_all = stock_basic_all.fillna("")

                resolved = resolve_ts_code(stock_basic_all, STOCK_INPUT)
                ts_code = resolved["ts_code"]
                stock_name = resolved["name"]
                exchange = exchange_from_ts_code(ts_code)

                company_df = safe_call("stock_company", pro.stock_company, exchange=exchange)
                if not company_df.empty and "ts_code" in company_df.columns:
                    company_df = company_df[company_df["ts_code"] == ts_code].reset_index(drop=True)

                datasets = {
                    "stock_basic": pd.DataFrame([resolved]),
                    "stock_company": company_df,
                    "daily": safe_call("daily", pro.daily, ts_code=ts_code, start_date=START_DATE, end_date=END_DATE),
                    "daily_basic": safe_call("daily_basic", pro.daily_basic, ts_code=ts_code, start_date=START_DATE, end_date=END_DATE),
                    "adj_factor": safe_call("adj_factor", pro.adj_factor, ts_code=ts_code, start_date=START_DATE, end_date=END_DATE),
                    "moneyflow": safe_call("moneyflow", pro.moneyflow, ts_code=ts_code, start_date=START_DATE, end_date=END_DATE),
                    "hk_hold": safe_call("hk_hold", pro.hk_hold, ts_code=ts_code, start_date=START_DATE, end_date=END_DATE),
                    "stk_holdertrade": safe_call("stk_holdertrade", pro.stk_holdertrade, ts_code=ts_code, start_date=START_DATE, end_date=END_DATE),
                    "income": safe_call("income", pro.income, ts_code=ts_code),
                    "balancesheet": safe_call("balancesheet", pro.balancesheet, ts_code=ts_code),
                    "cashflow": safe_call("cashflow", pro.cashflow, ts_code=ts_code),
                    "fina_indicator": safe_call("fina_indicator", pro.fina_indicator, ts_code=ts_code),
                    "fina_mainbz": safe_call("fina_mainbz", pro.fina_mainbz, ts_code=ts_code),
                    "forecast": safe_call("forecast", pro.forecast, ts_code=ts_code),
                    "express": safe_call("express", pro.express, ts_code=ts_code),
                    "dividend": safe_call("dividend", pro.dividend, ts_code=ts_code),
                    "disclosure_date": safe_call("disclosure_date", pro.disclosure_date, ts_code=ts_code),
                    "top10_holders": safe_call("top10_holders", pro.top10_holders, ts_code=ts_code),
                    "top10_floatholders": safe_call("top10_floatholders", pro.top10_floatholders, ts_code=ts_code),
                }

                datasets["daily"] = sort_desc(datasets["daily"]).head(KLINE_LIMIT)
                datasets["daily_basic"] = sort_desc(datasets["daily_basic"]).head(KLINE_LIMIT)
                datasets["adj_factor"] = sort_desc(datasets["adj_factor"]).head(KLINE_LIMIT)
                datasets["moneyflow"] = sort_desc(datasets["moneyflow"]).head(KLINE_LIMIT)
                datasets["hk_hold"] = sort_desc(datasets["hk_hold"]).head(HK_HOLD_LIMIT)
                datasets["stk_holdertrade"] = sort_desc(datasets["stk_holdertrade"]).head(HOLDER_TRADE_LIMIT)
                for name in ["income", "balancesheet", "cashflow", "fina_indicator", "forecast", "express", "dividend", "disclosure_date"]:
                    datasets[name] = sort_desc(datasets[name]).head(FINANCIAL_LIMIT)
                datasets["fina_mainbz"] = sort_desc(datasets["fina_mainbz"]).head(FINANCIAL_LIMIT * 8)
                datasets["top10_holders"] = sort_desc(datasets["top10_holders"]).head(HOLDER_HISTORY_LIMIT)
                datasets["top10_floatholders"] = sort_desc(datasets["top10_floatholders"]).head(HOLDER_HISTORY_LIMIT)

                latest_trade_date = pick_value(latest_row(datasets["daily"]), "trade_date") or END_DATE
                market_daily_basic = safe_call(
                    "daily_basic_market",
                    pro.daily_basic,
                    trade_date=latest_trade_date,
                    fields="ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_mv,circ_mv",
                )

                fund_portfolio_df, fund_portfolio_periods = fetch_fund_portfolio_history(
                    pro,
                    ts_code,
                    END_DATE,
                    FUND_PORTFOLIO_QUARTERS,
                )
                datasets["fund_portfolio"] = fund_portfolio_df

                peer_candidates_df, peer_selection_method = build_peer_candidates(
                    stock_basic_all,
                    resolved,
                    market_daily_basic,
                    PEER_TS_CODES,
                    PEER_LIMIT,
                )
                peer_comparison_df = build_peer_comparison(
                    pro,
                    resolved,
                    peer_candidates_df,
                    market_daily_basic,
                    datasets,
                )
                datasets["peer_comparison"] = peer_comparison_df

                row_counts = {name: int(len(df)) for name, df in datasets.items()}
                print(
                    {
                        "ts_code": ts_code,
                        "stock_name": stock_name,
                        "peer_selection_method": peer_selection_method,
                        "peer_count": max(int(len(peer_comparison_df)) - 1, 0),
                        "row_counts": row_counts,
                    }
                )
                """
            ),
            code_cell(
                """
                latest_quote = latest_row(datasets["daily"])
                latest_basic = latest_row(datasets["daily_basic"])
                latest_company = latest_row(datasets["stock_company"])
                latest_indicator = latest_row(datasets["fina_indicator"])
                latest_income = latest_row(datasets["income"])
                latest_balance = latest_row(datasets["balancesheet"])
                latest_cashflow = latest_row(datasets["cashflow"])
                latest_forecast = latest_row(datasets["forecast"])
                latest_express = latest_row(datasets["express"])
                latest_dividend = latest_row(datasets["dividend"])
                latest_disclosure = latest_row(datasets["disclosure_date"])

                daily_df = sort_desc(datasets["daily"])
                daily_basic_df = sort_desc(datasets["daily_basic"])
                moneyflow_df = sort_desc(datasets["moneyflow"])
                main_net_series = compute_main_net_amount(moneyflow_df)

                holder_df = sort_desc(datasets["top10_holders"])
                float_holder_df = sort_desc(datasets["top10_floatholders"])
                holder_latest = latest_period_rows(holder_df, "end_date", 10)
                float_holder_latest = latest_period_rows(float_holder_df, "end_date", 10)
                holder_latest_end_date = holder_latest["end_date"].iloc[0] if not holder_latest.empty else None
                float_holder_latest_end_date = float_holder_latest["end_date"].iloc[0] if not float_holder_latest.empty else None

                northbound_summary = summarize_hk_hold(datasets["hk_hold"])
                major_holder_trade_summary = summarize_holder_trade(datasets["stk_holdertrade"], recent_days=365)
                public_fund_top10_summary = summarize_holder_category(holder_df, ["基金", "社保", "养老金", "资管"], "public_fund_top10")
                insurance_top10_summary = summarize_holder_category(holder_df, ["保险", "寿险", "养老"], "insurance_top10")
                fund_count_trend = summarize_periodic_trend(fund_portfolio_periods, "funds_count")
                fund_amount_trend = summarize_periodic_trend(fund_portfolio_periods, "hold_amount_total")
                fund_mkv_trend = summarize_periodic_trend(fund_portfolio_periods, "hold_mkv_total")

                peer_metric_summary = peer_metric_context(
                    peer_comparison_df,
                    ts_code,
                    [
                        "pe_ttm",
                        "pb",
                        "ps_ttm",
                        "peg_ttm",
                        "roe",
                        "gross_margin",
                        "netprofit_margin",
                        "revenue_yoy",
                        "netprofit_yoy",
                        "money_cap",
                        "overseas_revenue_ratio",
                    ],
                )

                annual_target_period = "20251231"
                annual_2025_disclosure = latest_row(
                    sort_desc(datasets["disclosure_date"][datasets["disclosure_date"]["end_date"] == annual_target_period])
                ) if not datasets["disclosure_date"].empty and "end_date" in datasets["disclosure_date"].columns else {}
                annual_2025_express = latest_row(
                    sort_desc(datasets["express"][datasets["express"]["end_date"] == annual_target_period])
                ) if not datasets["express"].empty and "end_date" in datasets["express"].columns else {}
                annual_2025_dividend = latest_row(
                    sort_desc(datasets["dividend"][datasets["dividend"]["end_date"] == annual_target_period])
                ) if not datasets["dividend"].empty and "end_date" in datasets["dividend"].columns else {}

                analysis_summary = {
                    "meta": {
                        "generated_at": pd.Timestamp.now().isoformat(),
                        "stock_input": STOCK_INPUT,
                        "ts_code": ts_code,
                        "name": stock_name,
                        "date_window": {"start_date": START_DATE, "end_date": END_DATE},
                        "peer_selection_method": peer_selection_method,
                        "manual_peer_ts_codes": PEER_TS_CODES,
                    },
                    "company_profile": {
                        "area": resolved.get("area"),
                        "industry": resolved.get("industry"),
                        "market": resolved.get("market"),
                        "list_date": resolved.get("list_date"),
                        "exchange": exchange,
                        "chairman": pick_value(latest_company, "chairman"),
                        "manager": pick_value(latest_company, "manager"),
                        "employees": pick_value(latest_company, "employees"),
                        "province": pick_value(latest_company, "province"),
                        "city": pick_value(latest_company, "city"),
                        "website": pick_value(latest_company, "website"),
                        "business_scope": pick_value(latest_company, "business_scope", "main_business"),
                        "main_business_tags": extract_mainbz_tags(datasets["fina_mainbz"], 5),
                    },
                    "market_snapshot": {
                        "trade_date": pick_value(latest_quote, "trade_date"),
                        "close": to_number(pick_value(latest_quote, "close")),
                        "pct_chg": to_number(pick_value(latest_quote, "pct_chg")),
                        "open": to_number(pick_value(latest_quote, "open")),
                        "high": to_number(pick_value(latest_quote, "high")),
                        "low": to_number(pick_value(latest_quote, "low")),
                        "vol": to_number(pick_value(latest_quote, "vol")),
                        "amount": to_number(pick_value(latest_quote, "amount")),
                        "turnover_rate": to_number(pick_value(latest_basic, "turnover_rate", "turnover_rate_f")),
                        "volume_ratio": to_number(pick_value(latest_basic, "volume_ratio")),
                        "pe_ttm": to_number(pick_value(latest_basic, "pe_ttm", "pe")),
                        "pb": to_number(pick_value(latest_basic, "pb")),
                        "ps_ttm": to_number(pick_value(latest_basic, "ps_ttm", "ps")),
                        "dv_ttm": to_number(pick_value(latest_basic, "dv_ttm", "dv_ratio")),
                        "total_mv": to_number(pick_value(latest_basic, "total_mv")),
                        "circ_mv": to_number(pick_value(latest_basic, "circ_mv")),
                    },
                    "price_volume_trend": {
                        "return_5d_pct": to_number(compute_return(daily_df, 5)),
                        "return_20d_pct": to_number(compute_return(daily_df, 20)),
                        "return_60d_pct": to_number(compute_return(daily_df, 60)),
                        "avg_vol_5d": to_number(average_column(daily_df, "vol", 5)),
                        "avg_vol_20d": to_number(average_column(daily_df, "vol", 20)),
                        "avg_amount_5d": to_number(average_column(daily_df, "amount", 5)),
                        "avg_amount_20d": to_number(average_column(daily_df, "amount", 20)),
                        "avg_turnover_5d": to_number(average_column(daily_basic_df, "turnover_rate", 5)),
                        "avg_turnover_20d": to_number(average_column(daily_basic_df, "turnover_rate", 20)),
                    },
                    "fund_flow": {
                        "latest_trade_date": moneyflow_df["trade_date"].iloc[0] if not moneyflow_df.empty and "trade_date" in moneyflow_df.columns else None,
                        "main_net_amount_5d": to_number(main_net_series.head(5).sum() if not main_net_series.empty else None),
                        "main_net_amount_20d": to_number(main_net_series.head(20).sum() if not main_net_series.empty else None),
                        "main_net_amount_latest": to_number(main_net_series.iloc[0] if not main_net_series.empty else None),
                    },
                    "shareholder_signals": {
                        "northbound": northbound_summary,
                        "public_fund_portfolio": {
                            "funds_count_trend": fund_count_trend,
                            "hold_amount_trend": fund_amount_trend,
                            "hold_mkv_trend": fund_mkv_trend,
                            "recent_records": frame_tail(datasets["fund_portfolio"], 8),
                        },
                        "public_fund_top10": public_fund_top10_summary,
                        "insurance_top10": insurance_top10_summary,
                        "major_holder_trade": major_holder_trade_summary,
                        "top10_report_period": holder_latest_end_date,
                        "top10_total_holding_ratio": to_number(
                            pd.to_numeric(holder_latest.get("hold_ratio"), errors="coerce").sum()
                            if not holder_latest.empty and "hold_ratio" in holder_latest.columns
                            else None
                        ),
                        "top10_names": holder_latest["holder_name"].head(10).tolist() if not holder_latest.empty and "holder_name" in holder_latest.columns else [],
                        "top10_float_report_period": float_holder_latest_end_date,
                        "top10_float_holders": float_holder_latest["holder_name"].head(10).tolist()
                        if not float_holder_latest.empty and "holder_name" in float_holder_latest.columns
                        else [],
                    },
                    "financials": {
                        "report_period": pick_value(latest_indicator, "end_date") or pick_value(latest_income, "end_date"),
                        "roe": to_number(pick_value(latest_indicator, "roe", "q_roe")),
                        "gross_margin": to_number(pick_value(latest_indicator, "gross_margin", "grossprofit_margin")),
                        "debt_to_assets": to_number(pick_value(latest_indicator, "debt_to_assets")),
                        "ocf_to_or": to_number(pick_value(latest_indicator, "ocf_to_or", "q_ocf_to_sales")),
                        "netprofit_yoy": to_number(pick_value(latest_indicator, "netprofit_yoy", "dt_netprofit_yoy")),
                        "revenue_yoy": to_number(pick_value(latest_indicator, "tr_yoy", "or_yoy", "q_sales_yoy")),
                        "revenue": to_number(pick_value(latest_income, "total_revenue", "revenue")),
                        "net_profit": to_number(pick_value(latest_income, "n_income_attr_p", "n_income")),
                        "operate_cashflow": to_number(pick_value(latest_cashflow, "n_cashflow_act")),
                        "total_assets": to_number(pick_value(latest_balance, "total_assets")),
                        "total_liab": to_number(pick_value(latest_balance, "total_liab")),
                        "money_cap": to_number(pick_value(latest_balance, "money_cap")),
                        "overseas_revenue_ratio": to_number(extract_overseas_revenue_ratio(datasets["fina_mainbz"])),
                    },
                    "peer_comparison": {
                        "selection_method": peer_selection_method,
                        "industry": resolved.get("industry"),
                        "selected_ts_codes": peer_comparison_df["ts_code"].tolist() if not peer_comparison_df.empty else [],
                        "metric_vs_peer_median": peer_metric_summary,
                        "table_preview": frame_tail(peer_comparison_df, min(len(peer_comparison_df), PEER_LIMIT + 1)),
                    },
                    "performance_events": {
                        "latest_forecast": latest_forecast,
                        "latest_express": latest_express,
                        "latest_dividend": latest_dividend,
                        "latest_disclosure": latest_disclosure,
                    },
                    "annual_report_tracking": {
                        "focus_period": annual_target_period,
                        "disclosure_record": annual_2025_disclosure,
                        "express_record": annual_2025_express,
                        "dividend_record": annual_2025_dividend,
                        "text_items_still_needed": [
                            "年报正文中的管理层经营回顾和 2026 年指引",
                            "新品周期或新订单节奏",
                            "商誉、减值、投资收益等非经常性扰动解释",
                        ],
                    },
                    "dataset_previews": {
                        "daily": frame_tail(datasets["daily"], 5),
                        "daily_basic": frame_tail(datasets["daily_basic"], 5),
                        "moneyflow": frame_tail(datasets["moneyflow"], 5),
                        "hk_hold": frame_tail(datasets["hk_hold"], 5),
                        "stk_holdertrade": frame_tail(datasets["stk_holdertrade"], 5),
                        "fund_portfolio": frame_tail(datasets["fund_portfolio"], 5),
                        "peer_comparison": frame_tail(peer_comparison_df, min(len(peer_comparison_df), 5)),
                    },
                    "analysis_limitations": [
                        "北向资金仅适用于沪深港通标的，非互联互通股票会没有 hk_hold 数据。",
                        "保险加仓趋势目前主要基于前十大股东名单关键词识别，不保证覆盖全部保险资金。",
                        "海外收入占比依赖公司在主营构成表中的披露口径。",
                        "新品周期、管理层指引、减值原因等定性信息通常需要公告正文或投资者关系纪要。",
                    ],
                    "table_row_counts": row_counts,
                }

                peer_comparison_df
                """
            ),
            code_cell(
                '''
                export_dir = OUTPUT_DIR / ts_code.replace(".", "_")
                export_dir.mkdir(parents=True, exist_ok=True)

                for name, df in datasets.items():
                    if not df.empty:
                        df.to_csv(export_dir / f"{name}.csv", index=False)

                analysis_payload = json_safe(analysis_summary)

                payload_path = export_dir / "gpt_analysis_payload.json"
                with payload_path.open("w", encoding="utf-8") as f:
                    json.dump(analysis_payload, f, ensure_ascii=False, indent=2)

                prompt_path = export_dir / "gpt_prompt.txt"
                prompt_text = f"""请基于以下 A 股数据对 {stock_name}（{ts_code}）做系统分析：

                1. 先给出一句话结论。
                2. 分别分析基本面、成长性、盈利能力、偿债能力、现金流、估值、量价趋势、资金流、北向资金、公募持仓、股东增减持、同业对比。
                3. 判断当前估值相对同业是偏贵、合理还是偏便宜，并说明理由。
                4. 指出最强的 3 个支撑因素和最弱的 3 个风险点。
                5. 如果数据之间有冲突，明确说明冲突点。
                6. 最后列出还需要公告正文或管理层交流纪要补充验证的点。

                结构化数据如下：
                {json.dumps(analysis_payload, ensure_ascii=False, indent=2)}
                """
                prompt_path.write_text(prompt_text, encoding="utf-8")

                print(
                    {
                        "export_dir": str(export_dir.resolve()),
                        "payload_path": str(payload_path.resolve()),
                        "prompt_path": str(prompt_path.resolve()),
                    }
                )
                pd.DataFrame([{"table": name, "rows": len(df)} for name, df in datasets.items()]).sort_values("rows", ascending=False)
                '''
            ),
            markdown_cell(
                """
                ## 交给 GPT 的方式

                - 直接把 `gpt_prompt.txt` 的内容发给 GPT。
                - 或者把 `gpt_analysis_payload.json` 作为结构化数据输入，再单独补充你的分析要求。
                - 如果想要公告正文、管理层指引、新品周期等定性信息，还需要补公告文本源；当前 notebook 主要覆盖结构化 API 数据。
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9.6",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "output" / "jupyter-notebook" / "tushare-stock-full-analysis.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
