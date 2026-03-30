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
                # Experiment: 股东增持筛选股票

                Objective:
                - 从最近股东增持事件出发，筛选“业绩没有明显恶化 + 股价仍在相对低位 + 量价开始止跌转强”的 A 股候选。
                - 输出候选股票打分表、分项解释和可复核的原始导出文件，方便后续继续人工复盘。
                """
            ),
            markdown_cell(
                """
                ## 使用方式

                1. 只需要修改下面配置单元格，然后 `Run -> Run All Cells`。
                2. 先用 `daily_basic`、`moneyflow`、`stk_factor_pro` 对全市场做横截面快筛，主框架优先看 `资金 + 趋势 + 业绩安全`。
                3. `stk_holdertrade`、`share_float` 只做辅助修正，用来识别增减持、减持计划、解禁风险，不再作为主选股框架。
                4. 第一阶段先对全市场快筛后的前 80 名补抓 `fina_indicator`、`forecast`，生成不含筹码的前 10 名。
                5. 第二阶段只对这 10 只补抓 `cyq_perf`，再按筹码、过热惩罚、减持/解禁风控重排，同时输出 `稳健池` 和 `进攻池`。
                6. 如果你使用不限额 token，可以启用自定义 `http_url`；如果退回官方接口，也可以把第二阶段预算调低。
                7. 结果会写到 `output/jupyter-notebook/tushare_screen_exports/holder_increase_screen_<date>/`。
                """
            ),
            code_cell(
                """
                from __future__ import annotations

                import json
                import os
                import time
                import warnings
                from urllib.parse import urlparse
                from typing import Optional
                from pathlib import Path

                warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
                try:
                    from urllib3.exceptions import NotOpenSSLWarning
                    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
                except Exception:
                    pass

                import numpy as np
                import pandas as pd
                import tushare as ts

                warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

                pd.set_option("display.max_columns", 120)
                pd.set_option("display.width", 240)
                pd.set_option("display.max_colwidth", 120)
                API_ERROR_LOG: dict[str, str] = {}

                CWD = Path.cwd().resolve()
                if CWD.name == "jupyter-notebook" and CWD.parent.name == "output":
                    REPO_ROOT = CWD.parent.parent
                else:
                    REPO_ROOT = CWD

                TODAY = pd.Timestamp.today().normalize()
                TODAY_STR = TODAY.strftime("%Y%m%d")
                DEFAULT_ANN_START = (TODAY - pd.Timedelta(days=45)).strftime("%Y%m%d")
                DEFAULT_PRICE_START = (TODAY - pd.Timedelta(days=420)).strftime("%Y%m%d")
                OUTPUT_DIR = REPO_ROOT / "output" / "jupyter-notebook" / "tushare_screen_exports"
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                print(
                    {
                        "today": TODAY_STR,
                        "default_ann_start": DEFAULT_ANN_START,
                        "default_price_start": DEFAULT_PRICE_START,
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
                USE_CUSTOM_HTTP_ENDPOINT = True
                CUSTOM_HTTP_URL = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()

                ANN_START_DATE = DEFAULT_ANN_START
                END_DATE = TODAY_STR
                EVENT_CHUNK_DAYS = 5
                PRICE_LOOKBACK_DAYS = 250
                MONEYFLOW_LOOKBACK_DAYS = 5
                CYQ_LOOKBACK_DAYS = 20
                RECENT_SIGNAL_LOOKBACK_DAYS = 20
                MARKET_DATA_CUTOFF_HOUR = 20
                UNLOCK_LOOKAHEAD_DAYS = 30
                MAX_NEAR_UNLOCK_RATIO = 3.0
                MAX_UNLOCK_RATIO_30D = 8.0
                ACTIVE_REDUCTION_MIN_RATIO = 0.3
                API_SLEEP_SEC = 0.12
                CYQ_SLEEP_SEC = 0.12

                ALLOWED_HOLDER_TYPES = ["C", "G"]  # C=公司股东，G=高管；想放宽可加 P 等
                MIN_EVENT_COUNT = 1
                MIN_TOTAL_CHANGE_RATIO = 0.0
                MIN_VOLUME_RATIO = 1.2
                MAX_PRICE_POSITION = 0.45
                MAX_INDUSTRY_PB_PCT = 0.70
                MAX_DEEP_DIVE_STOCKS = 80
                TOP_N_STAGE1 = 10
                MAX_STAGE2_CYQ_STOCKS = 10
                STAGE2_CYQ_BUDGET = 10
                TOP_N_FINAL = 5
                TOP_N_AGGRESSIVE = 3
                MIN_FINAL_SCORE = 60
                MIN_AGGRESSIVE_SCORE = 52
                ENABLE_FORECAST = True
                ENABLE_STAGE2_CYQ = True

                if not TOKEN.strip():
                    TOKEN = input("输入 Tushare Token: ").strip()

                config_snapshot = {
                    "ann_start_date": ANN_START_DATE,
                    "end_date": END_DATE,
                    "use_custom_http_endpoint": USE_CUSTOM_HTTP_ENDPOINT,
                    "custom_http_url": CUSTOM_HTTP_URL if USE_CUSTOM_HTTP_ENDPOINT else "",
                    "event_chunk_days": EVENT_CHUNK_DAYS,
                    "price_lookback_days": PRICE_LOOKBACK_DAYS,
                    "moneyflow_lookback_days": MONEYFLOW_LOOKBACK_DAYS,
                    "cyq_lookback_days": CYQ_LOOKBACK_DAYS,
                    "recent_signal_lookback_days": RECENT_SIGNAL_LOOKBACK_DAYS,
                    "market_data_cutoff_hour": MARKET_DATA_CUTOFF_HOUR,
                    "unlock_lookahead_days": UNLOCK_LOOKAHEAD_DAYS,
                    "max_near_unlock_ratio": MAX_NEAR_UNLOCK_RATIO,
                    "max_unlock_ratio_30d": MAX_UNLOCK_RATIO_30D,
                    "active_reduction_min_ratio": ACTIVE_REDUCTION_MIN_RATIO,
                    "api_sleep_sec": API_SLEEP_SEC,
                    "cyq_sleep_sec": CYQ_SLEEP_SEC,
                    "allowed_holder_types": ALLOWED_HOLDER_TYPES,
                    "min_event_count": MIN_EVENT_COUNT,
                    "min_total_change_ratio": MIN_TOTAL_CHANGE_RATIO,
                    "min_volume_ratio": MIN_VOLUME_RATIO,
                    "max_price_position": MAX_PRICE_POSITION,
                    "max_industry_pb_pct": MAX_INDUSTRY_PB_PCT,
                    "max_deep_dive_stocks": MAX_DEEP_DIVE_STOCKS,
                    "top_n_stage1": TOP_N_STAGE1,
                    "max_stage2_cyq_stocks": MAX_STAGE2_CYQ_STOCKS,
                    "stage2_cyq_budget": STAGE2_CYQ_BUDGET,
                    "top_n_final": TOP_N_FINAL,
                    "top_n_aggressive": TOP_N_AGGRESSIVE,
                    "min_final_score": MIN_FINAL_SCORE,
                    "min_aggressive_score": MIN_AGGRESSIVE_SCORE,
                    "enable_forecast": ENABLE_FORECAST,
                    "enable_stage2_cyq": ENABLE_STAGE2_CYQ,
                }
                config_snapshot
                """
            ),
            code_cell(
                """
                def ensure_token(token: str) -> None:
                    if not token or token.startswith("PASTE_"):
                        raise ValueError("缺少 Tushare Token。请设置环境变量 TUSHARE_TOKEN 或直接在配置单元格里填写。")


                def configure_proxy_bypass(custom_http_url: str) -> None:
                    if not custom_http_url:
                        return
                    parsed = urlparse(custom_http_url)
                    host = (parsed.hostname or "").strip()
                    no_proxy_tokens: list[str] = []
                    for value in [os.getenv("NO_PROXY", ""), os.getenv("no_proxy", ""), "127.0.0.1", "localhost", host]:
                        for token in str(value).split(","):
                            token = token.strip()
                            if token and token not in no_proxy_tokens:
                                no_proxy_tokens.append(token)
                    if no_proxy_tokens:
                        merged = ",".join(no_proxy_tokens)
                        os.environ["NO_PROXY"] = merged
                        os.environ["no_proxy"] = merged
                    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
                        os.environ.pop(key, None)


                def configure_tushare_client(pro, token: str, use_custom_http_endpoint: bool, custom_http_url: str):
                    if use_custom_http_endpoint and custom_http_url:
                        configure_proxy_bypass(custom_http_url)
                        pro._DataApi__token = token
                        pro._DataApi__http_url = custom_http_url
                    return pro


                def safe_call(label: str, fn, sleep_sec: float = 0.0, **kwargs) -> pd.DataFrame:
                    if fn is None:
                        print(f"[{label}] 接口不可用")
                        return pd.DataFrame()
                    try:
                        df = fn(**kwargs)
                    except Exception as exc:
                        API_ERROR_LOG[label] = str(exc)
                        print(f"[{label}] 调用失败: {exc}")
                        return pd.DataFrame()
                    API_ERROR_LOG.pop(label, None)
                    if sleep_sec:
                        time.sleep(sleep_sec)
                    if df is None:
                        return pd.DataFrame()
                    return df.copy()


                def sort_desc(df: pd.DataFrame) -> pd.DataFrame:
                    if df.empty:
                        return df
                    for col in ["trade_date", "ann_date", "end_date", "f_ann_date"]:
                        if col in df.columns:
                            return df.sort_values(col, ascending=False).reset_index(drop=True)
                    return df.reset_index(drop=True)


                def latest_row(df: pd.DataFrame) -> dict:
                    ordered = sort_desc(df)
                    if ordered.empty:
                        return {}
                    row = ordered.iloc[0]
                    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


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


                def to_float(value):
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


                def rank_pct(series: pd.Series) -> pd.Series:
                    numeric = pd.to_numeric(series, errors="coerce")
                    ranked = numeric.rank(pct=True, method="average")
                    return ranked.fillna(0.5)


                def ensure_columns(df: pd.DataFrame, columns: list[str], fill_value=np.nan) -> pd.DataFrame:
                    work = df.copy()
                    for column in columns:
                        if column not in work.columns:
                            work[column] = fill_value
                    return work


                def clip_score(value, low: float, high: float) -> float:
                    if value is None:
                        return low
                    return float(min(max(value, low), high))


                def positive_flag(value, threshold: float = 0.0) -> bool:
                    number = to_float(value)
                    return number is not None and number > threshold


                def to_bool(value) -> bool:
                    if value is None:
                        return False
                    try:
                        if pd.isna(value):
                            return False
                    except Exception:
                        pass
                    if isinstance(value, str):
                        return value.strip().lower() in {"1", "true", "yes", "y"}
                    return bool(value)


                def is_rate_limit_error(message: Optional[str]) -> bool:
                    if not message:
                        return False
                    return any(token in str(message) for token in ["每分钟最多访问", "每小时最多访问", "每天最多访问"])


                def chunk_date_ranges(start_date: str, end_date: str, chunk_days: int = 5) -> list[tuple[str, str]]:
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date)
                    ranges: list[tuple[str, str]] = []
                    cursor = start_ts
                    while cursor <= end_ts:
                        right = min(cursor + pd.Timedelta(days=chunk_days - 1), end_ts)
                        ranges.append((cursor.strftime("%Y%m%d"), right.strftime("%Y%m%d")))
                        cursor = right + pd.Timedelta(days=1)
                    return ranges


                def get_recent_open_trade_dates(pro, end_date: str, count: int = 8) -> list[str]:
                    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(20, count * 4))).strftime("%Y%m%d")
                    cal = safe_call(
                        "trade_cal",
                        getattr(pro, "trade_cal", None),
                        start_date=start_date,
                        end_date=end_date,
                        is_open="1",
                    )
                    if cal.empty:
                        return [end_date]
                    date_col = "cal_date" if "cal_date" in cal.columns else "trade_date"
                    dates = sorted(cal[date_col].dropna().astype(str).unique().tolist())
                    return dates[-count:] if dates else [end_date]


                def choose_screen_end_date(now_ts: pd.Timestamp, end_date: str, today_str: str, cutoff_hour: int = 20) -> str:
                    if end_date == today_str and int(now_ts.hour) < int(cutoff_hour):
                        return (now_ts.normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
                    return end_date


                def fetch_latest_complete_market_inputs(
                    pro,
                    trade_dates: list[str],
                    moneyflow_lookback_days: int,
                    sleep_sec: float = 0.0,
                ) -> tuple[str, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                    if not trade_dates:
                        return TODAY_STR, [TODAY_STR], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

                    fallback_trade_date = trade_dates[-1]
                    fallback_moneyflow_dates = trade_dates[-moneyflow_lookback_days:]
                    fallback_daily_basic = pd.DataFrame()
                    fallback_tech = pd.DataFrame()
                    fallback_moneyflow = pd.DataFrame()

                    for idx in range(len(trade_dates) - 1, -1, -1):
                        trade_date = trade_dates[idx]
                        moneyflow_dates = trade_dates[max(0, idx - moneyflow_lookback_days + 1) : idx + 1]
                        daily_basic_df = safe_call(
                            f"daily_basic_{trade_date}",
                            getattr(pro, "daily_basic", None),
                            trade_date=trade_date,
                        )
                        tech_df = safe_call(
                            f"stk_factor_pro_{trade_date}",
                            getattr(pro, "stk_factor_pro", None),
                            trade_date=trade_date,
                        )
                        moneyflow_df = fetch_recent_moneyflow_summary(
                            pro,
                            moneyflow_dates,
                            sleep_sec=sleep_sec,
                        )

                        fallback_trade_date = trade_date
                        fallback_moneyflow_dates = moneyflow_dates
                        fallback_daily_basic = daily_basic_df
                        fallback_tech = tech_df
                        fallback_moneyflow = moneyflow_df

                        if not daily_basic_df.empty and not tech_df.empty and not moneyflow_df.empty:
                            return trade_date, moneyflow_dates, daily_basic_df, tech_df, moneyflow_df

                    return (
                        fallback_trade_date,
                        fallback_moneyflow_dates,
                        fallback_daily_basic,
                        fallback_tech,
                        fallback_moneyflow,
                    )


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


                def fetch_holdertrade_events(
                    pro,
                    start_date: str,
                    end_date: str,
                    chunk_days: int = 5,
                    sleep_sec: float = 0.0,
                ) -> pd.DataFrame:
                    frames: list[pd.DataFrame] = []
                    for left, right in chunk_date_ranges(start_date, end_date, chunk_days):
                        df = safe_call(
                            f"stk_holdertrade_{left}_{right}",
                            getattr(pro, "stk_holdertrade", None),
                            sleep_sec=sleep_sec,
                            start_date=left,
                            end_date=right,
                        )
                        if not df.empty:
                            frames.append(df)
                            print(f"holdertrade {left}-{right}: {len(df)} rows")
                    if not frames:
                        return pd.DataFrame()
                    combined = pd.concat(frames, ignore_index=True)
                    combined = combined.drop_duplicates().reset_index(drop=True)
                    return sort_desc(combined)


                def fetch_share_float_schedule(
                    pro,
                    start_date: str,
                    end_date: str,
                    chunk_days: int = 10,
                    sleep_sec: float = 0.0,
                ) -> pd.DataFrame:
                    frames: list[pd.DataFrame] = []
                    for left, right in chunk_date_ranges(start_date, end_date, chunk_days):
                        df = safe_call(
                            f"share_float_{left}_{right}",
                            getattr(pro, "share_float", None),
                            sleep_sec=sleep_sec,
                            start_date=left,
                            end_date=right,
                        )
                        if not df.empty:
                            frames.append(df)
                    if not frames:
                        return pd.DataFrame()
                    combined = pd.concat(frames, ignore_index=True)
                    combined = combined.drop_duplicates().reset_index(drop=True)
                    return sort_desc(combined)


                def prepare_event_pool(
                    holdertrade_df: pd.DataFrame,
                    stock_basic_df: pd.DataFrame,
                    allowed_holder_types: set[str],
                    end_date: str,
                ) -> pd.DataFrame:
                    if holdertrade_df.empty:
                        return pd.DataFrame()

                    work = holdertrade_df.copy()
                    if "in_de" in work.columns:
                        work = work[work["in_de"].fillna("") == "IN"].copy()
                    if allowed_holder_types and "holder_type" in work.columns:
                        work = work[work["holder_type"].fillna("").astype(str).isin(sorted(allowed_holder_types))].copy()
                    if work.empty:
                        return pd.DataFrame()

                    for col in ["change_vol", "change_ratio", "avg_price", "after_ratio", "total_share"]:
                        if col in work.columns:
                            work[col] = pd.to_numeric(work[col], errors="coerce")
                    work["ann_date"] = work["ann_date"].astype(str)
                    work["ann_date_dt"] = pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce")
                    work = work.dropna(subset=["ann_date_dt"]).copy()

                    basic_cols = [c for c in ["ts_code", "symbol", "name", "industry", "market", "list_date"] if c in stock_basic_df.columns]
                    basics = stock_basic_df[basic_cols].drop_duplicates(subset=["ts_code"])
                    work = work.merge(basics, on="ts_code", how="left")
                    work["holder_type"] = work["holder_type"].fillna("").astype(str)
                    work["holder_name"] = work["holder_name"].fillna("").astype(str)

                    end_ts = pd.Timestamp(end_date)
                    grouped_rows: list[dict] = []
                    for ts_code, sub in work.groupby("ts_code", dropna=False):
                        ordered = sub.sort_values(["ann_date_dt", "change_ratio", "change_vol"], ascending=[False, False, False]).reset_index(drop=True)
                        latest = ordered.iloc[0]
                        holder_types = sorted({value for value in ordered["holder_type"].tolist() if value})
                        holder_preview = [value for value in ordered["holder_name"].dropna().astype(str).tolist() if value][:3]
                        latest_ann = latest["ann_date_dt"]
                        grouped_rows.append(
                            {
                                "ts_code": ts_code,
                                "name": latest.get("name"),
                                "industry": latest.get("industry"),
                                "market": latest.get("market"),
                                "latest_ann_date": latest_ann.strftime("%Y%m%d"),
                                "days_since_latest_ann": int((end_ts - latest_ann.normalize()).days),
                                "event_count": int(len(ordered)),
                                "announcement_days": int(ordered["ann_date"].nunique()),
                                "holder_count": int(ordered["holder_name"].nunique()),
                                "holder_type_tags": ",".join(holder_types),
                                "holder_preview": " / ".join(holder_preview),
                                "total_change_vol": pd.to_numeric(ordered["change_vol"], errors="coerce").sum(),
                                "total_change_ratio": pd.to_numeric(ordered["change_ratio"], errors="coerce").sum(),
                                "avg_increase_price": pd.to_numeric(ordered["avg_price"], errors="coerce").mean(),
                                "latest_after_ratio": to_number(latest.get("after_ratio")),
                            }
                        )

                    grouped = pd.DataFrame(grouped_rows)
                    if grouped.empty:
                        return grouped

                    grouped["total_change_vol"] = pd.to_numeric(grouped["total_change_vol"], errors="coerce")
                    grouped["total_change_ratio"] = pd.to_numeric(grouped["total_change_ratio"], errors="coerce")
                    grouped["avg_increase_price"] = pd.to_numeric(grouped["avg_increase_price"], errors="coerce")
                    grouped["change_ratio_rank_pct"] = rank_pct(grouped["total_change_ratio"])
                    grouped["change_vol_rank_pct"] = rank_pct(grouped["total_change_vol"])
                    grouped["activity_rank_pct"] = rank_pct(grouped["event_count"])
                    grouped["recentness_pct"] = (
                        1.0 - pd.to_numeric(grouped["days_since_latest_ann"], errors="coerce").fillna(999).clip(lower=0) / max(1, (pd.Timestamp(end_date) - pd.Timestamp(ANN_START_DATE)).days + 1)
                    ).clip(lower=0.0, upper=1.0)
                    grouped["holder_quality_bonus"] = 0.0
                    grouped.loc[grouped["holder_type_tags"].str.contains("C", na=False), "holder_quality_bonus"] += 0.70
                    grouped.loc[grouped["holder_type_tags"].str.contains("G", na=False), "holder_quality_bonus"] += 0.30
                    grouped["event_score"] = (
                        grouped["change_ratio_rank_pct"] * 10
                        + grouped["change_vol_rank_pct"] * 8
                        + grouped["activity_rank_pct"] * 4
                        + grouped["recentness_pct"] * 5
                        + grouped["holder_quality_bonus"] * 3
                    ).clip(lower=0.0, upper=30.0)
                    return grouped.sort_values(["event_score", "total_change_ratio", "event_count"], ascending=[False, False, False]).reset_index(drop=True)


                def build_reverse_signal_snapshot(
                    holdertrade_df: pd.DataFrame,
                    end_date: str,
                    lookback_days: int = 20,
                    core_holder_types: Optional[set[str]] = None,
                ) -> pd.DataFrame:
                    if holdertrade_df.empty:
                        return pd.DataFrame()

                    work = holdertrade_df.copy()
                    work["ann_date"] = work["ann_date"].astype(str)
                    work["ann_date_dt"] = pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce")
                    work = work.dropna(subset=["ann_date_dt"]).copy()
                    if "holder_type" in work.columns:
                        work["holder_type"] = work["holder_type"].fillna("").astype(str)
                    else:
                        work["holder_type"] = ""
                    if "holder_name" in work.columns:
                        work["holder_name"] = work["holder_name"].fillna("").astype(str)
                    else:
                        work["holder_name"] = ""
                    if "in_de" in work.columns:
                        work["in_de"] = work["in_de"].fillna("").astype(str).str.upper()
                    else:
                        work["in_de"] = ""
                    if "change_ratio" in work.columns:
                        work["change_ratio"] = pd.to_numeric(work["change_ratio"], errors="coerce").fillna(0.0)
                    else:
                        work["change_ratio"] = 0.0

                    cutoff = pd.Timestamp(end_date) - pd.Timedelta(days=max(1, lookback_days - 1))
                    core_holder_types = core_holder_types or set()
                    rows: list[dict] = []

                    for ts_code, sub in work.groupby("ts_code", dropna=False):
                        ordered = sub.sort_values(["ann_date_dt", "change_ratio"], ascending=[False, False]).reset_index(drop=True)
                        latest = ordered.iloc[0]
                        recent = ordered[ordered["ann_date_dt"] >= cutoff].copy()
                        recent_in = recent[recent["in_de"] == "IN"].copy()
                        recent_de = recent[recent["in_de"] == "DE"].copy()
                        recent_core_de = recent_de[recent_de["holder_type"].isin(sorted(core_holder_types))].copy() if core_holder_types else recent_de.copy()

                        recent_increase_ratio = pd.to_numeric(recent_in["change_ratio"], errors="coerce").sum()
                        recent_decrease_ratio = pd.to_numeric(recent_de["change_ratio"], errors="coerce").sum()
                        recent_core_decrease_ratio = pd.to_numeric(recent_core_de["change_ratio"], errors="coerce").sum()
                        signal_balance = recent_increase_ratio - recent_decrease_ratio

                        rows.append(
                            {
                                "ts_code": ts_code,
                                "latest_change_dir": latest.get("in_de"),
                                "latest_change_date": latest.get("ann_date"),
                                "latest_change_holder": latest.get("holder_name"),
                                "recent_increase_ratio": to_number(recent_increase_ratio, 4),
                                "recent_decrease_ratio": to_number(recent_decrease_ratio, 4),
                                "recent_core_decrease_ratio": to_number(recent_core_decrease_ratio, 4),
                                "recent_decrease_events": int(len(recent_de)),
                                "recent_signal_balance": to_number(signal_balance, 4),
                                "mixed_signal_flag": bool(recent_increase_ratio > 0 and recent_decrease_ratio > 0),
                            }
                        )

                    return pd.DataFrame(rows)


                def build_reduction_plan_snapshot(
                    holdertrade_df: pd.DataFrame,
                    end_date: str,
                    lookback_days: int = 60,
                    core_holder_types: Optional[set[str]] = None,
                    min_ratio: float = 0.3,
                ) -> pd.DataFrame:
                    if holdertrade_df.empty:
                        return pd.DataFrame()

                    work = holdertrade_df.copy()
                    work["ann_date"] = work["ann_date"].astype(str)
                    work["ann_date_dt"] = pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce")
                    work = work.dropna(subset=["ann_date_dt"]).copy()
                    if "holder_type" in work.columns:
                        work["holder_type"] = work["holder_type"].fillna("").astype(str)
                    else:
                        work["holder_type"] = ""
                    if "holder_name" in work.columns:
                        work["holder_name"] = work["holder_name"].fillna("").astype(str)
                    else:
                        work["holder_name"] = ""
                    if "in_de" in work.columns:
                        work["in_de"] = work["in_de"].fillna("").astype(str).str.upper()
                    else:
                        work["in_de"] = ""
                    if "change_ratio" in work.columns:
                        work["change_ratio"] = pd.to_numeric(work["change_ratio"], errors="coerce").fillna(0.0)
                    else:
                        work["change_ratio"] = 0.0
                    if "close_date" in work.columns:
                        work["close_date"] = work["close_date"].fillna("").astype(str)
                        work["close_date_dt"] = pd.to_datetime(work["close_date"], format="%Y%m%d", errors="coerce")
                    else:
                        work["close_date"] = ""
                        work["close_date_dt"] = pd.NaT

                    core_holder_types = core_holder_types or set()
                    screen_end_ts = pd.Timestamp(end_date)
                    lookback_cutoff = screen_end_ts - pd.Timedelta(days=max(1, lookback_days - 1))
                    recent = work[(work["ann_date_dt"] >= lookback_cutoff) & (work["in_de"] == "DE")].copy()
                    if core_holder_types:
                        recent = recent[recent["holder_type"].isin(sorted(core_holder_types))].copy()
                    if recent.empty:
                        return pd.DataFrame()

                    rows: list[dict] = []
                    for ts_code, sub in recent.groupby("ts_code", dropna=False):
                        active = sub[sub["close_date_dt"].isna() | (sub["close_date_dt"] >= screen_end_ts)].copy()
                        plan_ratio = pd.to_numeric(active["change_ratio"], errors="coerce").fillna(0.0).sum() if not active.empty else 0.0
                        latest = active.sort_values(["ann_date_dt", "change_ratio"], ascending=[False, False]).iloc[0] if not active.empty else sub.sort_values(["ann_date_dt", "change_ratio"], ascending=[False, False]).iloc[0]
                        latest_close_date = latest.get("close_date")
                        rows.append(
                            {
                                "ts_code": ts_code,
                                "active_reduction_plan_flag": bool(not active.empty and plan_ratio >= min_ratio),
                                "active_reduction_plan_ratio": to_number(plan_ratio, 4),
                                "active_reduction_plan_ann_date": latest.get("ann_date"),
                                "active_reduction_plan_close_date": latest_close_date if latest_close_date else None,
                                "active_reduction_holder": latest.get("holder_name"),
                            }
                        )

                    return pd.DataFrame(rows)


                def build_unlock_snapshot(
                    share_float_df: pd.DataFrame,
                    end_date: str,
                    lookahead_days: int = 30,
                    max_near_unlock_ratio: float = 3.0,
                    max_unlock_ratio_30d: float = 8.0,
                ) -> pd.DataFrame:
                    if share_float_df.empty:
                        return pd.DataFrame()

                    work = share_float_df.copy()
                    if "float_date" not in work.columns:
                        return pd.DataFrame()
                    work["float_date"] = work["float_date"].astype(str)
                    work["float_date_dt"] = pd.to_datetime(work["float_date"], format="%Y%m%d", errors="coerce")
                    work = work.dropna(subset=["float_date_dt"]).copy()
                    if "float_ratio" in work.columns:
                        work["float_ratio"] = pd.to_numeric(work["float_ratio"], errors="coerce").fillna(0.0)
                    else:
                        work["float_ratio"] = 0.0

                    screen_end_ts = pd.Timestamp(end_date)
                    future_cutoff = screen_end_ts + pd.Timedelta(days=max(1, lookahead_days))
                    future = work[(work["float_date_dt"] >= screen_end_ts) & (work["float_date_dt"] <= future_cutoff)].copy()
                    if future.empty:
                        return pd.DataFrame()

                    rows: list[dict] = []
                    for ts_code, sub in future.groupby("ts_code", dropna=False):
                        ordered = sub.sort_values(["float_date_dt", "float_ratio"], ascending=[True, False]).reset_index(drop=True)
                        nearest = ordered.iloc[0]
                        nearest_ratio = pd.to_numeric(
                            ordered[ordered["float_date_dt"] == nearest["float_date_dt"]]["float_ratio"],
                            errors="coerce",
                        ).fillna(0.0).sum()
                        total_ratio = pd.to_numeric(ordered["float_ratio"], errors="coerce").fillna(0.0).sum()
                        days_to_unlock = int((nearest["float_date_dt"].normalize() - screen_end_ts.normalize()).days)
                        unlock_veto = bool(
                            (days_to_unlock <= 10 and nearest_ratio >= max_near_unlock_ratio)
                            or total_ratio >= max_unlock_ratio_30d
                        )
                        rows.append(
                            {
                                "ts_code": ts_code,
                                "nearest_unlock_date": nearest["float_date"],
                                "days_to_nearest_unlock": days_to_unlock,
                                "nearest_unlock_ratio": to_number(nearest_ratio, 4),
                                "unlock_ratio_30d": to_number(total_ratio, 4),
                                "unlock_risk_veto": unlock_veto,
                            }
                        )

                    return pd.DataFrame(rows)


                def fetch_recent_moneyflow_summary(
                    pro,
                    trade_dates: list[str],
                    sleep_sec: float = 0.0,
                ) -> pd.DataFrame:
                    if not trade_dates:
                        return pd.DataFrame()
                    frames: list[pd.DataFrame] = []
                    for trade_date in trade_dates:
                        df = safe_call(
                            f"moneyflow_{trade_date}",
                            getattr(pro, "moneyflow", None),
                            sleep_sec=sleep_sec,
                            trade_date=trade_date,
                        )
                        if not df.empty:
                            work = df.copy()
                            work["trade_date"] = trade_date
                            work["main_net_amount"] = compute_main_net_amount(work)
                            frames.append(work)
                    if not frames:
                        return pd.DataFrame()
                    combined = pd.concat(frames, ignore_index=True)
                    combined["main_net_amount"] = pd.to_numeric(combined["main_net_amount"], errors="coerce").fillna(0.0)

                    rows: list[dict] = []
                    latest_3 = set(trade_dates[-3:])
                    latest_5 = set(trade_dates[-5:])
                    for ts_code, sub in combined.groupby("ts_code", dropna=False):
                        ordered = sub.sort_values("trade_date").reset_index(drop=True)
                        positive_mask = pd.to_numeric(ordered["main_net_amount"], errors="coerce").fillna(0.0) > 0
                        consecutive_positive_days = 0
                        for value in positive_mask.iloc[::-1].tolist():
                            if value:
                                consecutive_positive_days += 1
                            else:
                                break
                        rows.append(
                            {
                                "ts_code": ts_code,
                                "main_net_amount_3d": to_number(sub[sub["trade_date"].isin(latest_3)]["main_net_amount"].sum(), 0),
                                "main_net_amount_5d": to_number(sub[sub["trade_date"].isin(latest_5)]["main_net_amount"].sum(), 0),
                                "main_net_positive_days_3d": int((sub[sub["trade_date"].isin(latest_3)]["main_net_amount"] > 0).sum()),
                                "main_net_positive_days_5d": int((sub[sub["trade_date"].isin(latest_5)]["main_net_amount"] > 0).sum()),
                                "main_net_consecutive_days": int(consecutive_positive_days),
                                "moneyflow_days": int(sub["trade_date"].nunique()),
                            }
                        )
                    return pd.DataFrame(rows)


                def build_market_snapshot(
                    stock_basic_df: pd.DataFrame,
                    daily_basic_df: pd.DataFrame,
                    tech_df: pd.DataFrame,
                    moneyflow_summary_df: pd.DataFrame,
                ) -> pd.DataFrame:
                    basic_cols = [c for c in ["ts_code", "symbol", "name", "industry", "market", "list_date"] if c in stock_basic_df.columns]
                    snapshot = stock_basic_df[basic_cols].drop_duplicates(subset=["ts_code"]).copy()

                    if not daily_basic_df.empty:
                        daily_cols = [
                            c
                            for c in [
                                "ts_code",
                                "trade_date",
                                "close",
                                "turnover_rate",
                                "turnover_rate_f",
                                "volume_ratio",
                                "pe",
                                "pe_ttm",
                                "pb",
                                "ps_ttm",
                                "total_mv",
                                "circ_mv",
                            ]
                            if c in daily_basic_df.columns
                        ]
                        snapshot = snapshot.merge(daily_basic_df[daily_cols].drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")

                    if not tech_df.empty:
                        tech_cols = [
                            c
                            for c in [
                                "ts_code",
                                "trade_date",
                                "close_qfq",
                                "ma_qfq_5",
                                "ma_qfq_10",
                                "ma_qfq_20",
                                "ma_qfq_60",
                                "ma_qfq_250",
                                "macd_dif_qfq",
                                "macd_dea_qfq",
                                "vol_ratio",
                            ]
                            if c in tech_df.columns
                        ]
                        tech_prepared = tech_df[tech_cols].drop_duplicates(subset=["ts_code"]).copy()
                        snapshot = snapshot.merge(tech_prepared, on="ts_code", how="left", suffixes=("", "_tech"))

                    if not moneyflow_summary_df.empty:
                        snapshot = snapshot.merge(moneyflow_summary_df, on="ts_code", how="left")

                    snapshot["pb_num"] = pd.to_numeric(snapshot.get("pb"), errors="coerce")
                    if "industry" in snapshot.columns:
                        snapshot["industry_pb_pct_rank"] = snapshot.groupby("industry")["pb_num"].rank(pct=True, method="average")
                        snapshot["industry_pb_pct_rank"] = snapshot["industry_pb_pct_rank"].fillna(0.5)
                    else:
                        snapshot["industry_pb_pct_rank"] = 0.5

                    if "volume_ratio" in snapshot.columns:
                        snapshot["volume_ratio"] = pd.to_numeric(snapshot["volume_ratio"], errors="coerce")
                    if "vol_ratio" in snapshot.columns:
                        vol_ratio_series = pd.to_numeric(snapshot["vol_ratio"], errors="coerce")
                        if "volume_ratio" in snapshot.columns:
                            snapshot["volume_ratio"] = snapshot["volume_ratio"].fillna(vol_ratio_series)
                        else:
                            snapshot["volume_ratio"] = vol_ratio_series

                    return snapshot


                def build_market_regime_snapshot(snapshot: pd.DataFrame) -> dict:
                    if snapshot.empty:
                        return {
                            "market_regime": "unknown",
                            "market_trend_breadth": None,
                            "market_flow_breadth": None,
                            "market_hot_ratio": None,
                            "market_regime_score": None,
                        }

                    work = snapshot.copy()
                    trend_mask = pd.Series(False, index=work.index)
                    if {"close_qfq", "ma_qfq_20"}.issubset(work.columns):
                        trend_mask = (
                            pd.to_numeric(work["close_qfq"], errors="coerce")
                            > pd.to_numeric(work["ma_qfq_20"], errors="coerce")
                        )

                    flow_mask = pd.Series(False, index=work.index)
                    if "main_net_amount_3d" in work.columns:
                        flow_mask = pd.to_numeric(work["main_net_amount_3d"], errors="coerce").fillna(0.0) > 0

                    hot_mask = pd.Series(False, index=work.index)
                    if "volume_ratio" in work.columns:
                        hot_mask = pd.to_numeric(work["volume_ratio"], errors="coerce").fillna(0.0) >= 1.2

                    trend_breadth = float(trend_mask.mean()) if len(trend_mask) else 0.0
                    flow_breadth = float(flow_mask.mean()) if len(flow_mask) else 0.0
                    hot_ratio = float(hot_mask.mean()) if len(hot_mask) else 0.0
                    regime_score = trend_breadth * 50 + flow_breadth * 35 + hot_ratio * 15

                    if regime_score >= 58:
                        regime = "risk_on"
                    elif regime_score >= 46:
                        regime = "neutral"
                    else:
                        regime = "defensive"

                    return {
                        "market_regime": regime,
                        "market_trend_breadth": to_number(trend_breadth * 100),
                        "market_flow_breadth": to_number(flow_breadth * 100),
                        "market_hot_ratio": to_number(hot_ratio * 100),
                        "market_regime_score": to_number(regime_score),
                    }


                def build_qfq_daily(daily_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
                    if daily_df.empty:
                        return pd.DataFrame()

                    work = daily_df.copy()
                    work["trade_date"] = work["trade_date"].astype(str)
                    work["trade_date_dt"] = pd.to_datetime(work["trade_date"], format="%Y%m%d", errors="coerce")
                    work = work.dropna(subset=["trade_date_dt"]).sort_values("trade_date_dt").reset_index(drop=True)
                    work["close"] = pd.to_numeric(work["close"], errors="coerce")

                    if adj_df.empty or "adj_factor" not in adj_df.columns:
                        work["close_qfq_calc"] = work["close"]
                        return work

                    factors = adj_df[["trade_date", "adj_factor"]].copy()
                    factors["trade_date"] = factors["trade_date"].astype(str)
                    factors["trade_date_dt"] = pd.to_datetime(factors["trade_date"], format="%Y%m%d", errors="coerce")
                    factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="coerce")
                    work = work.merge(factors[["trade_date_dt", "adj_factor"]], on="trade_date_dt", how="left")

                    valid_factor = work["adj_factor"].dropna()
                    if valid_factor.empty or valid_factor.iloc[-1] == 0:
                        work["close_qfq_calc"] = work["close"]
                    else:
                        latest_factor = valid_factor.iloc[-1]
                        work["close_qfq_calc"] = work["close"] * work["adj_factor"] / latest_factor
                    return work


                def summarize_price_metrics(daily_df: pd.DataFrame, adj_df: pd.DataFrame, window: int = 250) -> dict:
                    qfq = build_qfq_daily(daily_df, adj_df)
                    if qfq.empty or "close_qfq_calc" not in qfq.columns:
                        return {
                            "latest_close_qfq_calc": None,
                            "price_position_250": None,
                            "range_low_250": None,
                            "range_high_250": None,
                            "return_20d": None,
                            "return_60d": None,
                        }

                    recent = qfq.tail(window).copy()
                    closes = recent["close_qfq_calc"].dropna()
                    if closes.empty:
                        return {
                            "latest_close_qfq_calc": None,
                            "price_position_250": None,
                            "range_low_250": None,
                            "range_high_250": None,
                            "return_20d": None,
                            "return_60d": None,
                        }

                    latest_close = closes.iloc[-1]
                    low_250 = closes.min()
                    high_250 = closes.max()
                    if high_250 == low_250:
                        position = None
                    else:
                        position = (latest_close - low_250) / (high_250 - low_250)

                    ordered = qfq["close_qfq_calc"].dropna().reset_index(drop=True)
                    return_20d = None
                    return_60d = None
                    if len(ordered) > 20 and ordered.iloc[-21] != 0:
                        return_20d = (ordered.iloc[-1] / ordered.iloc[-21] - 1.0) * 100
                    if len(ordered) > 60 and ordered.iloc[-61] != 0:
                        return_60d = (ordered.iloc[-1] / ordered.iloc[-61] - 1.0) * 100

                    return {
                        "latest_close_qfq_calc": to_number(latest_close),
                        "price_position_250": to_number(position, 4),
                        "range_low_250": to_number(low_250),
                        "range_high_250": to_number(high_250),
                        "return_20d": to_number(return_20d),
                        "return_60d": to_number(return_60d),
                    }


                def summarize_indicator_metrics(indicator_df: pd.DataFrame) -> dict:
                    latest = latest_row(indicator_df)
                    return {
                        "report_period": latest.get("end_date"),
                        "roe": to_number(latest.get("roe")),
                        "gross_margin": to_number(latest.get("gross_margin") or latest.get("grossprofit_margin")),
                        "dt_netprofit_yoy": to_number(latest.get("dt_netprofit_yoy")),
                        "netprofit_yoy": to_number(latest.get("netprofit_yoy")),
                        "ocf_yoy": to_number(latest.get("ocf_yoy")),
                        "q_sales_yoy": to_number(latest.get("q_sales_yoy") or latest.get("tr_yoy") or latest.get("or_yoy")),
                        "debt_to_assets": to_number(latest.get("debt_to_assets")),
                    }


                def summarize_forecast_metrics(forecast_df: pd.DataFrame) -> dict:
                    latest = latest_row(forecast_df)
                    forecast_type = str(latest.get("type") or "")
                    negative = bool(forecast_type) and (
                        ("亏" in forecast_type)
                        or (forecast_type in {"预减", "略减", "续亏", "首亏"})
                    )
                    return {
                        "forecast_ann_date": latest.get("ann_date"),
                        "forecast_type": forecast_type or None,
                        "forecast_negative": negative,
                        "forecast_p_change_min": to_number(latest.get("p_change_min")),
                        "forecast_p_change_max": to_number(latest.get("p_change_max")),
                        "forecast_summary": latest.get("summary"),
                    }


                def summarize_cyq_metrics(cyq_df: pd.DataFrame, latest_close: Optional[float] = None) -> dict:
                    ordered = sort_desc(cyq_df)
                    latest = latest_row(ordered)
                    winner_rate_latest = to_float(latest.get("winner_rate"))
                    weight_avg = to_float(latest.get("weight_avg"))
                    cost_50pct = to_float(latest.get("cost_50pct"))

                    winner_change_5d = None
                    if not ordered.empty and "winner_rate" in ordered.columns:
                        winner_series = pd.to_numeric(ordered["winner_rate"], errors="coerce").dropna().reset_index(drop=True)
                        if len(winner_series) > 4:
                            winner_change_5d = winner_series.iloc[0] - winner_series.iloc[4]

                    close_vs_weight_avg = None
                    if latest_close not in (None, 0) and weight_avg not in (None, 0):
                        close_vs_weight_avg = (latest_close / weight_avg - 1.0) * 100

                    return {
                        "winner_rate": to_number(winner_rate_latest),
                        "winner_rate_change_5d": to_number(winner_change_5d),
                        "weight_avg": to_number(weight_avg),
                        "cost_50pct": to_number(cost_50pct),
                        "close_vs_weight_avg_pct": to_number(close_vs_weight_avg),
                    }


                def fetch_single_stock_cyq_metrics(
                    pro,
                    ts_code: str,
                    end_date: str,
                    cyq_lookback_days: int,
                    sleep_sec: float = 0.0,
                    latest_close: Optional[float] = None,
                ) -> dict:
                    cyq_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(60, cyq_lookback_days * 3))).strftime("%Y%m%d")
                    cyq_df = sort_desc(
                        safe_call(
                            f"cyq_perf_{ts_code}",
                            getattr(pro, "cyq_perf", None),
                            sleep_sec=sleep_sec,
                            ts_code=ts_code,
                            start_date=cyq_start_date,
                            end_date=end_date,
                        )
                    )
                    cyq_error = API_ERROR_LOG.get(f"cyq_perf_{ts_code}")
                    return {
                        "ts_code": ts_code,
                        "cyq_checked": not cyq_df.empty,
                        "cyq_rate_limited": is_rate_limit_error(cyq_error),
                        "cyq_error": cyq_error,
                        **summarize_cyq_metrics(cyq_df, latest_close=latest_close),
                    }


                def fetch_single_stock_deep_metrics(
                    pro,
                    ts_code: str,
                    end_date: str,
                    price_lookback_days: int,
                    cyq_lookback_days: int,
                    sleep_sec: float = 0.0,
                    cyq_sleep_sec: float = 0.0,
                    enable_forecast: bool = True,
                    enable_cyq: bool = True,
                ) -> dict:
                    price_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(420, price_lookback_days * 2))).strftime("%Y%m%d")
                    cyq_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(60, cyq_lookback_days * 3))).strftime("%Y%m%d")

                    daily_df = sort_desc(
                        safe_call(
                            f"daily_{ts_code}",
                            getattr(pro, "daily", None),
                            sleep_sec=sleep_sec,
                            ts_code=ts_code,
                            start_date=price_start_date,
                            end_date=end_date,
                        )
                    )
                    adj_df = sort_desc(
                        safe_call(
                            f"adj_factor_{ts_code}",
                            getattr(pro, "adj_factor", None),
                            sleep_sec=sleep_sec,
                            ts_code=ts_code,
                            start_date=price_start_date,
                            end_date=end_date,
                        )
                    )
                    indicator_df = sort_desc(
                        safe_call(
                            f"fina_indicator_{ts_code}",
                            getattr(pro, "fina_indicator", None),
                            sleep_sec=sleep_sec,
                            ts_code=ts_code,
                        )
                    )
                    forecast_df = sort_desc(
                        safe_call(
                            f"forecast_{ts_code}",
                            getattr(pro, "forecast", None) if enable_forecast else None,
                            sleep_sec=sleep_sec,
                            ts_code=ts_code,
                        )
                    )
                    cyq_df = pd.DataFrame()
                    if enable_cyq:
                        cyq_df = sort_desc(
                            safe_call(
                                f"cyq_perf_{ts_code}",
                                getattr(pro, "cyq_perf", None),
                                sleep_sec=cyq_sleep_sec,
                                ts_code=ts_code,
                                start_date=cyq_start_date,
                                end_date=end_date,
                            )
                        )

                    price_metrics = summarize_price_metrics(daily_df, adj_df, window=price_lookback_days)
                    indicator_metrics = summarize_indicator_metrics(indicator_df)
                    forecast_metrics = summarize_forecast_metrics(forecast_df)
                    cyq_metrics = summarize_cyq_metrics(cyq_df, latest_close=to_float(price_metrics.get("latest_close_qfq_calc")))

                    return {
                        "ts_code": ts_code,
                        **price_metrics,
                        **indicator_metrics,
                        **forecast_metrics,
                        **cyq_metrics,
                    }


                def build_earnings_score(row: dict) -> float:
                    score = 0.0

                    if not to_bool(row.get("forecast_negative")):
                        score += 8.0

                    dt_netprofit_yoy = to_float(row.get("dt_netprofit_yoy"))
                    if dt_netprofit_yoy is not None:
                        if dt_netprofit_yoy > 20:
                            score += 10.0
                        elif dt_netprofit_yoy > 0:
                            score += 8.0
                        elif dt_netprofit_yoy > -10:
                            score += 3.0

                    ocf_yoy = to_float(row.get("ocf_yoy"))
                    if ocf_yoy is not None:
                        if ocf_yoy > 15:
                            score += 6.0
                        elif ocf_yoy > 0:
                            score += 4.0

                    roe = to_float(row.get("roe"))
                    if roe is not None:
                        if roe >= 12:
                            score += 6.0
                        elif roe >= 8:
                            score += 5.0
                        elif roe >= 5:
                            score += 3.0

                    debt_to_assets = to_float(row.get("debt_to_assets"))
                    if debt_to_assets is not None:
                        if debt_to_assets <= 50:
                            score += 4.0
                        elif debt_to_assets <= 65:
                            score += 2.0
                        else:
                            score -= 4.0

                    return round(clip_score(score, 0.0, 30.0), 2)


                def build_value_score(row: dict) -> float:
                    score = 0.0
                    position = to_float(row.get("price_position_250"))
                    pb_pct = to_float(row.get("industry_pb_pct_rank"))
                    return_20d = to_float(row.get("return_20d"))

                    if position is not None:
                        score += max(0.0, (0.62 - min(position, 0.62)) / 0.62) * 12.0
                    if pb_pct is not None:
                        score += max(0.0, (0.80 - min(pb_pct, 0.80)) / 0.80) * 8.0
                    if return_20d is not None and return_20d < -12:
                        score -= 2.0

                    return round(clip_score(score, 0.0, 20.0), 2)


                def build_reversal_score(row: dict, min_volume_ratio: float = 1.2) -> float:
                    score = 0.0

                    close_qfq = to_float(row.get("close_qfq"))
                    ma_20 = to_float(row.get("ma_qfq_20"))
                    ma_5 = to_float(row.get("ma_qfq_5"))
                    ma_10 = to_float(row.get("ma_qfq_10"))
                    macd_dif = to_float(row.get("macd_dif_qfq"))
                    macd_dea = to_float(row.get("macd_dea_qfq"))
                    volume_ratio = to_float(row.get("volume_ratio"))
                    net_amount_3d = to_float(row.get("main_net_amount_3d"))
                    net_amount_5d = to_float(row.get("main_net_amount_5d"))
                    winner_change_5d = to_float(row.get("winner_rate_change_5d"))
                    return_20d = to_float(row.get("return_20d"))

                    if close_qfq is not None and ma_20 is not None and close_qfq > ma_20:
                        score += 5.0
                    if ma_5 is not None and ma_10 is not None and ma_5 > ma_10:
                        score += 4.0
                    if macd_dif is not None and macd_dea is not None and macd_dif > macd_dea:
                        score += 4.0
                    if volume_ratio is not None and volume_ratio >= min_volume_ratio:
                        score += 3.0
                    if net_amount_3d is not None and net_amount_3d > 0:
                        score += 2.0
                    if net_amount_5d is not None and net_amount_5d > 0:
                        score += 1.0
                    if winner_change_5d is not None and winner_change_5d > 0:
                        score += 1.0
                    if return_20d is not None and -5 <= return_20d <= 20:
                        score += 1.0

                    return round(clip_score(score, 0.0, 20.0), 2)


                def build_chip_score(row: dict) -> float:
                    score = 0.0
                    winner_rate = to_float(row.get("winner_rate"))
                    winner_change_5d = to_float(row.get("winner_rate_change_5d"))
                    close_vs_weight_avg_pct = to_float(row.get("close_vs_weight_avg_pct"))

                    if winner_rate is not None:
                        if winner_rate >= 60:
                            score += 8.0
                        elif winner_rate >= 45:
                            score += 6.0
                        elif winner_rate >= 30:
                            score += 4.0
                        elif winner_rate >= 15:
                            score += 2.0

                    if winner_change_5d is not None:
                        if winner_change_5d >= 8:
                            score += 8.0
                        elif winner_change_5d > 3:
                            score += 6.0
                        elif winner_change_5d > 0:
                            score += 4.0

                    if close_vs_weight_avg_pct is not None:
                        if -3 <= close_vs_weight_avg_pct <= 5:
                            score += 4.0
                        elif 5 < close_vs_weight_avg_pct <= 10:
                            score += 2.0
                        elif close_vs_weight_avg_pct < -3:
                            score -= 1.0

                    return round(clip_score(score, 0.0, 20.0), 2)


                def build_fund_flow_score(row: dict) -> float:
                    score = 0.0
                    net_amount_3d = to_float(row.get("main_net_amount_3d"))
                    net_amount_5d = to_float(row.get("main_net_amount_5d"))
                    flow_3d_rank = to_float(row.get("main_net_amount_3d_rank_pct"))
                    flow_5d_rank = to_float(row.get("main_net_amount_5d_rank_pct"))
                    volume_ratio = to_float(row.get("volume_ratio"))
                    positive_days_3d = to_float(row.get("main_net_positive_days_3d"))
                    positive_days_5d = to_float(row.get("main_net_positive_days_5d"))
                    consecutive_days = to_float(row.get("main_net_consecutive_days"))

                    if flow_3d_rank is not None:
                        score += flow_3d_rank * 10.0
                    if flow_5d_rank is not None:
                        score += flow_5d_rank * 4.0
                    if net_amount_3d is not None and net_amount_3d > 0:
                        score += 3.0
                    if net_amount_5d is not None and net_amount_5d > 0:
                        score += 1.5
                    if positive_days_3d is not None:
                        score += min(positive_days_3d, 3.0) * 1.2
                    if positive_days_5d is not None:
                        score += min(positive_days_5d, 5.0) * 0.4
                    if consecutive_days is not None:
                        score += min(consecutive_days, 5.0) * 0.8
                    if volume_ratio is not None:
                        if volume_ratio >= 2.0:
                            score += 1.5
                        elif volume_ratio >= 1.2:
                            score += 1.0

                    return round(clip_score(score, 0.0, 20.0), 2)


                def build_event_bonus_score(row: dict) -> float:
                    event_score = to_float(row.get("event_score"))
                    if event_score is None:
                        return 0.0
                    latest_change_dir = str(row.get("latest_change_dir") or "").upper()
                    bonus = event_score * 0.25
                    if latest_change_dir == "IN":
                        bonus += 1.0
                    return round(clip_score(bonus, 0.0, 8.0), 2)


                def build_overheat_penalty_score(row: dict) -> float:
                    score = 0.0
                    price_position = to_float(row.get("price_position_250"))
                    return_20d = to_float(row.get("return_20d"))
                    return_60d = to_float(row.get("return_60d"))
                    close_vs_weight_avg_pct = to_float(row.get("close_vs_weight_avg_pct"))
                    volume_ratio = to_float(row.get("volume_ratio"))

                    if price_position is not None:
                        if price_position >= 0.98:
                            score += 3.0
                        elif price_position >= 0.90:
                            score += 1.5
                    if return_20d is not None:
                        if return_20d >= 45:
                            score += 4.0
                        elif return_20d >= 30:
                            score += 2.0
                    if return_60d is not None and return_60d >= 80:
                        score += 1.5
                    if close_vs_weight_avg_pct is not None:
                        if close_vs_weight_avg_pct >= 12:
                            score += 2.0
                        elif close_vs_weight_avg_pct >= 8:
                            score += 1.0
                    if volume_ratio is not None and volume_ratio >= 2.5 and (return_20d or 0) >= 25:
                        score += 1.0

                    return round(clip_score(score, 0.0, 10.0), 2)


                def build_risk_penalty_score(row: dict) -> float:
                    score = 0.0
                    latest_change_dir = str(row.get("latest_change_dir") or "").upper()
                    recent_decrease_ratio = to_float(row.get("recent_decrease_ratio"))
                    recent_core_decrease_ratio = to_float(row.get("recent_core_decrease_ratio"))
                    recent_signal_balance = to_float(row.get("recent_signal_balance"))
                    active_reduction_plan_ratio = to_float(row.get("active_reduction_plan_ratio"))
                    nearest_unlock_ratio = to_float(row.get("nearest_unlock_ratio"))
                    unlock_ratio_30d = to_float(row.get("unlock_ratio_30d"))

                    if latest_change_dir == "DE":
                        score += 2.5
                    if recent_decrease_ratio is not None:
                        if recent_decrease_ratio >= 1.0:
                            score += 4.0
                        elif recent_decrease_ratio >= 0.3:
                            score += 2.5
                        elif recent_decrease_ratio > 0:
                            score += 1.0
                    if recent_core_decrease_ratio is not None:
                        if recent_core_decrease_ratio >= 0.8:
                            score += 4.0
                        elif recent_core_decrease_ratio >= 0.2:
                            score += 2.0
                    if to_bool(row.get("mixed_signal_flag")):
                        score += 1.5
                    if recent_signal_balance is not None and recent_signal_balance < 0:
                        score += 2.0
                    if to_bool(row.get("active_reduction_plan_flag")):
                        score += 6.0
                    elif active_reduction_plan_ratio is not None and active_reduction_plan_ratio > 0:
                        score += 2.0
                    if to_bool(row.get("unlock_risk_veto")):
                        score += 5.0
                    else:
                        if nearest_unlock_ratio is not None and nearest_unlock_ratio >= 2.0:
                            score += 1.5
                        if unlock_ratio_30d is not None and unlock_ratio_30d >= 5.0:
                            score += 2.0

                    return round(clip_score(score, 0.0, 20.0), 2)


                def build_stable_score(row: dict, market_regime: str = "neutral") -> float:
                    regime_bias = 2.0 if market_regime == "defensive" else (1.0 if market_regime == "neutral" else 0.0)
                    score = (
                        (to_float(row.get("earnings_score")) or 0.0) * 0.90
                        + (to_float(row.get("reversal_score")) or 0.0) * 0.75
                        + (to_float(row.get("fund_flow_score")) or 0.0) * 0.95
                        + (to_float(row.get("value_score")) or 0.0) * 0.35
                        + (to_float(row.get("event_bonus_score")) or 0.0) * 0.35
                        + (to_float(row.get("chip_score")) or 0.0) * 0.50
                        + regime_bias
                        - (to_float(row.get("risk_penalty_score")) or 0.0)
                        - (to_float(row.get("overheat_penalty_score")) or 0.0)
                    )
                    return round(clip_score(score, 0.0, 100.0), 2)


                def build_aggressive_score(row: dict, market_regime: str = "neutral") -> float:
                    regime_bias = 4.0 if market_regime == "risk_on" else (0.0 if market_regime == "neutral" else -4.0)
                    return_20d = to_float(row.get("return_20d"))
                    price_position = to_float(row.get("price_position_250"))
                    volume_ratio = to_float(row.get("volume_ratio"))

                    breakout_bonus = 0.0
                    if return_20d is not None:
                        if -5 <= return_20d <= 18:
                            breakout_bonus += 4.0
                        elif 18 < return_20d <= 30:
                            breakout_bonus += 2.0
                        elif return_20d > 30:
                            breakout_bonus -= 3.0
                    if price_position is not None:
                        if 0.15 <= price_position <= 0.65:
                            breakout_bonus += 4.0
                        elif price_position > 0.85:
                            breakout_bonus -= 2.0
                    if volume_ratio is not None and volume_ratio >= 1.8:
                            breakout_bonus += 2.0

                    score = (
                        (to_float(row.get("earnings_score")) or 0.0) * 0.45
                        + (to_float(row.get("reversal_score")) or 0.0) * 0.90
                        + (to_float(row.get("fund_flow_score")) or 0.0) * 1.05
                        + (to_float(row.get("value_score")) or 0.0) * 0.15
                        + (to_float(row.get("event_bonus_score")) or 0.0) * 0.20
                        + (to_float(row.get("chip_score")) or 0.0) * 0.65
                        + breakout_bonus
                        + regime_bias
                        - (to_float(row.get("risk_penalty_score")) or 0.0) * 0.80
                        - (to_float(row.get("overheat_penalty_score")) or 0.0) * 0.45
                    )
                    return round(clip_score(score, 0.0, 100.0), 2)


                def build_preliminary_score(row: dict) -> float:
                    score = (
                        (to_float(row.get("fund_flow_score")) or 0.0) * 1.05
                        + (to_float(row.get("reversal_score")) or 0.0) * 0.95
                        + (to_float(row.get("event_bonus_score")) or 0.0) * 0.15
                        - (to_float(row.get("risk_penalty_score")) or 0.0) * 0.80
                        - (to_float(row.get("overheat_penalty_score")) or 0.0) * 0.35
                    )
                    return round(clip_score(score, 0.0, 100.0), 2)


                def build_candidate_flags(
                    row: dict,
                    max_price_position: float,
                    max_industry_pb_pct: float,
                    min_volume_ratio: float,
                    min_final_score: float,
                    min_aggressive_score: float,
                ) -> dict:
                    price_position = to_float(row.get("price_position_250"))
                    industry_pb_pct = to_float(row.get("industry_pb_pct_rank"))
                    recent_decrease_ratio = to_float(row.get("recent_decrease_ratio"))
                    overheat_penalty = to_float(row.get("overheat_penalty_score")) or 0.0
                    reduction_plan_veto = to_bool(row.get("active_reduction_plan_flag"))
                    unlock_risk_veto = to_bool(row.get("unlock_risk_veto"))

                    earnings_ok = build_earnings_score(row) >= 18 and not to_bool(row.get("forecast_negative"))
                    value_ok = (
                        price_position is not None
                        and price_position <= max_price_position
                        and industry_pb_pct is not None
                        and industry_pb_pct <= max_industry_pb_pct
                    )
                    reversal_ok = build_reversal_score(row, min_volume_ratio) >= 10
                    fund_flow_ok = build_fund_flow_score(row) >= 12
                    signal_ok = not reduction_plan_veto and not unlock_risk_veto

                    aggressive_value_ok = price_position is not None and 0.10 <= price_position <= 0.70
                    aggressive_signal_ok = (recent_decrease_ratio is None or recent_decrease_ratio < 1.5) and not reduction_plan_veto and not unlock_risk_veto

                    stable_candidate = (
                        earnings_ok
                        and reversal_ok
                        and fund_flow_ok
                        and signal_ok
                        and value_ok
                        and overheat_penalty < 6
                        and (to_float(row.get("stable_score")) or 0.0) >= min_final_score
                    )
                    aggressive_candidate = (
                        not to_bool(row.get("forecast_negative"))
                        and reversal_ok
                        and fund_flow_ok
                        and aggressive_value_ok
                        and aggressive_signal_ok
                        and overheat_penalty < 8
                        and (to_float(row.get("aggressive_score")) or 0.0) >= min_aggressive_score
                    )

                    return {
                        "earnings_ok": earnings_ok,
                        "value_ok": value_ok,
                        "reversal_ok": reversal_ok,
                        "fund_flow_ok": fund_flow_ok,
                        "signal_ok": signal_ok,
                        "reduction_plan_veto": reduction_plan_veto,
                        "unlock_risk_veto": unlock_risk_veto,
                        "stable_candidate": stable_candidate,
                        "aggressive_candidate": aggressive_candidate,
                    }


                def display_columns() -> list[str]:
                    return [
                        "ts_code",
                        "name",
                        "industry",
                        "market_regime",
                        "preferred_pool",
                        "priority_score",
                        "stable_score",
                        "aggressive_score",
                        "preliminary_score",
                        "final_score",
                        "event_bonus_score",
                        "earnings_score",
                        "value_score",
                        "reversal_score",
                        "fund_flow_score",
                        "overheat_penalty_score",
                        "risk_penalty_score",
                        "latest_ann_date",
                        "latest_change_dir",
                        "event_count",
                        "holder_type_tags",
                        "total_change_ratio",
                        "recent_decrease_ratio",
                        "recent_core_decrease_ratio",
                        "active_reduction_plan_flag",
                        "active_reduction_plan_ratio",
                        "unlock_risk_veto",
                        "nearest_unlock_date",
                        "nearest_unlock_ratio",
                        "unlock_ratio_30d",
                        "cyq_checked",
                        "chip_score",
                        "volume_ratio",
                        "price_position_250",
                        "industry_pb_pct_rank",
                        "main_net_amount_3d",
                        "main_net_amount_5d",
                        "main_net_positive_days_3d",
                        "main_net_positive_days_5d",
                        "main_net_consecutive_days",
                        "winner_rate",
                        "winner_rate_change_5d",
                        "return_20d",
                        "dt_netprofit_yoy",
                        "ocf_yoy",
                        "roe",
                        "debt_to_assets",
                        "forecast_type",
                    ]


                def score_candidates(
                    df: pd.DataFrame,
                    min_volume_ratio: float,
                    max_price_position: float,
                    max_industry_pb_pct: float,
                    min_final_score: float,
                    min_aggressive_score: float,
                    market_regime: str,
                ) -> pd.DataFrame:
                    work = df.copy()
                    work = work.drop(
                        columns=[
                            "earnings_score",
                            "value_score",
                            "reversal_score",
                            "fund_flow_score",
                            "event_bonus_score",
                            "overheat_penalty_score",
                            "chip_score",
                            "risk_penalty_score",
                            "preliminary_score",
                            "final_score",
                            "stable_score",
                            "aggressive_score",
                            "priority_score",
                            "preferred_pool",
                            "earnings_ok",
                            "value_ok",
                            "reversal_ok",
                            "fund_flow_ok",
                            "signal_ok",
                            "reduction_plan_veto",
                            "unlock_risk_veto",
                            "stable_candidate",
                            "aggressive_candidate",
                            "is_focus_candidate",
                            "main_net_amount_3d_rank_pct",
                            "main_net_amount_5d_rank_pct",
                        ],
                        errors="ignore",
                    )

                    def numeric_series(column: str) -> pd.Series:
                        if column in work.columns:
                            return pd.to_numeric(work[column], errors="coerce")
                        return pd.Series(np.nan, index=work.index, dtype=float)

                    work["main_net_amount_3d_rank_pct"] = rank_pct(numeric_series("main_net_amount_3d"))
                    work["main_net_amount_5d_rank_pct"] = rank_pct(numeric_series("main_net_amount_5d"))
                    work["earnings_score"] = work.apply(lambda row: build_earnings_score(row.to_dict()), axis=1)
                    work["value_score"] = work.apply(lambda row: build_value_score(row.to_dict()), axis=1)
                    work["reversal_score"] = work.apply(
                        lambda row: build_reversal_score(row.to_dict(), min_volume_ratio),
                        axis=1,
                    )
                    work["event_bonus_score"] = work.apply(lambda row: build_event_bonus_score(row.to_dict()), axis=1)
                    work["chip_score"] = work.apply(lambda row: build_chip_score(row.to_dict()), axis=1)
                    work["fund_flow_score"] = work.apply(lambda row: build_fund_flow_score(row.to_dict()), axis=1)
                    work["overheat_penalty_score"] = work.apply(lambda row: build_overheat_penalty_score(row.to_dict()), axis=1)
                    work["risk_penalty_score"] = work.apply(lambda row: build_risk_penalty_score(row.to_dict()), axis=1)
                    work["preliminary_score"] = work.apply(lambda row: build_preliminary_score(row.to_dict()), axis=1)

                    work["final_score"] = (
                        pd.to_numeric(work["earnings_score"], errors="coerce").fillna(0.0) * 0.85
                        + pd.to_numeric(work["reversal_score"], errors="coerce").fillna(0.0) * 0.95
                        + pd.to_numeric(work["fund_flow_score"], errors="coerce").fillna(0.0) * 1.10
                        + pd.to_numeric(work["chip_score"], errors="coerce").fillna(0.0) * 0.40
                        + pd.to_numeric(work["value_score"], errors="coerce").fillna(0.0) * 0.20
                        + pd.to_numeric(work["event_bonus_score"], errors="coerce").fillna(0.0) * 0.35
                        - pd.to_numeric(work["risk_penalty_score"], errors="coerce").fillna(0.0)
                        - pd.to_numeric(work["overheat_penalty_score"], errors="coerce").fillna(0.0) * 0.60
                    ).clip(lower=0.0, upper=100.0).round(2)
                    work["stable_score"] = work.apply(lambda row: build_stable_score(row.to_dict(), market_regime), axis=1)
                    work["aggressive_score"] = work.apply(lambda row: build_aggressive_score(row.to_dict(), market_regime), axis=1)

                    if market_regime == "risk_on":
                        work["priority_score"] = (
                            pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0) * 0.60
                            + pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0) * 0.40
                        ).round(2)
                    elif market_regime == "defensive":
                        work["priority_score"] = (
                            pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0) * 0.70
                            + pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0) * 0.30
                        ).round(2)
                    else:
                        work["priority_score"] = (
                            pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0) * 0.55
                            + pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0) * 0.45
                        ).round(2)

                    work["market_regime"] = market_regime
                    work["preferred_pool"] = np.where(
                        pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0)
                        > pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0),
                        "aggressive",
                        "stable",
                    )

                    flag_rows = work.apply(
                        lambda row: build_candidate_flags(
                            row.to_dict(),
                            max_price_position=max_price_position,
                            max_industry_pb_pct=max_industry_pb_pct,
                            min_volume_ratio=min_volume_ratio,
                            min_final_score=min_final_score,
                            min_aggressive_score=min_aggressive_score,
                        ),
                        axis=1,
                        result_type="expand",
                    )
                    work = pd.concat([work, flag_rows], axis=1)
                    work["is_focus_candidate"] = work["stable_candidate"].fillna(False)

                    return work.sort_values(
                        ["priority_score", "stable_candidate", "aggressive_candidate", "final_score", "fund_flow_score", "reversal_score", "event_bonus_score"],
                        ascending=[False, False, False, False, False, False, False],
                    ).reset_index(drop=True)
                """
            ),
            code_cell(
                """
                ensure_token(TOKEN)
                pro = ts.pro_api(TOKEN)
                pro = configure_tushare_client(pro, TOKEN, USE_CUSTOM_HTTP_ENDPOINT, CUSTOM_HTTP_URL)
                print(
                    {
                        "use_custom_http_endpoint": USE_CUSTOM_HTTP_ENDPOINT,
                        "custom_http_url": CUSTOM_HTTP_URL if USE_CUSTOM_HTTP_ENDPOINT else "",
                    }
                )

                allowed_holder_type_set = {value.strip().upper() for value in ALLOWED_HOLDER_TYPES if value and value.strip()}
                screen_end_date = choose_screen_end_date(pd.Timestamp.now(), END_DATE, TODAY_STR, MARKET_DATA_CUTOFF_HOUR)
                recent_trade_dates = get_recent_open_trade_dates(pro, screen_end_date, count=max(MONEYFLOW_LOOKBACK_DAYS, 10))

                stock_basic_all = safe_call(
                    "stock_basic_all",
                    getattr(pro, "stock_basic", None),
                    exchange="",
                    list_status="L",
                    fields="ts_code,symbol,name,area,industry,market,list_date",
                )
                if stock_basic_all.empty:
                    raise ValueError("stock_basic 接口未返回数据，无法继续。")
                stock_basic_all = stock_basic_all.fillna("")

                holdertrade_raw = fetch_holdertrade_events(
                    pro,
                    ANN_START_DATE,
                    screen_end_date,
                    chunk_days=EVENT_CHUNK_DAYS,
                    sleep_sec=API_SLEEP_SEC,
                )
                event_pool = prepare_event_pool(holdertrade_raw, stock_basic_all, allowed_holder_type_set, screen_end_date)
                if not event_pool.empty:
                    event_pool = event_pool[
                        (event_pool["event_count"] >= MIN_EVENT_COUNT)
                        & (pd.to_numeric(event_pool["total_change_ratio"], errors="coerce").fillna(0.0) >= MIN_TOTAL_CHANGE_RATIO)
                    ].reset_index(drop=True)

                reverse_signal_snapshot = build_reverse_signal_snapshot(
                    holdertrade_raw,
                    screen_end_date,
                    lookback_days=RECENT_SIGNAL_LOOKBACK_DAYS,
                    core_holder_types=allowed_holder_type_set,
                )
                reduction_plan_snapshot = build_reduction_plan_snapshot(
                    holdertrade_raw,
                    screen_end_date,
                    lookback_days=max(45, RECENT_SIGNAL_LOOKBACK_DAYS * 2),
                    core_holder_types=allowed_holder_type_set,
                    min_ratio=ACTIVE_REDUCTION_MIN_RATIO,
                )
                unlock_end_date = (pd.Timestamp(screen_end_date) + pd.Timedelta(days=UNLOCK_LOOKAHEAD_DAYS)).strftime("%Y%m%d")
                share_float_schedule = fetch_share_float_schedule(
                    pro,
                    start_date=screen_end_date,
                    end_date=unlock_end_date,
                    chunk_days=10,
                    sleep_sec=API_SLEEP_SEC,
                )
                unlock_snapshot = build_unlock_snapshot(
                    share_float_schedule,
                    screen_end_date,
                    lookahead_days=UNLOCK_LOOKAHEAD_DAYS,
                    max_near_unlock_ratio=MAX_NEAR_UNLOCK_RATIO,
                    max_unlock_ratio_30d=MAX_UNLOCK_RATIO_30D,
                )

                latest_trade_date, market_moneyflow_dates, daily_basic_latest, tech_latest, moneyflow_summary = fetch_latest_complete_market_inputs(
                    pro,
                    recent_trade_dates,
                    moneyflow_lookback_days=MONEYFLOW_LOOKBACK_DAYS,
                    sleep_sec=API_SLEEP_SEC,
                )

                market_snapshot = build_market_snapshot(
                    stock_basic_all,
                    daily_basic_latest,
                    tech_latest,
                    moneyflow_summary,
                )
                market_regime_snapshot = build_market_regime_snapshot(market_snapshot)
                market_regime = market_regime_snapshot["market_regime"]
                print(market_regime_snapshot)

                candidate_base = stock_basic_all.merge(
                    market_snapshot,
                    on="ts_code",
                    how="left",
                    suffixes=("", "_mkt"),
                )
                if not event_pool.empty:
                    candidate_base = candidate_base.merge(event_pool, on="ts_code", how="left", suffixes=("", "_evt"))
                if not reverse_signal_snapshot.empty:
                    candidate_base = candidate_base.merge(reverse_signal_snapshot, on="ts_code", how="left")
                if not reduction_plan_snapshot.empty:
                    candidate_base = candidate_base.merge(reduction_plan_snapshot, on="ts_code", how="left")
                if not unlock_snapshot.empty:
                    candidate_base = candidate_base.merge(unlock_snapshot, on="ts_code", how="left")
                for col in ["name_mkt", "industry_mkt", "market_mkt"]:
                    if col in candidate_base.columns:
                        candidate_base[col] = candidate_base[col].fillna("")

                for col in [
                    "close",
                    "volume_ratio",
                    "pb",
                    "pe_ttm",
                    "total_mv",
                    "close_qfq",
                    "ma_qfq_5",
                    "ma_qfq_10",
                    "ma_qfq_20",
                    "ma_qfq_250",
                    "macd_dif_qfq",
                    "macd_dea_qfq",
                    "main_net_amount_3d",
                    "main_net_amount_5d",
                    "main_net_positive_days_3d",
                    "main_net_positive_days_5d",
                    "main_net_consecutive_days",
                    "industry_pb_pct_rank",
                    "event_score",
                    "total_change_ratio",
                    "event_count",
                    "recent_increase_ratio",
                    "recent_decrease_ratio",
                    "recent_core_decrease_ratio",
                    "recent_signal_balance",
                    "active_reduction_plan_ratio",
                    "nearest_unlock_ratio",
                    "unlock_ratio_30d",
                ]:
                    if col in candidate_base.columns:
                        candidate_base[col] = pd.to_numeric(candidate_base[col], errors="coerce")
                for col in [
                    "event_score",
                    "total_change_ratio",
                    "event_count",
                    "announcement_days",
                    "holder_count",
                    "recent_increase_ratio",
                    "recent_decrease_ratio",
                    "recent_core_decrease_ratio",
                    "recent_signal_balance",
                    "main_net_amount_3d",
                    "main_net_amount_5d",
                    "main_net_positive_days_3d",
                    "main_net_positive_days_5d",
                    "main_net_consecutive_days",
                    "active_reduction_plan_ratio",
                    "nearest_unlock_ratio",
                    "unlock_ratio_30d",
                ]:
                    if col in candidate_base.columns:
                        candidate_base[col] = candidate_base[col].fillna(0.0)
                for col in [
                    "mixed_signal_flag",
                    "active_reduction_plan_flag",
                    "unlock_risk_veto",
                ]:
                    if col in candidate_base.columns:
                        candidate_base[col] = candidate_base[col].fillna(False)
                for col in [
                    "holder_type_tags",
                    "holder_preview",
                    "latest_change_dir",
                    "latest_change_date",
                    "latest_change_holder",
                    "active_reduction_plan_ann_date",
                    "active_reduction_plan_close_date",
                    "active_reduction_holder",
                    "nearest_unlock_date",
                ]:
                    if col in candidate_base.columns:
                        candidate_base[col] = candidate_base[col].fillna("")
                if "market_regime" not in candidate_base.columns:
                    candidate_base["market_regime"] = market_regime
                candidate_base["market_regime"] = candidate_base["market_regime"].fillna(market_regime)
                candidate_base = score_candidates(
                    candidate_base,
                    min_volume_ratio=MIN_VOLUME_RATIO,
                    max_price_position=MAX_PRICE_POSITION,
                    max_industry_pb_pct=MAX_INDUSTRY_PB_PCT,
                    min_final_score=MIN_FINAL_SCORE,
                    min_aggressive_score=MIN_AGGRESSIVE_SCORE,
                    market_regime=market_regime,
                )

                preview_cols = [
                    "ts_code",
                    "name",
                    "industry",
                    "market_regime",
                    "preliminary_score",
                    "fund_flow_score",
                    "reversal_score",
                    "event_bonus_score",
                    "volume_ratio",
                    "main_net_amount_3d",
                    "main_net_amount_5d",
                    "main_net_consecutive_days",
                    "overheat_penalty_score",
                    "active_reduction_plan_flag",
                    "unlock_risk_veto",
                    "recent_decrease_ratio",
                ]
                candidate_base = ensure_columns(candidate_base, preview_cols)
                candidate_base[preview_cols].head(20)
                """
            ),
            code_cell(
                """
                deep_dive_targets = candidate_base.sort_values(
                    ["preliminary_score", "fund_flow_score", "reversal_score", "event_bonus_score"],
                    ascending=[False, False, False, False],
                ).head(MAX_DEEP_DIVE_STOCKS)["ts_code"].dropna().tolist()

                deep_rows: list[dict] = []
                total_targets = len(deep_dive_targets)
                for idx, ts_code in enumerate(deep_dive_targets, start=1):
                    print(f"[stage1 {idx}/{total_targets}] deep dive {ts_code}")
                    deep_rows.append(
                        fetch_single_stock_deep_metrics(
                            pro,
                            ts_code=ts_code,
                            end_date=screen_end_date,
                            price_lookback_days=PRICE_LOOKBACK_DAYS,
                            cyq_lookback_days=CYQ_LOOKBACK_DAYS,
                            sleep_sec=API_SLEEP_SEC,
                            cyq_sleep_sec=CYQ_SLEEP_SEC,
                            enable_forecast=ENABLE_FORECAST,
                            enable_cyq=False,
                        )
                    )

                deep_metrics_stage1 = pd.DataFrame(deep_rows)
                screened_stage1 = candidate_base.drop(
                    columns=[
                        "earnings_score",
                        "value_score",
                        "reversal_score",
                        "event_bonus_score",
                        "chip_score",
                        "fund_flow_score",
                        "overheat_penalty_score",
                        "risk_penalty_score",
                        "preliminary_score",
                        "final_score",
                        "stable_score",
                        "aggressive_score",
                        "priority_score",
                        "preferred_pool",
                        "earnings_ok",
                        "value_ok",
                        "reversal_ok",
                        "fund_flow_ok",
                        "signal_ok",
                        "reduction_plan_veto",
                        "unlock_risk_veto",
                        "stable_candidate",
                        "aggressive_candidate",
                        "is_focus_candidate",
                        "main_net_amount_3d_rank_pct",
                        "main_net_amount_5d_rank_pct",
                    ],
                    errors="ignore",
                ).merge(deep_metrics_stage1, on="ts_code", how="left")
                screened_stage1["cyq_checked"] = False
                screened_stage1 = score_candidates(
                    screened_stage1,
                    min_volume_ratio=MIN_VOLUME_RATIO,
                    max_price_position=MAX_PRICE_POSITION,
                    max_industry_pb_pct=MAX_INDUSTRY_PB_PCT,
                    min_final_score=MIN_FINAL_SCORE,
                    min_aggressive_score=MIN_AGGRESSIVE_SCORE,
                    market_regime=market_regime,
                )
                stable_focus_stage1 = screened_stage1[screened_stage1["stable_candidate"]].reset_index(drop=True)
                aggressive_focus_stage1 = screened_stage1[screened_stage1["aggressive_candidate"]].reset_index(drop=True)
                focus_candidates_stage1 = stable_focus_stage1.copy()
                preferred_stage1 = screened_stage1[
                    screened_stage1["stable_candidate"].fillna(False) | screened_stage1["aggressive_candidate"].fillna(False)
                ].reset_index(drop=True)
                ranked_candidates_stage1 = (
                    preferred_stage1.head(TOP_N_STAGE1).copy()
                    if not preferred_stage1.empty
                    else screened_stage1.head(TOP_N_STAGE1).copy()
                )

                export_dir = OUTPUT_DIR / f"holder_increase_screen_{TODAY_STR}"
                export_dir.mkdir(parents=True, exist_ok=True)

                holdertrade_raw.to_csv(export_dir / "holdertrade_raw.csv", index=False)
                event_pool.to_csv(export_dir / "event_pool.csv", index=False)
                reverse_signal_snapshot.to_csv(export_dir / "reverse_signal_snapshot.csv", index=False)
                reduction_plan_snapshot.to_csv(export_dir / "reduction_plan_snapshot.csv", index=False)
                share_float_schedule.to_csv(export_dir / "share_float_schedule.csv", index=False)
                unlock_snapshot.to_csv(export_dir / "unlock_snapshot.csv", index=False)
                candidate_base.to_csv(export_dir / "candidate_base.csv", index=False)
                deep_metrics_stage1.to_csv(export_dir / "deep_metrics_stage1.csv", index=False)
                screened_stage1.to_csv(export_dir / "screened_candidates_stage1.csv", index=False)
                ranked_candidates_stage1.to_csv(export_dir / "ranked_candidates_stage1.csv", index=False)

                screen_summary = {
                    "latest_trade_date": latest_trade_date,
                    "ann_start_date": ANN_START_DATE,
                    "requested_end_date": END_DATE,
                    "screen_end_date": screen_end_date,
                    "market_moneyflow_dates": market_moneyflow_dates,
                    "market_regime": market_regime,
                    "market_regime_score": market_regime_snapshot.get("market_regime_score"),
                    "market_trend_breadth": market_regime_snapshot.get("market_trend_breadth"),
                    "market_flow_breadth": market_regime_snapshot.get("market_flow_breadth"),
                    "market_hot_ratio": market_regime_snapshot.get("market_hot_ratio"),
                    "unlock_rows_lookahead": int(len(unlock_snapshot)),
                    "active_reduction_plan_rows": int(len(reduction_plan_snapshot)),
                    "raw_event_rows": int(len(holdertrade_raw)),
                    "event_pool_stocks": int(len(event_pool)),
                    "candidate_base_stocks": int(len(candidate_base)),
                    "deep_dive_stocks_stage1": int(len(deep_metrics_stage1)),
                    "focus_candidates_stage1": int(len(focus_candidates_stage1)),
                    "stable_focus_stage1": int(len(stable_focus_stage1)),
                    "aggressive_focus_stage1": int(len(aggressive_focus_stage1)),
                    "top_output_rows_stage1": int(len(ranked_candidates_stage1)),
                    "export_dir": str(export_dir.resolve()),
                }

                summary_path = export_dir / "screen_summary.json"
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump(json_safe(screen_summary), f, ensure_ascii=False, indent=2)

                print(screen_summary)

                ranked_candidates_stage1 = ensure_columns(ranked_candidates_stage1, display_columns())
                ranked_candidates_stage1[display_columns()].head(TOP_N_STAGE1)
                """
            ),
            markdown_cell(
                """
                ## 第二阶段：补筹码并重排

                - 第一阶段先用 `资金 + 趋势 + 业绩安全` 做全市场快筛，增减持和解禁只做修正项，固定留下 `TOP_N_STAGE1=10` 只。
                - 第二阶段只对这 10 只股票补 `cyq_perf`。
                - 如果退回官方接口，`STAGE2_CYQ_BUDGET` 可以控制这次最多新增拉取几只筹码数据；已经缓存过的股票不会重复消耗额度。
                - 补完后会额外计算 `chip_score`，同时叠加 `过热降权 / 减持计划否决 / 解禁风险否决 / 连续资金流入加分`，输出 `稳健池 Top 5`、`进攻池 Top 3` 和 `今日首选`。
                """
            ),
            code_cell(
                """
                stage2_targets = []
                if ENABLE_STAGE2_CYQ and not ranked_candidates_stage1.empty:
                    stage2_targets = ranked_candidates_stage1["ts_code"].dropna().head(MAX_STAGE2_CYQ_STOCKS).tolist()

                stage2_cache_path = export_dir / "stage2_cyq_metrics.csv"
                cached_stage2_cyq = pd.DataFrame()
                if stage2_cache_path.exists():
                    cached_stage2_cyq = pd.read_csv(stage2_cache_path)
                    if "ts_code" in cached_stage2_cyq.columns:
                        cached_stage2_cyq = cached_stage2_cyq[cached_stage2_cyq["ts_code"].isin(stage2_targets)].copy()
                        cached_stage2_cyq = cached_stage2_cyq.drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)
                    else:
                        cached_stage2_cyq = pd.DataFrame()

                cached_targets = set(cached_stage2_cyq["ts_code"].tolist()) if not cached_stage2_cyq.empty and "ts_code" in cached_stage2_cyq.columns else set()
                fetch_targets = [ts_code for ts_code in stage2_targets if ts_code not in cached_targets][:STAGE2_CYQ_BUDGET]

                stage2_cyq_rows: list[dict] = []
                total_stage2 = len(fetch_targets)
                for idx, ts_code in enumerate(fetch_targets, start=1):
                    latest_close = None
                    matched = ranked_candidates_stage1[ranked_candidates_stage1["ts_code"] == ts_code]
                    if not matched.empty:
                        latest_close = to_float(matched.iloc[0].get("latest_close_qfq_calc"))
                    print(f"[stage2 {idx}/{total_stage2}] cyq fetch {ts_code}")
                    cyq_row = fetch_single_stock_cyq_metrics(
                        pro,
                        ts_code=ts_code,
                        end_date=screen_end_date,
                        cyq_lookback_days=CYQ_LOOKBACK_DAYS,
                        sleep_sec=CYQ_SLEEP_SEC,
                        latest_close=latest_close,
                    )
                    stage2_cyq_rows.append(cyq_row)
                    if cyq_row.get("cyq_rate_limited"):
                        print("cyq_perf 已触发限额，停止后续筹码请求；保留第一阶段前5结果。")
                        break

                fetched_stage2_cyq = pd.DataFrame(stage2_cyq_rows)
                stage2_cyq_metrics = pd.concat([cached_stage2_cyq, fetched_stage2_cyq], ignore_index=True) if not cached_stage2_cyq.empty or not fetched_stage2_cyq.empty else pd.DataFrame()
                if not stage2_cyq_metrics.empty and "ts_code" in stage2_cyq_metrics.columns:
                    stage2_cyq_metrics = stage2_cyq_metrics.drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)

                reranked_candidates = ranked_candidates_stage1.copy()
                reranked_candidates = reranked_candidates.drop(
                    columns=["cyq_checked", "cyq_rate_limited", "cyq_error", "winner_rate", "winner_rate_change_5d", "weight_avg", "cost_50pct", "close_vs_weight_avg_pct"],
                    errors="ignore",
                )
                reranked_candidates = reranked_candidates.merge(stage2_cyq_metrics, on="ts_code", how="left")
                reranked_candidates = ensure_columns(
                    reranked_candidates,
                    ["cyq_checked", "cyq_rate_limited", "cyq_error", "winner_rate", "winner_rate_change_5d", "weight_avg", "cost_50pct", "close_vs_weight_avg_pct"],
                )
                reranked_candidates["cyq_checked"] = reranked_candidates["cyq_checked"].fillna(False)
                reranked_candidates["cyq_rate_limited"] = reranked_candidates["cyq_rate_limited"].fillna(False)

                reranked_candidates = score_candidates(
                    reranked_candidates,
                    min_volume_ratio=MIN_VOLUME_RATIO,
                    max_price_position=MAX_PRICE_POSITION,
                    max_industry_pb_pct=MAX_INDUSTRY_PB_PCT,
                    min_final_score=MIN_FINAL_SCORE,
                    min_aggressive_score=MIN_AGGRESSIVE_SCORE,
                    market_regime=market_regime,
                ).reset_index(drop=True)
                reranked_candidates = reranked_candidates.sort_values(
                    ["priority_score", "cyq_checked", "chip_score", "final_score", "event_score", "total_change_ratio"],
                    ascending=[False, False, False, False, False, False],
                ).reset_index(drop=True)

                stable_focus_stage2 = reranked_candidates[reranked_candidates["stable_candidate"]].reset_index(drop=True)
                aggressive_focus_stage2 = reranked_candidates[reranked_candidates["aggressive_candidate"]].reset_index(drop=True)
                stable_candidates = (
                    stable_focus_stage2.sort_values(["stable_score", "priority_score", "chip_score"], ascending=[False, False, False]).head(TOP_N_FINAL).copy()
                    if not stable_focus_stage2.empty
                    else reranked_candidates.sort_values(["stable_score", "priority_score", "chip_score"], ascending=[False, False, False]).head(TOP_N_FINAL).copy()
                )
                aggressive_candidates = (
                    aggressive_focus_stage2.sort_values(["aggressive_score", "priority_score", "chip_score"], ascending=[False, False, False]).head(TOP_N_AGGRESSIVE).copy()
                    if not aggressive_focus_stage2.empty
                    else reranked_candidates.sort_values(["aggressive_score", "priority_score", "chip_score"], ascending=[False, False, False]).head(TOP_N_AGGRESSIVE).copy()
                )

                today_direction = "偏进攻" if market_regime == "risk_on" else "偏稳健"
                today_direction_pool = aggressive_candidates if today_direction == "偏进攻" else stable_candidates
                if today_direction_pool.empty:
                    today_direction_pool = stable_candidates if not stable_candidates.empty else aggressive_candidates
                best_pick_candidate = today_direction_pool.head(1).copy()
                final_candidates = stable_candidates.copy()

                stage2_cyq_metrics.to_csv(export_dir / "stage2_cyq_metrics.csv", index=False)
                reranked_candidates.to_csv(export_dir / "reranked_candidates_stage2.csv", index=False)
                stable_candidates.to_csv(export_dir / "stable_candidates.csv", index=False)
                aggressive_candidates.to_csv(export_dir / "aggressive_candidates.csv", index=False)
                final_candidates.to_csv(export_dir / "final_candidates.csv", index=False)
                best_pick_candidate.to_csv(export_dir / "best_pick_candidate.csv", index=False)

                screen_summary["stage2_cyq_checked"] = int(pd.to_numeric(reranked_candidates["cyq_checked"], errors="coerce").fillna(0).sum())
                screen_summary["stage2_cyq_rate_limited"] = bool(reranked_candidates["cyq_rate_limited"].fillna(False).any())
                screen_summary["stage2_cyq_cached"] = int(len(cached_targets))
                screen_summary["stage2_cyq_requested_this_run"] = int(len(fetch_targets))
                screen_summary["stage2_cyq_pending"] = int(max(0, len(stage2_targets) - int(pd.to_numeric(reranked_candidates["cyq_checked"], errors="coerce").fillna(0).sum())))
                screen_summary["top_output_rows_stage2"] = int(len(final_candidates))
                screen_summary["stable_focus_stage2"] = int(len(stable_focus_stage2))
                screen_summary["aggressive_focus_stage2"] = int(len(aggressive_focus_stage2))
                screen_summary["today_direction"] = today_direction
                screen_summary["best_pick_ts_code"] = best_pick_candidate.iloc[0]["ts_code"] if not best_pick_candidate.empty else None
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump(json_safe(screen_summary), f, ensure_ascii=False, indent=2)

                print(screen_summary)

                stable_candidates = ensure_columns(stable_candidates, display_columns())
                aggressive_candidates = ensure_columns(aggressive_candidates, display_columns())
                best_pick_candidate = ensure_columns(best_pick_candidate, display_columns())

                print(f"today_direction={today_direction}")
                display(stable_candidates[display_columns()].head(TOP_N_FINAL))
                display(aggressive_candidates[display_columns()].head(TOP_N_AGGRESSIVE))
                display(best_pick_candidate[display_columns()].head(1))
                """
            ),
            markdown_cell(
                """
                ## 怎么看结果

                - `earnings_score`：业绩安全分，主看 `forecast` 和 `fina_indicator`，这是主框架的一部分。
                - `reversal_score`：趋势启动分，主看均线、MACD、量比和量价是否转强。
                - `fund_flow_score`：资金持续性分，主看 `3日/5日净流入 + 连续净流入天数`，这是主框架里的核心。
                - `event_bonus_score`：增减持辅助分，不再主导选股，只做加减分。
                - `overheat_penalty_score`：过热降权，涨太急、位置太高、离筹码均价太远会被扣分。
                - `risk_penalty_score`：风险扣分，最近减持、减持计划、解禁压力都会体现在这里。
                - `active_reduction_plan_flag / unlock_risk_veto`：这两项是硬风控，触发后原则上不进主候选。
                - `stable_candidate=True`：更适合弱市和震荡市，强调 `资金持续 + 趋势不坏 + 业绩安全 + 风控干净`。
                - `aggressive_candidate=True`：更适合强市和情绪扩散阶段，强调 `资金加速 + 趋势扩张 + 筹码跟随`，但仍然要避开减持/解禁硬风险。
                - `ranked_candidates_stage1`：第一阶段初筛名单，固定保留 10 只。
                - `stable_candidates`：第二阶段补筹码后的稳健池 5 只，也是 `final_candidates` 的兼容别名。
                - `aggressive_candidates`：第二阶段补筹码后的进攻池 3 只。
                - `today_direction`：当天建议优先看的主方向，只会给你 `偏稳健` 或 `偏进攻`。
                - `best_pick_candidate`：当天建议先看的那 1 只，默认从 `today_direction` 对应池子的第 1 名里选。

                这份 notebook 适合先做全市场初筛。进入候选名单后，建议再人工补三类验证：
                - 公告正文：确认增持是不是被动行为、计划式增持还是管理层真金白银持续买入。
                - 风险事件：排除大额解禁、被立案、ST、商誉减值、质押风险。
                - 行业位置：弱行业里的“相对强”不一定能跑赢强行业里的“低位转强”。
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
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "output" / "jupyter-notebook" / "tushare-holder-increase-screening.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
