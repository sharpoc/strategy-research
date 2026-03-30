from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

import numpy as np
import pandas as pd
import tushare as ts

try:
    from limitup_l1l2_strategy import build_limitup_l1l2_snapshot
except ImportError:
    from scripts.limitup_l1l2_strategy import build_limitup_l1l2_snapshot


API_ERROR_LOG: dict[str, str] = {}
LIMITUP_L1L2_RANK_WEIGHTS: dict[str, float] = {
    "base_pattern_score": 1.0,
    "buy_signal_bonus": 10.0,
    "buy_recent_bonus": 6.0,
    "ready_bonus": 4.0,
    "flow_3d_rank": 8.0,
    "flow_5d_rank": 3.0,
    "volume_ratio_good": 4.0,
    "volume_ratio_ok": 1.5,
    "volume_ratio_overheat_penalty": -2.0,
    "above_ma20_bonus": 3.0,
    "above_ma60_bonus": 2.0,
    "close_vs_l2_near_bonus": 6.0,
    "close_vs_l2_extend_bonus": 3.0,
    "close_vs_l2_overextend_penalty": -5.0,
    "close_vs_l2_break_penalty": -8.0,
    "hold_buffer_bonus": 4.0,
    "hold_buffer_break_penalty": -6.0,
}


def log_step(message: str) -> None:
    print(f"[limitup_l1l2] {message}", flush=True)


def cache_enabled() -> bool:
    return os.getenv("LIMITUP_L1L2_USE_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def cache_root_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = repo_root / "output" / "cache" / "limitup_l1l2_api"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cache_file_path(label: str, kwargs: dict) -> Path:
    normalized = json.dumps(json_safe(kwargs), ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(f"{label}|{normalized}".encode("utf-8")).hexdigest()[:12]
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    return cache_root_dir() / f"{safe_label}_{digest}.csv"


def load_cached_frame(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        log_step(f"cache hit {path.name} rows={len(df)}")
        return df
    except Exception as exc:
        log_step(f"cache read failed {path.name} error={exc}")
        return None


def save_cached_frame(path: Path, df: pd.DataFrame) -> None:
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        log_step(f"cache write failed {path.name} error={exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen the market for the limit-up L1/L2 strategy and pick the strongest stock.")
    parser.add_argument("--end-date", default="", help="Requested end date in YYYYMMDD. Default: today with 20:00 cutoff.")
    parser.add_argument("--history-bars", type=int, default=100, help="Recent open-market bars used for strategy reconstruction.")
    parser.add_argument("--moneyflow-lookback-days", type=int, default=5, help="Moneyflow lookback days.")
    parser.add_argument("--recent-buy-window", type=int, default=0, help="How many bars after a fresh buy signal still count as active. Default 0 keeps it closest to the TradingView script.")
    parser.add_argument("--min-score", type=float, default=55.0, help="Minimum strategy score to keep as a candidate.")
    parser.add_argument("--cutoff-hour", type=int, default=20, help="Use previous trading day before this hour.")
    parser.add_argument("--show-top", type=int, default=10, help="Rows to print from the strategy ranking.")
    return parser.parse_args()


def ensure_token(token: str) -> None:
    if not token or token.startswith("PASTE_"):
        raise SystemExit("Missing TUSHARE_TOKEN in environment.")


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


def to_number(value, digits: int = 2):
    numeric = to_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


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


def safe_call(label: str, fn, sleep_sec: float = 0.0, retries: int = 2, **kwargs) -> pd.DataFrame:
    if fn is None:
        return pd.DataFrame()
    last_exc: Optional[Exception] = None
    cache_path = cache_file_path(label, kwargs) if cache_enabled() else None
    if cache_path is not None:
        cached = load_cached_frame(cache_path)
        if cached is not None:
            return cached
    log_step(f"calling {label}")
    for attempt in range(retries + 1):
        try:
            df = fn(**kwargs)
            API_ERROR_LOG.pop(label, None)
            if sleep_sec:
                time.sleep(sleep_sec)
            if df is None:
                log_step(f"{label} returned 0 rows")
                return pd.DataFrame()
            log_step(f"{label} ok rows={len(df)}")
            if cache_path is not None and not df.empty:
                save_cached_frame(cache_path, df)
            return df.copy()
        except Exception as exc:
            last_exc = exc
            API_ERROR_LOG[label] = str(exc)
            log_step(f"{label} failed attempt={attempt + 1} error={exc}")
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
    print(f"[{label}] 调用失败: {last_exc}")
    return pd.DataFrame()


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


def choose_screen_end_date(now_ts: pd.Timestamp, end_date: str, today_str: str, cutoff_hour: int = 20) -> str:
    requested = end_date or today_str
    if requested == today_str and int(now_ts.hour) < int(cutoff_hour):
        return (now_ts.normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    return requested


def get_recent_open_trade_dates(pro, end_date: str, count: int = 10) -> list[str]:
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


def compute_main_net_amount(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if "net_mf_amount" in df.columns:
        return pd.to_numeric(df["net_mf_amount"], errors="coerce").fillna(0.0)
    buys = pd.Series(0.0, index=df.index)
    sells = pd.Series(0.0, index=df.index)
    for col in ["buy_lg_amount", "buy_elg_amount"]:
        if col in df.columns:
            buys = buys.add(pd.to_numeric(df[col], errors="coerce").fillna(0.0), fill_value=0.0)
    for col in ["sell_lg_amount", "sell_elg_amount"]:
        if col in df.columns:
            sells = sells.add(pd.to_numeric(df[col], errors="coerce").fillna(0.0), fill_value=0.0)
    return buys - sells


def fetch_recent_moneyflow_summary(pro, trade_dates: list[str], sleep_sec: float = 0.0) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        log_step(f"moneyflow summary trade_date={trade_date}")
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


def fetch_latest_complete_market_inputs(
    pro,
    trade_dates: list[str],
    moneyflow_lookback_days: int,
    sleep_sec: float = 0.0,
) -> tuple[str, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fallback_trade_date = trade_dates[-1]
    fallback_moneyflow_dates = trade_dates[-moneyflow_lookback_days:]
    fallback_daily_basic = pd.DataFrame()
    fallback_tech = pd.DataFrame()
    fallback_moneyflow = pd.DataFrame()

    for idx in range(len(trade_dates) - 1, -1, -1):
        trade_date = trade_dates[idx]
        moneyflow_dates = trade_dates[max(0, idx - moneyflow_lookback_days + 1) : idx + 1]
        log_step(f"checking market inputs trade_date={trade_date} moneyflow_dates={','.join(moneyflow_dates)}")
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
        moneyflow_df = fetch_recent_moneyflow_summary(pro, moneyflow_dates, sleep_sec=sleep_sec)

        fallback_trade_date = trade_date
        fallback_moneyflow_dates = moneyflow_dates
        fallback_daily_basic = daily_basic_df
        fallback_tech = tech_df
        fallback_moneyflow = moneyflow_df

        if not daily_basic_df.empty and not tech_df.empty and not moneyflow_df.empty:
            log_step(f"selected complete market inputs trade_date={trade_date}")
            return trade_date, moneyflow_dates, daily_basic_df, tech_df, moneyflow_df

    return fallback_trade_date, fallback_moneyflow_dates, fallback_daily_basic, fallback_tech, fallback_moneyflow


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
        snapshot = snapshot.merge(tech_df[tech_cols].drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")

    if not moneyflow_summary_df.empty:
        snapshot = snapshot.merge(moneyflow_summary_df, on="ts_code", how="left")

    if "volume_ratio" in snapshot.columns:
        snapshot["volume_ratio"] = pd.to_numeric(snapshot["volume_ratio"], errors="coerce")
    if "vol_ratio" in snapshot.columns:
        vol_ratio_series = pd.to_numeric(snapshot["vol_ratio"], errors="coerce")
        if "volume_ratio" in snapshot.columns:
            snapshot["volume_ratio"] = snapshot["volume_ratio"].fillna(vol_ratio_series)
        else:
            snapshot["volume_ratio"] = vol_ratio_series

    return snapshot


def fetch_market_daily_history(pro, trade_dates: list[str], sleep_sec: float = 0.0) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for idx, trade_date in enumerate(trade_dates, start=1):
        log_step(f"daily history {idx}/{len(trade_dates)} trade_date={trade_date}")
        df = safe_call(
            f"daily_all_{trade_date}",
            getattr(pro, "daily", None),
            sleep_sec=sleep_sec,
            trade_date=trade_date,
        )
        if df.empty:
            continue
        keep_cols = [c for c in ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"] if c in df.columns]
        frames.append(df[keep_cols].copy())
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["trade_date"] = combined["trade_date"].astype(str)
    return combined


def build_strategy_rank_score(row: dict) -> float:
    weights = LIMITUP_L1L2_RANK_WEIGHTS
    base_score = to_float(row.get("limitup_l1l2_score")) or 0.0
    buy_signal = bool(row.get("limitup_l1l2_buy_signal"))
    buy_recent = bool(row.get("limitup_l1l2_buy_recent"))
    ready = bool(row.get("limitup_l1l2_ready"))
    buy_age = to_float(row.get("limitup_l1l2_bars_since_buy"))
    close_vs_l2_pct = to_float(row.get("limitup_l1l2_close_vs_l2_pct"))
    hold_buffer_pct = to_float(row.get("limitup_l1l2_hold_buffer_pct"))
    volume_ratio = to_float(row.get("volume_ratio"))
    flow_3d_rank = to_float(row.get("main_net_amount_3d_rank_pct"))
    flow_5d_rank = to_float(row.get("main_net_amount_5d_rank_pct"))
    close_qfq = to_float(row.get("close_qfq"))
    ma_20 = to_float(row.get("ma_qfq_20"))
    ma_60 = to_float(row.get("ma_qfq_60"))

    score = base_score * weights["base_pattern_score"]
    if buy_signal:
        score += weights["buy_signal_bonus"]
    elif buy_recent:
        score += max(2.0, weights["buy_recent_bonus"] - (buy_age or 0.0) * 1.5)
    elif ready:
        score += weights["ready_bonus"]

    if flow_3d_rank is not None:
        score += flow_3d_rank * weights["flow_3d_rank"]
    if flow_5d_rank is not None:
        score += flow_5d_rank * weights["flow_5d_rank"]

    if volume_ratio is not None:
        if 1.2 <= volume_ratio <= 2.8:
            score += weights["volume_ratio_good"]
        elif 0.9 <= volume_ratio < 1.2:
            score += weights["volume_ratio_ok"]
        elif volume_ratio > 4.0:
            score += weights["volume_ratio_overheat_penalty"]

    if close_qfq is not None and ma_20 is not None and close_qfq > ma_20:
        score += weights["above_ma20_bonus"]
    if close_qfq is not None and ma_60 is not None and close_qfq > ma_60:
        score += weights["above_ma60_bonus"]

    if close_vs_l2_pct is not None:
        if 0 <= close_vs_l2_pct <= 6:
            score += weights["close_vs_l2_near_bonus"]
        elif 6 < close_vs_l2_pct <= 12:
            score += weights["close_vs_l2_extend_bonus"]
        elif close_vs_l2_pct > 15:
            score += weights["close_vs_l2_overextend_penalty"]
        elif close_vs_l2_pct < -1.0:
            score += weights["close_vs_l2_break_penalty"]

    if hold_buffer_pct is not None:
        if hold_buffer_pct >= 0.5:
            score += weights["hold_buffer_bonus"]
        elif hold_buffer_pct < 0:
            score += weights["hold_buffer_break_penalty"]

    return round(max(score, 0.0), 2)


def display_columns() -> list[str]:
    return [
        "ts_code",
        "name",
        "industry",
        "strategy_name",
        "strategy_rank_score",
        "limitup_l1l2_score",
        "limitup_l1l2_stage",
        "limitup_l1l2_reason",
        "limitup_l1l2_buy_signal",
        "limitup_l1l2_buy_recent",
        "limitup_l1l2_ready",
        "limitup_l1l2_limit_date",
        "limitup_l1l2_l1_date",
        "limitup_l1l2_l2_date",
        "limitup_l1l2_buy_date",
        "limitup_l1l2_impulse_pct",
        "limitup_l1l2_pullback_pct",
        "limitup_l1l2_l2_above_l1_pct",
        "limitup_l1l2_confirm_vol_ratio",
        "limitup_l1l2_close_vs_l2_pct",
        "limitup_l1l2_hold_buffer_pct",
        "main_net_amount_3d",
        "main_net_amount_5d",
        "main_net_positive_days_3d",
        "main_net_positive_days_5d",
        "main_net_consecutive_days",
        "volume_ratio",
        "turnover_rate",
        "close_qfq",
        "ma_qfq_20",
        "ma_qfq_60",
    ]


def run_limitup_l1l2_screen(
    end_date: str = "",
    history_bars: int = 100,
    moneyflow_lookback_days: int = 5,
    recent_buy_window: int = 0,
    min_score: float = 55.0,
    cutoff_hour: int = 20,
) -> dict:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    use_custom_http_endpoint = bool(custom_http_url)
    ensure_token(token)
    socket.setdefaulttimeout(int(os.getenv("LIMITUP_L1L2_SOCKET_TIMEOUT", "45")))
    log_step("initializing tushare client")

    pro = ts.pro_api(token)
    pro = configure_tushare_client(pro, token, use_custom_http_endpoint, custom_http_url)

    now_ts = pd.Timestamp.now()
    today_str = now_ts.strftime("%Y%m%d")
    requested_end_date = str(end_date or "").strip() or today_str
    screen_end_date = choose_screen_end_date(now_ts, requested_end_date, today_str, cutoff_hour)
    log_step(f"screen_end_date={screen_end_date} requested_end_date={requested_end_date}")

    trade_dates = get_recent_open_trade_dates(pro, screen_end_date, count=max(moneyflow_lookback_days, 10))
    pattern_trade_dates = get_recent_open_trade_dates(pro, screen_end_date, count=max(history_bars, 80))
    log_step(f"trade_dates={len(trade_dates)} pattern_trade_dates={len(pattern_trade_dates)}")
    latest_trade_date, market_moneyflow_dates, daily_basic_latest, tech_latest, moneyflow_summary = fetch_latest_complete_market_inputs(
        pro,
        trade_dates,
        moneyflow_lookback_days=moneyflow_lookback_days,
        sleep_sec=0.12,
    )
    log_step(f"latest_trade_date={latest_trade_date}")

    stock_basic_all = safe_call(
        "stock_basic_all",
        getattr(pro, "stock_basic", None),
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    if stock_basic_all.empty:
        raise SystemExit("stock_basic returned no data.")
    stock_basic_all = stock_basic_all.fillna("")
    log_step(f"stock_basic_all rows={len(stock_basic_all)}")

    market_snapshot = build_market_snapshot(stock_basic_all, daily_basic_latest, tech_latest, moneyflow_summary)
    log_step(f"market_snapshot rows={len(market_snapshot)}")
    market_daily_history = fetch_market_daily_history(pro, pattern_trade_dates, sleep_sec=0.0)
    log_step(f"market_daily_history rows={len(market_daily_history)}")
    pattern_snapshot = build_limitup_l1l2_snapshot(
        market_daily_history,
        stock_basic_df=stock_basic_all,
        config={
            "recent_buy_window": recent_buy_window,
            "candidate_score_threshold": min_score,
        },
    )

    candidates = stock_basic_all.merge(market_snapshot, on="ts_code", how="left", suffixes=("", "_mkt"))
    if not pattern_snapshot.empty:
        candidates = candidates.merge(pattern_snapshot, on="ts_code", how="left")
    candidates = ensure_columns(
        candidates,
        [
            "limitup_l1l2_stage",
            "limitup_l1l2_signal",
            "limitup_l1l2_buy_signal",
            "limitup_l1l2_buy_recent",
            "limitup_l1l2_ready",
            "limitup_l1l2_score",
            "limitup_l1l2_reason",
            "limitup_l1l2_limit_date",
            "limitup_l1l2_l1_date",
            "limitup_l1l2_l2_date",
            "limitup_l1l2_buy_date",
            "limitup_l1l2_bars_since_buy",
            "limitup_l1l2_impulse_pct",
            "limitup_l1l2_pullback_pct",
            "limitup_l1l2_l2_above_l1_pct",
            "limitup_l1l2_confirm_vol_ratio",
            "limitup_l1l2_close_vs_l2_pct",
            "limitup_l1l2_hold_buffer_pct",
            "limitup_l1l2_trend_ok",
            "limitup_l1l2_volume_ok",
            "limitup_l1l2_limit_sealed",
        ],
    )

    for col in [
        "main_net_amount_3d",
        "main_net_amount_5d",
        "main_net_positive_days_3d",
        "main_net_positive_days_5d",
        "main_net_consecutive_days",
        "volume_ratio",
        "turnover_rate",
        "turnover_rate_f",
        "close_qfq",
        "ma_qfq_20",
        "ma_qfq_60",
        "limitup_l1l2_score",
        "limitup_l1l2_bars_since_buy",
        "limitup_l1l2_impulse_pct",
        "limitup_l1l2_pullback_pct",
        "limitup_l1l2_l2_above_l1_pct",
        "limitup_l1l2_confirm_vol_ratio",
        "limitup_l1l2_close_vs_l2_pct",
        "limitup_l1l2_hold_buffer_pct",
    ]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    for col in [
        "limitup_l1l2_signal",
        "limitup_l1l2_buy_signal",
        "limitup_l1l2_buy_recent",
        "limitup_l1l2_ready",
        "limitup_l1l2_trend_ok",
        "limitup_l1l2_volume_ok",
        "limitup_l1l2_limit_sealed",
    ]:
        if col in candidates.columns:
            candidates[col] = candidates[col].fillna(False)
    for col in [
        "main_net_amount_3d",
        "main_net_amount_5d",
        "main_net_positive_days_3d",
        "main_net_positive_days_5d",
        "main_net_consecutive_days",
        "limitup_l1l2_score",
    ]:
        if col in candidates.columns:
            candidates[col] = candidates[col].fillna(0.0)
    for col in [
        "limitup_l1l2_stage",
        "limitup_l1l2_reason",
        "limitup_l1l2_limit_date",
        "limitup_l1l2_l1_date",
        "limitup_l1l2_l2_date",
        "limitup_l1l2_buy_date",
    ]:
        if col in candidates.columns:
            candidates[col] = candidates[col].fillna("")

    def numeric_series(column: str) -> pd.Series:
        if column in candidates.columns:
            return pd.to_numeric(candidates[column], errors="coerce")
        return pd.Series(np.nan, index=candidates.index, dtype=float)

    candidates["main_net_amount_3d_rank_pct"] = rank_pct(numeric_series("main_net_amount_3d"))
    candidates["main_net_amount_5d_rank_pct"] = rank_pct(numeric_series("main_net_amount_5d"))
    candidates["strategy_name"] = np.where(candidates.get("limitup_l1l2_signal", False), "涨停L1L2", "")
    candidates["strategy_rank_score"] = candidates.apply(lambda row: build_strategy_rank_score(row.to_dict()), axis=1)

    strategy_candidates = candidates[candidates["limitup_l1l2_signal"].fillna(False)].copy()
    strategy_candidates = strategy_candidates.sort_values(
        [
            "strategy_rank_score",
            "limitup_l1l2_buy_signal",
            "limitup_l1l2_buy_recent",
            "limitup_l1l2_score",
            "main_net_amount_3d",
            "volume_ratio",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    best_pick_candidate = strategy_candidates.head(1).copy()

    repo_root = Path(__file__).resolve().parent.parent
    output_root = repo_root / "output" / "jupyter-notebook" / "tushare_limitup_l1l2_exports"
    output_root.mkdir(parents=True, exist_ok=True)
    run_tag = f"limitup_l1l2_pick_{screen_end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = output_root / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)

    pattern_snapshot.to_csv(export_dir / "pattern_snapshot.csv", index=False)
    strategy_candidates.to_csv(export_dir / "strategy_candidates.csv", index=False)
    best_pick_candidate.to_csv(export_dir / "best_pick_candidate.csv", index=False)
    log_step(f"export_dir={export_dir.resolve()}")

    summary = {
        "strategy_name": "涨停L1L2",
        "requested_end_date": requested_end_date,
        "screen_end_date": screen_end_date,
        "latest_trade_date": latest_trade_date,
        "market_moneyflow_dates": market_moneyflow_dates,
        "history_bars": history_bars,
        "recent_buy_window": recent_buy_window,
        "min_score": min_score,
        "pattern_snapshot_stocks": int(len(pattern_snapshot)),
        "strategy_candidates": int(len(strategy_candidates)),
        "buy_signal_today": int(strategy_candidates["limitup_l1l2_buy_signal"].fillna(False).sum()) if not strategy_candidates.empty else 0,
        "buy_signal_recent": int(strategy_candidates["limitup_l1l2_buy_recent"].fillna(False).sum()) if not strategy_candidates.empty else 0,
        "l2_ready_count": int(strategy_candidates["limitup_l1l2_ready"].fillna(False).sum()) if not strategy_candidates.empty else 0,
        "best_pick_ts_code": best_pick_candidate.iloc[0]["ts_code"] if not best_pick_candidate.empty else None,
        "export_dir": str(export_dir.resolve()),
    }
    with (export_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    return {
        "summary": summary,
        "pattern_snapshot": pattern_snapshot,
        "strategy_candidates": strategy_candidates,
        "best_pick_candidate": best_pick_candidate,
        "columns": [c for c in display_columns() if c in strategy_candidates.columns],
        "best_columns": [c for c in display_columns() if c in best_pick_candidate.columns],
        "export_dir": export_dir,
    }


def main() -> None:
    args = parse_args()
    result = run_limitup_l1l2_screen(
        end_date=args.end_date,
        history_bars=args.history_bars,
        moneyflow_lookback_days=args.moneyflow_lookback_days,
        recent_buy_window=args.recent_buy_window,
        min_score=args.min_score,
        cutoff_hour=args.cutoff_hour,
    )

    summary = result["summary"]
    strategy_candidates = result["strategy_candidates"]
    best_pick_candidate = result["best_pick_candidate"]
    columns = result["columns"]
    best_columns = result["best_columns"]

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print("===== strategy candidates =====")
    print(ensure_columns(strategy_candidates, columns)[columns].head(args.show_top).to_string(index=False))
    print("===== best pick =====")
    if best_pick_candidate.empty:
        print("No candidate matched the strategy today.")
    else:
        print(ensure_columns(best_pick_candidate, best_columns)[best_columns].to_string(index=False))
    print(f"export_dir={result['export_dir']}")


if __name__ == "__main__":
    main()
