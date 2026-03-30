from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tushare as ts


def log_step(message: str) -> None:
    print(f"[research] {message}", flush=True)


def repo_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def research_cache_root_dir() -> Path:
    cache_dir = repo_root_dir() / "output" / "cache" / "research_api"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def market_cache_dirs() -> list[Path]:
    base = repo_root_dir() / "output" / "cache"
    dirs = [
        research_cache_root_dir(),
        base / "holder_increase_api",
        base / "double_bottom_api",
        base / "platform_breakout_api",
        base / "limitup_l1l2_api",
    ]
    return [path for path in dirs if path.exists() or path == research_cache_root_dir()]


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


def ensure_columns(df: pd.DataFrame, columns: list[str], fill_value=np.nan) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column not in work.columns:
            work[column] = fill_value
    return work


def cache_file_name(label: str, kwargs: dict) -> str:
    normalized = json.dumps(json_safe(kwargs), ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(f"{label}|{normalized}".encode("utf-8")).hexdigest()[:12]
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    return f"{safe_label}_{digest}.csv"


def load_cached_frame(file_name: str) -> Optional[pd.DataFrame]:
    for cache_dir in market_cache_dirs():
        path = cache_dir / file_name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            log_step(f"cache hit {path.parent.name}/{path.name} rows={len(df)}")
            return df
        except Exception as exc:
            log_step(f"cache read failed {path.name} error={exc}")
    return None


def save_cached_frame(file_name: str, df: pd.DataFrame) -> None:
    path = research_cache_root_dir() / file_name
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        log_step(f"cache write failed {path.name} error={exc}")


def discover_cached_trade_dates(start_date: str, end_date: str) -> list[str]:
    pattern = re.compile(r"^daily_all_(\d{8})_[0-9a-f]{12}\.csv$")
    dates: set[str] = set()
    for cache_dir in market_cache_dirs():
        if not cache_dir.exists():
            continue
        for path in cache_dir.iterdir():
            match = pattern.match(path.name)
            if not match:
                continue
            trade_date = match.group(1)
            if start_date <= trade_date <= end_date:
                dates.add(trade_date)
    return sorted(dates)


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


def configure_tushare_client(token: str, custom_http_url: str = ""):
    socket.setdefaulttimeout(int(os.getenv("RESEARCH_SOCKET_TIMEOUT", "45")))
    ts.set_token(token)
    pro = ts.pro_api(token)
    if custom_http_url:
        configure_proxy_bypass(custom_http_url)
        pro._DataApi__token = token
        pro._DataApi__http_url = custom_http_url
    return pro


def safe_call(label: str, fn, sleep_sec: float = 0.0, retries: int = 2, **kwargs) -> pd.DataFrame:
    if fn is None:
        return pd.DataFrame()
    file_name = cache_file_name(label, kwargs)
    cached = load_cached_frame(file_name)
    if cached is not None:
        return cached

    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            df = fn(**kwargs)
            if sleep_sec:
                time.sleep(sleep_sec)
            if df is None:
                return pd.DataFrame()
            if not df.empty:
                save_cached_frame(file_name, df)
            return df.copy()
        except Exception as exc:
            last_exc = exc
            log_step(f"{label} failed attempt={attempt + 1} error={exc}")
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
    log_step(f"{label} failed permanently error={last_exc}")
    return pd.DataFrame()


def get_open_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    cal = safe_call(
        "trade_cal",
        getattr(pro, "trade_cal", None),
        start_date=start_date,
        end_date=end_date,
        is_open="1",
    )
    if cal.empty:
        cached = discover_cached_trade_dates(start_date, end_date)
        if cached:
            log_step(f"trade dates fallback to cache count={len(cached)}")
            return cached
        return []
    date_col = "cal_date" if "cal_date" in cal.columns else "trade_date"
    dates = sorted(cal[date_col].dropna().astype(str).unique().tolist())
    log_step(f"trade dates from trade_cal count={len(dates)}")
    return dates


def fetch_stock_basic_all(pro) -> pd.DataFrame:
    stock_basic_all = safe_call(
        "stock_basic_all",
        getattr(pro, "stock_basic", None),
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    return stock_basic_all.fillna("") if not stock_basic_all.empty else stock_basic_all


def fetch_market_daily_history(pro, trade_dates: list[str], sleep_sec: float = 0.0, fallback_pro=None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, trade_date in enumerate(trade_dates, start=1):
        log_step(f"daily history {idx}/{len(trade_dates)} trade_date={trade_date}")
        df = safe_call(
            f"daily_all_{trade_date}",
            getattr(pro, "daily", None),
            sleep_sec=sleep_sec,
            retries=1,
            trade_date=trade_date,
        )
        if df.empty and fallback_pro is not None:
            log_step(f"daily history fallback official trade_date={trade_date}")
            df = safe_call(
                f"daily_all_{trade_date}",
                getattr(fallback_pro, "daily", None),
                sleep_sec=sleep_sec,
                retries=1,
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


def load_cached_market_daily_history(trade_dates: list[str]) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        file_name = cache_file_name(f"daily_all_{trade_date}", {"trade_date": trade_date})
        df = load_cached_frame(file_name)
        if df is None or df.empty:
            continue
        keep_cols = [c for c in ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"] if c in df.columns]
        frames.append(df[keep_cols].copy())
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["trade_date"] = combined["trade_date"].astype(str)
    return combined


def build_daily_frame_map(market_daily_history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    history = market_daily_history.copy()
    history["trade_date"] = history["trade_date"].astype(str)
    return {trade_date: sub.copy() for trade_date, sub in history.groupby("trade_date", sort=True)}


def build_forward_return_table(
    market_daily_history: pd.DataFrame,
    hold_days: list[int],
) -> pd.DataFrame:
    if market_daily_history.empty:
        return pd.DataFrame()

    hold_days = sorted({int(day) for day in hold_days if int(day) > 0})
    history = market_daily_history.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    for column in ["open", "high", "low", "close"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    grouped = history.groupby("ts_code", sort=False)
    history["entry_trade_date"] = grouped["trade_date"].shift(-1)
    history["entry_open"] = grouped["open"].shift(-1)
    for day in hold_days:
        close_col = f"close_tplus_{day}"
        high_col = f"high_tplus_{day}"
        low_col = f"low_tplus_{day}"
        history[close_col] = grouped["close"].shift(-day)
        history[high_col] = grouped["high"].shift(-day)
        history[low_col] = grouped["low"].shift(-day)
        history[f"return_open_to_close_{day}d_pct"] = np.where(
            history["entry_open"] > 0,
            (history[close_col] / history["entry_open"] - 1.0) * 100.0,
            np.nan,
        )
        history[f"max_runup_{day}d_pct"] = np.where(
            history["entry_open"] > 0,
            (history[high_col] / history["entry_open"] - 1.0) * 100.0,
            np.nan,
        )
        history[f"max_drawdown_{day}d_pct"] = np.where(
            history["entry_open"] > 0,
            (history[low_col] / history["entry_open"] - 1.0) * 100.0,
            np.nan,
        )

    keep_cols = ["ts_code", "trade_date", "entry_trade_date", "entry_open"]
    keep_cols.extend([col for col in history.columns if col.startswith("return_open_to_close_")])
    keep_cols.extend([col for col in history.columns if col.startswith("max_runup_")])
    keep_cols.extend([col for col in history.columns if col.startswith("max_drawdown_")])
    return history[keep_cols].copy()
