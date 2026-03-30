from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
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


API_ERROR_LOG: dict[str, str] = {}
STRATEGY_NAME = "星曜增持臻选"


@dataclass(frozen=True)
class HolderStrategyConfig:
    ann_start_date: str
    end_date: str
    event_chunk_days: int = 5
    price_lookback_days: int = 250
    moneyflow_lookback_days: int = 5
    cyq_lookback_days: int = 20
    recent_signal_lookback_days: int = 20
    market_data_cutoff_hour: int = 20
    unlock_lookahead_days: int = 30
    max_near_unlock_ratio: float = 3.0
    max_unlock_ratio_30d: float = 8.0
    active_reduction_min_ratio: float = 0.3
    api_sleep_sec: float = 0.12
    cyq_sleep_sec: float = 0.12
    allowed_holder_types: tuple[str, ...] = ("C", "G")
    min_event_count: int = 1
    min_total_change_ratio: float = 0.0
    min_volume_ratio: float = 1.2
    max_price_position: float = 0.45
    max_industry_pb_pct: float = 0.70
    max_deep_dive_stocks: int = 80
    top_n_stage1: int = 10
    max_stage2_cyq_stocks: int = 10
    stage2_cyq_budget: int = 10
    top_n_final: int = 5
    top_n_aggressive: int = 3
    min_final_score: float = 60.0
    min_aggressive_score: float = 52.0
    enable_forecast: bool = True
    enable_stage2_cyq: bool = True

    @classmethod
    def for_end_date(
        cls,
        end_date: str,
        ann_start_date: str = "",
        **overrides: Any,
    ) -> "HolderStrategyConfig":
        end_str = normalize_trade_day(end_date)
        ann_str = normalize_trade_day(ann_start_date) if ann_start_date else ""
        if not ann_str:
            ann_str = (pd.Timestamp(end_str) - pd.Timedelta(days=45)).strftime("%Y%m%d")
        return cls(ann_start_date=ann_str, end_date=end_str, **overrides)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HolderStrategyConfig":
        return cls(
            ann_start_date=normalize_trade_day(data.get("ann_start_date", "")),
            end_date=normalize_trade_day(data.get("end_date", "")),
            event_chunk_days=int(data.get("event_chunk_days", 5)),
            price_lookback_days=int(data.get("price_lookback_days", 250)),
            moneyflow_lookback_days=int(data.get("moneyflow_lookback_days", 5)),
            cyq_lookback_days=int(data.get("cyq_lookback_days", 20)),
            recent_signal_lookback_days=int(data.get("recent_signal_lookback_days", 20)),
            market_data_cutoff_hour=int(data.get("market_data_cutoff_hour", 20)),
            unlock_lookahead_days=int(data.get("unlock_lookahead_days", 30)),
            max_near_unlock_ratio=float(data.get("max_near_unlock_ratio", 3.0)),
            max_unlock_ratio_30d=float(data.get("max_unlock_ratio_30d", 8.0)),
            active_reduction_min_ratio=float(data.get("active_reduction_min_ratio", 0.3)),
            api_sleep_sec=float(data.get("api_sleep_sec", 0.12)),
            cyq_sleep_sec=float(data.get("cyq_sleep_sec", 0.12)),
            allowed_holder_types=tuple(str(value).strip().upper() for value in data.get("allowed_holder_types", ["C", "G"]) if str(value).strip()),
            min_event_count=int(data.get("min_event_count", 1)),
            min_total_change_ratio=float(data.get("min_total_change_ratio", 0.0)),
            min_volume_ratio=float(data.get("min_volume_ratio", 1.2)),
            max_price_position=float(data.get("max_price_position", 0.45)),
            max_industry_pb_pct=float(data.get("max_industry_pb_pct", 0.70)),
            max_deep_dive_stocks=int(data.get("max_deep_dive_stocks", 80)),
            top_n_stage1=int(data.get("top_n_stage1", 10)),
            max_stage2_cyq_stocks=int(data.get("max_stage2_cyq_stocks", 10)),
            stage2_cyq_budget=int(data.get("stage2_cyq_budget", 10)),
            top_n_final=int(data.get("top_n_final", 5)),
            top_n_aggressive=int(data.get("top_n_aggressive", 3)),
            min_final_score=float(data.get("min_final_score", 60.0)),
            min_aggressive_score=float(data.get("min_aggressive_score", 52.0)),
            enable_forecast=to_bool(data.get("enable_forecast", True)),
            enable_stage2_cyq=to_bool(data.get("enable_stage2_cyq", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_step(message: str) -> None:
    print(f"[holder] {message}", flush=True)


def repo_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def cache_enabled() -> bool:
    return os.getenv("HOLDER_STRATEGY_USE_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def cache_root_dir() -> Path:
    cache_dir = repo_root_dir() / "output" / "cache" / "holder_increase_api"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def output_root_dir() -> Path:
    path = repo_root_dir() / "output" / "jupyter-notebook" / "tushare_screen_exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv_checkpoint(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def write_json_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def dedupe_stage_rows(df: pd.DataFrame, key_col: str = "ts_code") -> pd.DataFrame:
    if df.empty or key_col not in df.columns:
        return df.copy()
    work = df.copy()
    work[key_col] = work[key_col].astype(str)
    return work.drop_duplicates(subset=[key_col], keep="last").reset_index(drop=True)


def upsert_stage_row(existing_df: pd.DataFrame, row: dict[str, Any], key_col: str = "ts_code") -> pd.DataFrame:
    row_df = pd.DataFrame([row])
    if existing_df.empty:
        return dedupe_stage_rows(row_df, key_col=key_col)

    work = existing_df.copy()
    for column in row_df.columns:
        if column not in work.columns:
            work[column] = np.nan
    for column in work.columns:
        if column not in row_df.columns:
            row_df[column] = np.nan
    combined = pd.concat([work, row_df[work.columns]], ignore_index=True)
    return dedupe_stage_rows(combined, key_col=key_col)


def completed_stage_codes(
    df: pd.DataFrame,
    target_codes: list[str],
    *,
    complete_col: str,
    key_col: str = "ts_code",
) -> set[str]:
    if df.empty or key_col not in df.columns:
        return set()
    work = dedupe_stage_rows(df, key_col=key_col)
    work[key_col] = work[key_col].astype(str)
    work = work[work[key_col].isin(target_codes)].copy()
    if work.empty:
        return set()
    if complete_col not in work.columns:
        return set(work[key_col].tolist())
    completed_mask = work[complete_col].fillna(False).astype(bool)
    return set(work.loc[completed_mask, key_col].tolist())


def build_stage_progress_payload(
    *,
    config: "HolderStrategyConfig",
    status: str,
    trade_date: str,
    deep_dive_targets: list[str],
    deep_metrics_stage1: pd.DataFrame,
    stage2_targets: list[str],
    stage2_cyq_metrics: pd.DataFrame,
    error: str = "",
) -> dict[str, Any]:
    stage1_done = sorted(completed_stage_codes(deep_metrics_stage1, deep_dive_targets, complete_col="stage1_complete"))
    stage2_done = sorted(completed_stage_codes(stage2_cyq_metrics, stage2_targets, complete_col="stage2_complete"))
    return {
        "status": status,
        "trade_date": trade_date,
        "config_snapshot": json_safe(config.to_dict()),
        "deep_dive_target_count": len(deep_dive_targets),
        "deep_dive_completed_count": len(stage1_done),
        "deep_dive_pending_codes": [code for code in deep_dive_targets if code not in set(stage1_done)],
        "stage2_target_count": len(stage2_targets),
        "stage2_completed_count": len(stage2_done),
        "stage2_pending_codes": [code for code in stage2_targets if code not in set(stage2_done)],
        "updated_at": pd.Timestamp.now().isoformat(),
        "error": error or None,
    }


def normalize_trade_day(value: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        raw = str(int(value))
    elif isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return ""
        raw = str(int(value)) if float(value).is_integer() else str(value)
    else:
        raw = str(value or "").strip()
    if raw.endswith(".0") and raw[:-2].isdigit():
        raw = raw[:-2]
    raw = raw.replace("-", "")
    if not raw:
        return ""
    if raw.isdigit() and len(raw) == 8:
        return raw
    try:
        ts = pd.Timestamp(raw)
    except Exception:
        return ""
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y%m%d")


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


def configure_tushare_client(token: str, custom_http_url: str = ""):
    socket.setdefaulttimeout(int(os.getenv("HOLDER_STRATEGY_SOCKET_TIMEOUT", "45")))
    ts.set_token(token)
    pro = ts.pro_api(token)
    if custom_http_url:
        configure_proxy_bypass(custom_http_url)
        pro._DataApi__token = token
        pro._DataApi__http_url = custom_http_url
    return pro


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


def cache_file_name(label: str, kwargs: dict) -> str:
    normalized = json.dumps(json_safe(kwargs), ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(f"{label}|{normalized}".encode("utf-8")).hexdigest()[:12]
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    return f"{safe_label}_{digest}.csv"


def load_cached_frame(file_name: str) -> Optional[pd.DataFrame]:
    path = cache_root_dir() / file_name
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        log_step(f"cache hit {path.name} rows={len(df)}")
        return df
    except Exception as exc:
        log_step(f"cache read failed {path.name} error={exc}")
        return None


def save_cached_frame(file_name: str, df: pd.DataFrame) -> None:
    path = cache_root_dir() / file_name
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        log_step(f"cache write failed {path.name} error={exc}")


def is_transient_api_error(message: Optional[str]) -> bool:
    if not message:
        return False
    text = str(message)
    transient_tokens = [
        "Read timed out",
        "RemoteDisconnected",
        "Connection aborted",
        "timed out",
        "temporarily unavailable",
        "Bad Gateway",
        "502",
        "503",
        "504",
    ]
    return any(token in text for token in transient_tokens)


def retry_sleep_seconds(message: Optional[str], attempt: int) -> float:
    attempt_index = max(1, attempt + 1)
    base_sleep = 0.8 * attempt_index
    if is_rate_limit_error(message):
        rate_limit_sleep = float(os.getenv("HOLDER_STRATEGY_RATE_LIMIT_SLEEP_SEC", "15"))
        return max(base_sleep, rate_limit_sleep * attempt_index)
    if is_transient_api_error(message):
        transient_sleep = float(os.getenv("HOLDER_STRATEGY_TRANSIENT_RETRY_SLEEP_SEC", "3"))
        return max(base_sleep, transient_sleep * attempt_index)
    return base_sleep


def safe_call(label: str, fn, sleep_sec: float = 0.0, retries: int = 2, **kwargs) -> pd.DataFrame:
    if fn is None:
        return pd.DataFrame()
    file_name = cache_file_name(label, kwargs)
    if cache_enabled():
        cached = load_cached_frame(file_name)
        if cached is not None:
            return cached
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            df = fn(**kwargs)
            API_ERROR_LOG.pop(label, None)
            if sleep_sec:
                time.sleep(sleep_sec)
            if df is None:
                return pd.DataFrame()
            if cache_enabled() and not df.empty:
                save_cached_frame(file_name, df)
            return df.copy()
        except Exception as exc:
            last_exc = exc
            API_ERROR_LOG[label] = str(exc)
            log_step(f"{label} failed attempt={attempt + 1} error={exc}")
            if attempt < retries:
                time.sleep(retry_sleep_seconds(str(exc), attempt))
    log_step(f"{label} failed permanently error={last_exc}")
    return pd.DataFrame()


def sort_desc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    for col in ["trade_date", "ann_date", "f_ann_date", "end_date", "float_date"]:
        if col in df.columns:
            ordered = df.copy()
            ordered[col] = ordered[col].astype(str)
            return ordered.sort_values(col, ascending=False).reset_index(drop=True)
    return df.reset_index(drop=True).copy()


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


def to_number(value: Any, digits: int = 2):
    numeric = to_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


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


def latest_row(df: pd.DataFrame) -> dict[str, Any]:
    ordered = sort_desc(df)
    if ordered.empty:
        return {}
    row = ordered.iloc[0]
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


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


def choose_screen_end_date(now_ts: pd.Timestamp, end_date: str, today_str: str, cutoff_hour: int = 20) -> str:
    requested = end_date or today_str
    if requested == today_str and int(now_ts.hour) < int(cutoff_hour):
        return (now_ts.normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    return requested


def filter_frame_as_of(df: pd.DataFrame, as_of_date: str, preferred_date_cols: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    as_of = normalize_trade_day(as_of_date)
    work = df.copy()
    for col in preferred_date_cols:
        if col not in work.columns:
            continue
        values = work[col].fillna("").astype(str).str.replace("-", "", regex=False)
        mask = values.str.fullmatch(r"\d{8}") & (values <= as_of)
        filtered = work[mask].copy()
        if filtered.empty:
            return work.iloc[0:0].copy()
        filtered[col] = values[mask]
        return sort_desc(filtered)
    return sort_desc(work)


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


def fetch_stock_basic_all(pro) -> pd.DataFrame:
    stock_basic_all = safe_call(
        "stock_basic_all",
        getattr(pro, "stock_basic", None),
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    return stock_basic_all.fillna("") if not stock_basic_all.empty else stock_basic_all


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
        if df.empty:
            continue
        work = df.copy()
        work["trade_date"] = trade_date
        work["main_net_amount"] = compute_main_net_amount(work)
        frames.append(work)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["main_net_amount"] = pd.to_numeric(combined["main_net_amount"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
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


def build_market_technical_snapshot_from_cached_history(trade_date: str) -> pd.DataFrame:
    try:
        from research_backtest_utils import discover_cached_trade_dates, load_cached_market_daily_history
    except Exception as exc:
        log_step(f"cached tech fallback import failed error={exc}")
        return pd.DataFrame()

    start_date = (pd.Timestamp(trade_date) - pd.Timedelta(days=520)).strftime("%Y%m%d")
    cached_dates = discover_cached_trade_dates(start_date, trade_date)
    if not cached_dates or trade_date not in cached_dates:
        return pd.DataFrame()
    history = load_cached_market_daily_history(cached_dates)
    if history.empty:
        return pd.DataFrame()

    work = history.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work = work[work["trade_date"] <= trade_date].copy()
    if work.empty:
        return pd.DataFrame()

    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["vol"] = pd.to_numeric(work.get("vol"), errors="coerce")
    work = work.dropna(subset=["ts_code", "trade_date", "close"]).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby("ts_code", sort=False)
    for window, column in [
        (5, "ma_qfq_5"),
        (10, "ma_qfq_10"),
        (20, "ma_qfq_20"),
        (60, "ma_qfq_60"),
        (250, "ma_qfq_250"),
    ]:
        work[column] = grouped["close"].rolling(window=window, min_periods=min(window, 3)).mean().reset_index(level=0, drop=True)

    ema12 = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    dif = ema12 - ema26
    dea = dif.groupby(work["ts_code"], sort=False).transform(lambda s: s.ewm(span=9, adjust=False).mean())
    work["close_qfq"] = work["close"]
    work["macd_dif_qfq"] = dif
    work["macd_dea_qfq"] = dea

    prev5_vol = grouped["vol"].transform(lambda s: s.shift(1).rolling(window=5, min_periods=3).mean())
    work["vol_ratio"] = np.where(prev5_vol > 0, work["vol"] / prev5_vol, np.nan)
    snapshot = work[work["trade_date"] == trade_date].copy()
    if snapshot.empty:
        return pd.DataFrame()
    keep_cols = [
        column
        for column in [
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
        if column in snapshot.columns
    ]
    log_step(f"cached tech fallback trade_date={trade_date} rows={len(snapshot)}")
    return snapshot[keep_cols].reset_index(drop=True)


def fetch_latest_complete_market_inputs(
    pro,
    trade_dates: list[str],
    moneyflow_lookback_days: int,
    sleep_sec: float = 0.0,
) -> tuple[str, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not trade_dates:
        return "", [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
        if tech_df.empty:
            tech_df = build_market_technical_snapshot_from_cached_history(trade_date)
        moneyflow_df = fetch_recent_moneyflow_summary(pro, moneyflow_dates, sleep_sec=sleep_sec)

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
    ann_start_date: str,
    end_date: str,
) -> pd.DataFrame:
    work = filter_frame_as_of(holdertrade_df, end_date, ("ann_date",))
    if work.empty:
        return pd.DataFrame()
    if "in_de" in work.columns:
        work = work[work["in_de"].fillna("").astype(str).str.upper() == "IN"].copy()
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
    grouped_rows: list[dict[str, Any]] = []
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
        1.0
        - pd.to_numeric(grouped["days_since_latest_ann"], errors="coerce").fillna(999).clip(lower=0)
        / max(1, (pd.Timestamp(end_date) - pd.Timestamp(ann_start_date)).days + 1)
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
    work = filter_frame_as_of(holdertrade_df, end_date, ("ann_date",))
    if work.empty:
        return pd.DataFrame()
    work["ann_date"] = work["ann_date"].astype(str)
    work["ann_date_dt"] = pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce")
    work = work.dropna(subset=["ann_date_dt"]).copy()
    work["holder_type"] = work.get("holder_type", "").fillna("").astype(str) if "holder_type" in work.columns else ""
    work["holder_name"] = work.get("holder_name", "").fillna("").astype(str) if "holder_name" in work.columns else ""
    work["in_de"] = work.get("in_de", "").fillna("").astype(str).str.upper() if "in_de" in work.columns else ""
    work["change_ratio"] = pd.to_numeric(work.get("change_ratio", 0.0), errors="coerce").fillna(0.0)

    cutoff = pd.Timestamp(end_date) - pd.Timedelta(days=max(1, lookback_days - 1))
    core_holder_types = core_holder_types or set()
    rows: list[dict[str, Any]] = []
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
    work = filter_frame_as_of(holdertrade_df, end_date, ("ann_date",))
    if work.empty:
        return pd.DataFrame()
    work["ann_date"] = work["ann_date"].astype(str)
    work["ann_date_dt"] = pd.to_datetime(work["ann_date"], format="%Y%m%d", errors="coerce")
    work = work.dropna(subset=["ann_date_dt"]).copy()
    work["holder_type"] = work.get("holder_type", "").fillna("").astype(str) if "holder_type" in work.columns else ""
    work["holder_name"] = work.get("holder_name", "").fillna("").astype(str) if "holder_name" in work.columns else ""
    work["in_de"] = work.get("in_de", "").fillna("").astype(str).str.upper() if "in_de" in work.columns else ""
    work["change_ratio"] = pd.to_numeric(work.get("change_ratio", 0.0), errors="coerce").fillna(0.0)
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
    rows: list[dict[str, Any]] = []
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
    if share_float_df.empty or "float_date" not in share_float_df.columns:
        return pd.DataFrame()
    work = share_float_df.copy()
    work["float_date"] = work["float_date"].astype(str)
    work["float_date_dt"] = pd.to_datetime(work["float_date"], format="%Y%m%d", errors="coerce")
    work = work.dropna(subset=["float_date_dt"]).copy()
    work["float_ratio"] = pd.to_numeric(work.get("float_ratio", 0.0), errors="coerce").fillna(0.0)

    screen_end_ts = pd.Timestamp(end_date)
    future_cutoff = screen_end_ts + pd.Timedelta(days=max(1, lookahead_days))
    future = work[(work["float_date_dt"] >= screen_end_ts) & (work["float_date_dt"] <= future_cutoff)].copy()
    if future.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ts_code, sub in future.groupby("ts_code", dropna=False):
        ordered = sub.sort_values(["float_date_dt", "float_ratio"], ascending=[True, False]).reset_index(drop=True)
        nearest = ordered.iloc[0]
        nearest_ratio = pd.to_numeric(ordered[ordered["float_date_dt"] == nearest["float_date_dt"]]["float_ratio"], errors="coerce").fillna(0.0).sum()
        total_ratio = pd.to_numeric(ordered["float_ratio"], errors="coerce").fillna(0.0).sum()
        days_to_unlock = int((nearest["float_date_dt"].normalize() - screen_end_ts.normalize()).days)
        unlock_veto = bool((days_to_unlock <= 10 and nearest_ratio >= max_near_unlock_ratio) or total_ratio >= max_unlock_ratio_30d)
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


def build_market_snapshot(
    stock_basic_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    tech_df: pd.DataFrame,
    moneyflow_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    basic_cols = [c for c in ["ts_code", "symbol", "name", "industry", "market", "list_date"] if c in stock_basic_df.columns]
    snapshot = stock_basic_df[basic_cols].drop_duplicates(subset=["ts_code"]).copy()

    if not daily_basic_df.empty:
        daily_cols = [c for c in ["ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv"] if c in daily_basic_df.columns]
        snapshot = snapshot.merge(daily_basic_df[daily_cols].drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")

    if not tech_df.empty:
        tech_cols = [c for c in ["ts_code", "trade_date", "close_qfq", "ma_qfq_5", "ma_qfq_10", "ma_qfq_20", "ma_qfq_60", "ma_qfq_250", "macd_dif_qfq", "macd_dea_qfq", "vol_ratio"] if c in tech_df.columns]
        tech_prepared = tech_df[tech_cols].drop_duplicates(subset=["ts_code"]).copy()
        snapshot = snapshot.merge(tech_prepared, on="ts_code", how="left", suffixes=("", "_tech"))

    if not moneyflow_summary_df.empty:
        snapshot = snapshot.merge(moneyflow_summary_df, on="ts_code", how="left")

    snapshot["pb_num"] = pd.to_numeric(snapshot.get("pb"), errors="coerce")
    if "industry" in snapshot.columns:
        snapshot["industry_pb_pct_rank"] = snapshot.groupby("industry")["pb_num"].rank(pct=True, method="average").fillna(0.5)
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


def build_market_regime_snapshot(snapshot: pd.DataFrame) -> dict[str, Any]:
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
        trend_mask = pd.to_numeric(work["close_qfq"], errors="coerce") > pd.to_numeric(work["ma_qfq_20"], errors="coerce")
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


def summarize_price_metrics(daily_df: pd.DataFrame, adj_df: pd.DataFrame, window: int = 250) -> dict[str, Any]:
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
    position = None if high_250 == low_250 else (latest_close - low_250) / (high_250 - low_250)

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


def summarize_indicator_metrics(indicator_df: pd.DataFrame, end_date: str) -> dict[str, Any]:
    latest = latest_row(filter_frame_as_of(indicator_df, end_date, ("ann_date", "f_ann_date")))
    return {
        "report_period": latest.get("end_date"),
        "indicator_ann_date": latest.get("ann_date") or latest.get("f_ann_date"),
        "roe": to_number(latest.get("roe")),
        "gross_margin": to_number(latest.get("gross_margin") or latest.get("grossprofit_margin")),
        "dt_netprofit_yoy": to_number(latest.get("dt_netprofit_yoy")),
        "netprofit_yoy": to_number(latest.get("netprofit_yoy")),
        "ocf_yoy": to_number(latest.get("ocf_yoy")),
        "q_sales_yoy": to_number(latest.get("q_sales_yoy") or latest.get("tr_yoy") or latest.get("or_yoy")),
        "debt_to_assets": to_number(latest.get("debt_to_assets")),
    }


def summarize_forecast_metrics(forecast_df: pd.DataFrame, end_date: str) -> dict[str, Any]:
    latest = latest_row(filter_frame_as_of(forecast_df, end_date, ("ann_date",)))
    forecast_type = str(latest.get("type") or "")
    negative = bool(forecast_type) and (("亏" in forecast_type) or (forecast_type in {"预减", "略减", "续亏", "首亏"}))
    return {
        "forecast_ann_date": latest.get("ann_date"),
        "forecast_type": forecast_type or None,
        "forecast_negative": negative,
        "forecast_p_change_min": to_number(latest.get("p_change_min")),
        "forecast_p_change_max": to_number(latest.get("p_change_max")),
        "forecast_summary": latest.get("summary"),
    }


def summarize_cyq_metrics(cyq_df: pd.DataFrame, latest_close: Optional[float] = None) -> dict[str, Any]:
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


def fetch_single_stock_history_bundle(
    pro,
    ts_code: str,
    end_date: str,
    price_lookback_days: int,
    cyq_lookback_days: int,
    sleep_sec: float = 0.0,
    cyq_sleep_sec: float = 0.0,
    enable_forecast: bool = True,
    enable_cyq: bool = True,
) -> dict[str, Any]:
    price_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(420, price_lookback_days * 2))).strftime("%Y%m%d")
    cyq_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(60, cyq_lookback_days * 3))).strftime("%Y%m%d")
    daily_label = f"daily_{ts_code}"
    daily_df = sort_desc(
        safe_call(
            daily_label,
            getattr(pro, "daily", None),
            sleep_sec=sleep_sec,
            ts_code=ts_code,
            start_date=price_start_date,
            end_date=end_date,
        )
    )
    adj_label = f"adj_factor_{ts_code}"
    adj_df = sort_desc(
        safe_call(
            adj_label,
            getattr(pro, "adj_factor", None),
            sleep_sec=sleep_sec,
            ts_code=ts_code,
            start_date=price_start_date,
            end_date=end_date,
        )
    )
    indicator_label = f"fina_indicator_{ts_code}"
    indicator_df = sort_desc(
        safe_call(
            indicator_label,
            getattr(pro, "fina_indicator", None),
            sleep_sec=sleep_sec,
            ts_code=ts_code,
        )
    )
    forecast_label = f"forecast_{ts_code}"
    forecast_df = sort_desc(
        safe_call(
            forecast_label,
            getattr(pro, "forecast", None) if enable_forecast else None,
            sleep_sec=sleep_sec,
            ts_code=ts_code,
        )
    )
    cyq_df = pd.DataFrame()
    cyq_label = f"cyq_perf_{ts_code}"
    if enable_cyq:
        cyq_df = sort_desc(
            safe_call(
                cyq_label,
                getattr(pro, "cyq_perf", None),
                sleep_sec=cyq_sleep_sec,
                ts_code=ts_code,
                start_date=cyq_start_date,
                end_date=end_date,
            )
        )
    return {
        "ts_code": ts_code,
        "daily_df": daily_df,
        "adj_df": adj_df,
        "indicator_df": indicator_df,
        "forecast_df": forecast_df,
        "cyq_df": cyq_df,
        "daily_error": API_ERROR_LOG.get(daily_label),
        "adj_error": API_ERROR_LOG.get(adj_label),
        "indicator_error": API_ERROR_LOG.get(indicator_label),
        "forecast_error": API_ERROR_LOG.get(forecast_label),
        "cyq_error": API_ERROR_LOG.get(cyq_label),
    }


def build_stock_deep_metrics_from_bundle(
    bundle: dict[str, Any],
    end_date: str,
    enable_forecast: bool = True,
) -> dict[str, Any]:
    ts_code = str(bundle.get("ts_code") or "")
    daily_df = filter_frame_as_of(bundle.get("daily_df", pd.DataFrame()), end_date, ("trade_date",))
    adj_df = filter_frame_as_of(bundle.get("adj_df", pd.DataFrame()), end_date, ("trade_date",))
    indicator_df = bundle.get("indicator_df", pd.DataFrame())
    forecast_df = bundle.get("forecast_df", pd.DataFrame()) if enable_forecast else pd.DataFrame()
    cyq_df = filter_frame_as_of(bundle.get("cyq_df", pd.DataFrame()), end_date, ("trade_date",))

    price_metrics = summarize_price_metrics(daily_df, adj_df, window=250)
    indicator_metrics = summarize_indicator_metrics(indicator_df, end_date)
    forecast_metrics = summarize_forecast_metrics(forecast_df, end_date)
    cyq_metrics = summarize_cyq_metrics(cyq_df, latest_close=to_float(price_metrics.get("latest_close_qfq_calc")))
    return {
        "ts_code": ts_code,
        **price_metrics,
        **indicator_metrics,
        **forecast_metrics,
        **cyq_metrics,
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
) -> dict[str, Any]:
    bundle = fetch_single_stock_history_bundle(
        pro,
        ts_code=ts_code,
        end_date=end_date,
        price_lookback_days=price_lookback_days,
        cyq_lookback_days=cyq_lookback_days,
        sleep_sec=sleep_sec,
        cyq_sleep_sec=cyq_sleep_sec,
        enable_forecast=enable_forecast,
        enable_cyq=enable_cyq,
    )
    metrics = build_stock_deep_metrics_from_bundle(bundle, end_date=end_date, enable_forecast=enable_forecast)
    cyq_checked = not filter_frame_as_of(bundle.get("cyq_df", pd.DataFrame()), end_date, ("trade_date",)).empty
    daily_error = bundle.get("daily_error")
    adj_error = bundle.get("adj_error")
    indicator_error = bundle.get("indicator_error")
    forecast_error = bundle.get("forecast_error") if enable_forecast else None
    stage1_errors = [daily_error, adj_error, indicator_error, forecast_error]
    stage1_retry_needed = any(is_transient_api_error(err) or is_rate_limit_error(err) for err in stage1_errors if err)
    return {
        **metrics,
        "cyq_checked": cyq_checked,
        "cyq_rate_limited": is_rate_limit_error(bundle.get("cyq_error")),
        "cyq_error": bundle.get("cyq_error"),
        "daily_error": daily_error,
        "adj_error": adj_error,
        "indicator_error": indicator_error,
        "forecast_error": forecast_error,
        "stage1_complete": not stage1_retry_needed,
    }


def fetch_single_stock_cyq_metrics(
    pro,
    ts_code: str,
    end_date: str,
    cyq_lookback_days: int,
    sleep_sec: float = 0.0,
    latest_close: Optional[float] = None,
) -> dict[str, Any]:
    bundle = fetch_single_stock_history_bundle(
        pro,
        ts_code=ts_code,
        end_date=end_date,
        price_lookback_days=250,
        cyq_lookback_days=cyq_lookback_days,
        sleep_sec=0.0,
        cyq_sleep_sec=sleep_sec,
        enable_forecast=False,
        enable_cyq=True,
    )
    cyq_df = filter_frame_as_of(bundle.get("cyq_df", pd.DataFrame()), end_date, ("trade_date",))
    cyq_error = bundle.get("cyq_error")
    stage2_retry_needed = bool(cyq_error) and (is_transient_api_error(cyq_error) or is_rate_limit_error(cyq_error))
    return {
        "ts_code": ts_code,
        "cyq_checked": not cyq_df.empty,
        "cyq_rate_limited": is_rate_limit_error(cyq_error),
        "cyq_error": cyq_error,
        "stage2_complete": not stage2_retry_needed,
        **summarize_cyq_metrics(cyq_df, latest_close=latest_close),
    }


def build_earnings_score(row: dict[str, Any]) -> float:
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


def build_value_score(row: dict[str, Any]) -> float:
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


def build_reversal_score(row: dict[str, Any], min_volume_ratio: float = 1.2) -> float:
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


def build_chip_score(row: dict[str, Any]) -> float:
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


def build_fund_flow_score(row: dict[str, Any]) -> float:
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


def build_event_bonus_score(row: dict[str, Any]) -> float:
    event_score = to_float(row.get("event_score"))
    if event_score is None:
        return 0.0
    latest_change_dir = str(row.get("latest_change_dir") or "").upper()
    bonus = event_score * 0.25
    if latest_change_dir == "IN":
        bonus += 1.0
    return round(clip_score(bonus, 0.0, 8.0), 2)


def build_overheat_penalty_score(row: dict[str, Any]) -> float:
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


def build_risk_penalty_score(row: dict[str, Any]) -> float:
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


def build_stable_score(row: dict[str, Any], market_regime: str = "neutral") -> float:
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


def build_aggressive_score(row: dict[str, Any], market_regime: str = "neutral") -> float:
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


def build_preliminary_score(row: dict[str, Any]) -> float:
    score = (
        (to_float(row.get("fund_flow_score")) or 0.0) * 1.05
        + (to_float(row.get("reversal_score")) or 0.0) * 0.95
        + (to_float(row.get("event_bonus_score")) or 0.0) * 0.15
        - (to_float(row.get("risk_penalty_score")) or 0.0) * 0.80
        - (to_float(row.get("overheat_penalty_score")) or 0.0) * 0.35
    )
    return round(clip_score(score, 0.0, 100.0), 2)


def build_candidate_flags(
    row: dict[str, Any],
    max_price_position: float,
    max_industry_pb_pct: float,
    min_volume_ratio: float,
    min_final_score: float,
    min_aggressive_score: float,
) -> dict[str, Any]:
    price_position = to_float(row.get("price_position_250"))
    industry_pb_pct = to_float(row.get("industry_pb_pct_rank"))
    recent_decrease_ratio = to_float(row.get("recent_decrease_ratio"))
    overheat_penalty = to_float(row.get("overheat_penalty_score")) or 0.0
    reduction_plan_veto = to_bool(row.get("active_reduction_plan_flag"))
    unlock_risk_veto = to_bool(row.get("unlock_risk_veto"))

    earnings_ok = build_earnings_score(row) >= 18 and not to_bool(row.get("forecast_negative"))
    value_ok = price_position is not None and price_position <= max_price_position and industry_pb_pct is not None and industry_pb_pct <= max_industry_pb_pct
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
    work["reversal_score"] = work.apply(lambda row: build_reversal_score(row.to_dict(), min_volume_ratio), axis=1)
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
        pd.to_numeric(work["aggressive_score"], errors="coerce").fillna(0.0) > pd.to_numeric(work["stable_score"], errors="coerce").fillna(0.0),
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


def build_holder_candidate_base(
    config: HolderStrategyConfig,
    stock_basic_all: pd.DataFrame,
    holdertrade_raw: pd.DataFrame,
    share_float_schedule: pd.DataFrame,
    latest_trade_date: str,
    market_moneyflow_dates: list[str],
    daily_basic_latest: pd.DataFrame,
    tech_latest: pd.DataFrame,
    moneyflow_summary: pd.DataFrame,
) -> dict[str, Any]:
    allowed_holder_type_set = {value.strip().upper() for value in config.allowed_holder_types if value and value.strip()}
    event_pool = prepare_event_pool(holdertrade_raw, stock_basic_all, allowed_holder_type_set, config.ann_start_date, config.end_date)
    if not event_pool.empty:
        event_pool = event_pool[
            (event_pool["event_count"] >= config.min_event_count)
            & (pd.to_numeric(event_pool["total_change_ratio"], errors="coerce").fillna(0.0) >= config.min_total_change_ratio)
        ].reset_index(drop=True)
    reverse_signal_snapshot = build_reverse_signal_snapshot(
        holdertrade_raw,
        config.end_date,
        lookback_days=config.recent_signal_lookback_days,
        core_holder_types=allowed_holder_type_set,
    )
    reduction_plan_snapshot = build_reduction_plan_snapshot(
        holdertrade_raw,
        config.end_date,
        lookback_days=max(45, config.recent_signal_lookback_days * 2),
        core_holder_types=allowed_holder_type_set,
        min_ratio=config.active_reduction_min_ratio,
    )
    unlock_snapshot = build_unlock_snapshot(
        share_float_schedule,
        config.end_date,
        lookahead_days=config.unlock_lookahead_days,
        max_near_unlock_ratio=config.max_near_unlock_ratio,
        max_unlock_ratio_30d=config.max_unlock_ratio_30d,
    )
    market_snapshot = build_market_snapshot(stock_basic_all, daily_basic_latest, tech_latest, moneyflow_summary)
    market_regime_snapshot = build_market_regime_snapshot(market_snapshot)
    market_regime = market_regime_snapshot["market_regime"]

    candidate_base = stock_basic_all.merge(market_snapshot, on="ts_code", how="left", suffixes=("", "_mkt"))
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
    for col in ["mixed_signal_flag", "active_reduction_plan_flag", "unlock_risk_veto"]:
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
        min_volume_ratio=config.min_volume_ratio,
        max_price_position=config.max_price_position,
        max_industry_pb_pct=config.max_industry_pb_pct,
        min_final_score=config.min_final_score,
        min_aggressive_score=config.min_aggressive_score,
        market_regime=market_regime,
    )
    return {
        "latest_trade_date": latest_trade_date,
        "market_moneyflow_dates": market_moneyflow_dates,
        "event_pool": event_pool,
        "reverse_signal_snapshot": reverse_signal_snapshot,
        "reduction_plan_snapshot": reduction_plan_snapshot,
        "unlock_snapshot": unlock_snapshot,
        "market_snapshot": market_snapshot,
        "market_regime_snapshot": market_regime_snapshot,
        "market_regime": market_regime,
        "candidate_base": candidate_base,
    }


def select_stage1_targets(candidate_base: pd.DataFrame, config: HolderStrategyConfig) -> list[str]:
    if candidate_base.empty:
        return []
    ranked = candidate_base.sort_values(
        ["preliminary_score", "fund_flow_score", "reversal_score", "event_bonus_score"],
        ascending=[False, False, False, False],
    )
    return ranked.head(config.max_deep_dive_stocks)["ts_code"].dropna().astype(str).tolist()


def apply_holder_stage1(
    candidate_base: pd.DataFrame,
    deep_metrics_stage1: pd.DataFrame,
    config: HolderStrategyConfig,
    market_regime: str,
) -> dict[str, Any]:
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
        min_volume_ratio=config.min_volume_ratio,
        max_price_position=config.max_price_position,
        max_industry_pb_pct=config.max_industry_pb_pct,
        min_final_score=config.min_final_score,
        min_aggressive_score=config.min_aggressive_score,
        market_regime=market_regime,
    )
    stable_focus_stage1 = screened_stage1[screened_stage1["stable_candidate"]].reset_index(drop=True)
    aggressive_focus_stage1 = screened_stage1[screened_stage1["aggressive_candidate"]].reset_index(drop=True)
    focus_candidates_stage1 = stable_focus_stage1.copy()
    preferred_stage1 = screened_stage1[
        screened_stage1["stable_candidate"].fillna(False) | screened_stage1["aggressive_candidate"].fillna(False)
    ].reset_index(drop=True)
    ranked_candidates_stage1 = preferred_stage1.head(config.top_n_stage1).copy() if not preferred_stage1.empty else screened_stage1.head(config.top_n_stage1).copy()
    return {
        "screened_stage1": screened_stage1,
        "stable_focus_stage1": stable_focus_stage1,
        "aggressive_focus_stage1": aggressive_focus_stage1,
        "focus_candidates_stage1": focus_candidates_stage1,
        "preferred_stage1": preferred_stage1,
        "ranked_candidates_stage1": ranked_candidates_stage1,
    }


def select_stage2_targets(ranked_candidates_stage1: pd.DataFrame, config: HolderStrategyConfig) -> list[str]:
    if not config.enable_stage2_cyq or ranked_candidates_stage1.empty:
        return []
    return ranked_candidates_stage1["ts_code"].dropna().astype(str).head(config.max_stage2_cyq_stocks).tolist()


def apply_holder_stage2(
    ranked_candidates_stage1: pd.DataFrame,
    stage2_cyq_metrics: pd.DataFrame,
    config: HolderStrategyConfig,
    market_regime: str,
) -> dict[str, Any]:
    reranked_candidates = ranked_candidates_stage1.copy()
    reranked_candidates = reranked_candidates.drop(
        columns=["cyq_checked", "cyq_rate_limited", "cyq_error", "winner_rate", "winner_rate_change_5d", "weight_avg", "cost_50pct", "close_vs_weight_avg_pct"],
        errors="ignore",
    )
    cyq_metrics = stage2_cyq_metrics.copy()
    if cyq_metrics.empty or "ts_code" not in cyq_metrics.columns:
        cyq_metrics = pd.DataFrame(columns=["ts_code"])
    reranked_candidates = reranked_candidates.merge(cyq_metrics, on="ts_code", how="left")
    reranked_candidates = ensure_columns(
        reranked_candidates,
        ["cyq_checked", "cyq_rate_limited", "cyq_error", "winner_rate", "winner_rate_change_5d", "weight_avg", "cost_50pct", "close_vs_weight_avg_pct"],
    )
    reranked_candidates["cyq_checked"] = reranked_candidates["cyq_checked"].where(pd.notna(reranked_candidates["cyq_checked"]), False).astype(bool)
    reranked_candidates["cyq_rate_limited"] = reranked_candidates["cyq_rate_limited"].where(pd.notna(reranked_candidates["cyq_rate_limited"]), False).astype(bool)
    reranked_candidates = score_candidates(
        reranked_candidates,
        min_volume_ratio=config.min_volume_ratio,
        max_price_position=config.max_price_position,
        max_industry_pb_pct=config.max_industry_pb_pct,
        min_final_score=config.min_final_score,
        min_aggressive_score=config.min_aggressive_score,
        market_regime=market_regime,
    ).reset_index(drop=True)
    reranked_candidates = reranked_candidates.sort_values(
        ["priority_score", "cyq_checked", "chip_score", "final_score", "event_score", "total_change_ratio"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    stable_focus_stage2 = reranked_candidates[reranked_candidates["stable_candidate"]].reset_index(drop=True)
    aggressive_focus_stage2 = reranked_candidates[reranked_candidates["aggressive_candidate"]].reset_index(drop=True)
    stable_candidates = (
        stable_focus_stage2.sort_values(["stable_score", "priority_score", "chip_score"], ascending=[False, False, False]).head(config.top_n_final).copy()
        if not stable_focus_stage2.empty
        else reranked_candidates.sort_values(["stable_score", "priority_score", "chip_score"], ascending=[False, False, False]).head(config.top_n_final).copy()
    )
    aggressive_candidates = (
        aggressive_focus_stage2.sort_values(["aggressive_score", "priority_score", "chip_score"], ascending=[False, False, False]).head(config.top_n_aggressive).copy()
        if not aggressive_focus_stage2.empty
        else reranked_candidates.sort_values(["aggressive_score", "priority_score", "chip_score"], ascending=[False, False, False]).head(config.top_n_aggressive).copy()
    )
    today_direction = "偏进攻" if market_regime == "risk_on" else "偏稳健"
    today_direction_pool = aggressive_candidates if today_direction == "偏进攻" else stable_candidates
    if today_direction_pool.empty:
        today_direction_pool = stable_candidates if not stable_candidates.empty else aggressive_candidates
    best_pick_candidate = today_direction_pool.head(1).copy()
    final_candidates = stable_candidates.copy()
    return {
        "reranked_candidates": reranked_candidates,
        "stable_focus_stage2": stable_focus_stage2,
        "aggressive_focus_stage2": aggressive_focus_stage2,
        "stable_candidates": stable_candidates,
        "aggressive_candidates": aggressive_candidates,
        "today_direction": today_direction,
        "best_pick_candidate": best_pick_candidate,
        "final_candidates": final_candidates,
    }


def build_screen_summary(
    config: HolderStrategyConfig,
    export_dir: Path,
    base_result: dict[str, Any],
    stage1_result: dict[str, Any],
    stage2_result: dict[str, Any],
    stage2_targets: list[str],
    stage2_cyq_metrics: pd.DataFrame,
) -> dict[str, Any]:
    reranked_candidates = stage2_result["reranked_candidates"]
    return {
        "latest_trade_date": base_result["latest_trade_date"],
        "ann_start_date": config.ann_start_date,
        "requested_end_date": config.end_date,
        "screen_end_date": config.end_date,
        "market_moneyflow_dates": base_result["market_moneyflow_dates"],
        "market_regime": base_result["market_regime"],
        "market_regime_score": base_result["market_regime_snapshot"].get("market_regime_score"),
        "market_trend_breadth": base_result["market_regime_snapshot"].get("market_trend_breadth"),
        "market_flow_breadth": base_result["market_regime_snapshot"].get("market_flow_breadth"),
        "market_hot_ratio": base_result["market_regime_snapshot"].get("market_hot_ratio"),
        "unlock_rows_lookahead": int(len(base_result["unlock_snapshot"])),
        "active_reduction_plan_rows": int(len(base_result["reduction_plan_snapshot"])),
        "raw_event_rows": int(len(base_result["event_pool"])),
        "event_pool_stocks": int(len(base_result["event_pool"])),
        "candidate_base_stocks": int(len(base_result["candidate_base"])),
        "deep_dive_stocks_stage1": int(stage1_result["deep_metrics_stage1"].shape[0]) if "deep_metrics_stage1" in stage1_result else 0,
        "focus_candidates_stage1": int(len(stage1_result["focus_candidates_stage1"])),
        "stable_focus_stage1": int(len(stage1_result["stable_focus_stage1"])),
        "aggressive_focus_stage1": int(len(stage1_result["aggressive_focus_stage1"])),
        "top_output_rows_stage1": int(len(stage1_result["ranked_candidates_stage1"])),
        "stage2_cyq_checked": int(pd.to_numeric(reranked_candidates["cyq_checked"], errors="coerce").fillna(0).sum()) if not reranked_candidates.empty else 0,
        "stage2_cyq_rate_limited": bool(reranked_candidates["cyq_rate_limited"].fillna(False).any()) if not reranked_candidates.empty and "cyq_rate_limited" in reranked_candidates.columns else False,
        "stage2_cyq_cached": 0,
        "stage2_cyq_requested_this_run": int(len(stage2_targets)),
        "stage2_cyq_pending": int(max(0, len(stage2_targets) - int(pd.to_numeric(reranked_candidates["cyq_checked"], errors="coerce").fillna(0).sum()))) if not reranked_candidates.empty else 0,
        "top_output_rows_stage2": int(len(stage2_result["final_candidates"])),
        "stable_focus_stage2": int(len(stage2_result["stable_focus_stage2"])),
        "aggressive_focus_stage2": int(len(stage2_result["aggressive_focus_stage2"])),
        "today_direction": stage2_result["today_direction"],
        "best_pick_ts_code": stage2_result["best_pick_candidate"].iloc[0]["ts_code"] if not stage2_result["best_pick_candidate"].empty else None,
        "export_dir": str(export_dir.resolve()),
        "config_snapshot": json_safe(config.to_dict()),
        "stage2_cyq_rows": int(len(stage2_cyq_metrics)),
    }


def run_holder_strategy_screening(
    config: HolderStrategyConfig,
    pro=None,
    custom_http_url: str = "",
    export_results: bool = True,
    export_root: Optional[Path] = None,
    resume_existing: bool = False,
    require_complete: bool = False,
) -> dict[str, Any]:
    if pro is None:
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        ensure_token(token)
        pro = configure_tushare_client(token, custom_http_url=custom_http_url or os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip())

    base_export_root = export_root if export_root is not None else output_root_dir()
    base_export_root.mkdir(parents=True, exist_ok=True)
    export_dir = base_export_root / f"holder_increase_screen_{config.end_date}"
    progress_path = export_dir / "screen_progress.json"
    if export_results:
        export_dir.mkdir(parents=True, exist_ok=True)

    recent_trade_dates = get_recent_open_trade_dates(pro, config.end_date, count=max(config.moneyflow_lookback_days, 10))
    stock_basic_all = fetch_stock_basic_all(pro)
    if stock_basic_all.empty:
        raise ValueError("stock_basic 接口未返回数据，无法继续。")

    holdertrade_raw = fetch_holdertrade_events(
        pro,
        config.ann_start_date,
        config.end_date,
        chunk_days=config.event_chunk_days,
        sleep_sec=config.api_sleep_sec,
    )
    unlock_end_date = (pd.Timestamp(config.end_date) + pd.Timedelta(days=config.unlock_lookahead_days)).strftime("%Y%m%d")
    share_float_schedule = fetch_share_float_schedule(
        pro,
        start_date=config.end_date,
        end_date=unlock_end_date,
        chunk_days=10,
        sleep_sec=config.api_sleep_sec,
    )
    latest_trade_date, market_moneyflow_dates, daily_basic_latest, tech_latest, moneyflow_summary = fetch_latest_complete_market_inputs(
        pro,
        recent_trade_dates,
        moneyflow_lookback_days=config.moneyflow_lookback_days,
        sleep_sec=config.api_sleep_sec,
    )
    base_result = build_holder_candidate_base(
        config=config,
        stock_basic_all=stock_basic_all,
        holdertrade_raw=holdertrade_raw,
        share_float_schedule=share_float_schedule,
        latest_trade_date=latest_trade_date,
        market_moneyflow_dates=market_moneyflow_dates,
        daily_basic_latest=daily_basic_latest,
        tech_latest=tech_latest,
        moneyflow_summary=moneyflow_summary,
    )
    candidate_base = base_result["candidate_base"]
    if export_results:
        write_csv_checkpoint(holdertrade_raw, export_dir / "holdertrade_raw.csv")
        write_csv_checkpoint(base_result["event_pool"], export_dir / "event_pool.csv")
        write_csv_checkpoint(base_result["reverse_signal_snapshot"], export_dir / "reverse_signal_snapshot.csv")
        write_csv_checkpoint(base_result["reduction_plan_snapshot"], export_dir / "reduction_plan_snapshot.csv")
        write_csv_checkpoint(share_float_schedule, export_dir / "share_float_schedule.csv")
        write_csv_checkpoint(base_result["unlock_snapshot"], export_dir / "unlock_snapshot.csv")
        write_csv_checkpoint(candidate_base, export_dir / "candidate_base.csv")

    deep_dive_targets = select_stage1_targets(candidate_base, config)

    deep_metrics_stage1 = pd.DataFrame()
    if export_results and resume_existing:
        deep_metrics_stage1 = dedupe_stage_rows(read_csv_if_exists(export_dir / "deep_metrics_stage1.csv"))
        if not deep_metrics_stage1.empty and "ts_code" in deep_metrics_stage1.columns:
            deep_metrics_stage1["ts_code"] = deep_metrics_stage1["ts_code"].astype(str)
            deep_metrics_stage1 = deep_metrics_stage1[deep_metrics_stage1["ts_code"].isin(deep_dive_targets)].reset_index(drop=True)

    completed_stage1_codes = completed_stage_codes(deep_metrics_stage1, deep_dive_targets, complete_col="stage1_complete")
    pending_stage1_targets = [code for code in deep_dive_targets if code not in completed_stage1_codes]
    if export_results:
        write_json_checkpoint(
            build_stage_progress_payload(
                config=config,
                status="stage1_running",
                trade_date=config.end_date,
                deep_dive_targets=deep_dive_targets,
                deep_metrics_stage1=deep_metrics_stage1,
                stage2_targets=[],
                stage2_cyq_metrics=pd.DataFrame(),
            ),
            progress_path,
        )

    for idx, ts_code in enumerate(pending_stage1_targets, start=1):
        log_step(f"[stage1 {idx}/{len(pending_stage1_targets)}] deep dive {ts_code}")
        stage1_row = fetch_single_stock_deep_metrics(
            pro,
            ts_code=ts_code,
            end_date=config.end_date,
            price_lookback_days=config.price_lookback_days,
            cyq_lookback_days=config.cyq_lookback_days,
            sleep_sec=config.api_sleep_sec,
            cyq_sleep_sec=config.cyq_sleep_sec,
            enable_forecast=config.enable_forecast,
            enable_cyq=False,
        )
        deep_metrics_stage1 = upsert_stage_row(deep_metrics_stage1, stage1_row)
        if export_results:
            write_csv_checkpoint(deep_metrics_stage1, export_dir / "deep_metrics_stage1.csv")
            write_json_checkpoint(
                build_stage_progress_payload(
                    config=config,
                    status="stage1_running",
                    trade_date=config.end_date,
                    deep_dive_targets=deep_dive_targets,
                    deep_metrics_stage1=deep_metrics_stage1,
                    stage2_targets=[],
                    stage2_cyq_metrics=pd.DataFrame(),
                ),
                progress_path,
            )

    completed_stage1_codes = completed_stage_codes(deep_metrics_stage1, deep_dive_targets, complete_col="stage1_complete")
    pending_stage1_targets = [code for code in deep_dive_targets if code not in completed_stage1_codes]
    if require_complete and pending_stage1_targets:
        error_message = f"stage1 incomplete pending={','.join(pending_stage1_targets[:10])}"
        if export_results:
            write_json_checkpoint(
                build_stage_progress_payload(
                    config=config,
                    status="stage1_incomplete",
                    trade_date=config.end_date,
                    deep_dive_targets=deep_dive_targets,
                    deep_metrics_stage1=deep_metrics_stage1,
                    stage2_targets=[],
                    stage2_cyq_metrics=pd.DataFrame(),
                    error=error_message,
                ),
                progress_path,
            )
        raise RuntimeError(error_message)

    stage1_result = apply_holder_stage1(candidate_base, deep_metrics_stage1, config, base_result["market_regime"])
    stage1_result["deep_metrics_stage1"] = deep_metrics_stage1
    stage2_targets = select_stage2_targets(stage1_result["ranked_candidates_stage1"], config)

    stage2_target_budget = stage2_targets[: config.stage2_cyq_budget]
    stage2_cyq_metrics = pd.DataFrame()
    if export_results and resume_existing:
        stage2_cyq_metrics = dedupe_stage_rows(read_csv_if_exists(export_dir / "stage2_cyq_metrics.csv"))
        if not stage2_cyq_metrics.empty and "ts_code" in stage2_cyq_metrics.columns:
            stage2_cyq_metrics["ts_code"] = stage2_cyq_metrics["ts_code"].astype(str)
            stage2_cyq_metrics = stage2_cyq_metrics[stage2_cyq_metrics["ts_code"].isin(stage2_target_budget)].reset_index(drop=True)

    completed_stage2_codes = completed_stage_codes(stage2_cyq_metrics, stage2_target_budget, complete_col="stage2_complete")
    pending_stage2_targets = [code for code in stage2_target_budget if code not in completed_stage2_codes]
    if export_results:
        write_json_checkpoint(
            build_stage_progress_payload(
                config=config,
                status="stage2_running",
                trade_date=config.end_date,
                deep_dive_targets=deep_dive_targets,
                deep_metrics_stage1=deep_metrics_stage1,
                stage2_targets=stage2_target_budget,
                stage2_cyq_metrics=stage2_cyq_metrics,
            ),
            progress_path,
        )

    for idx, ts_code in enumerate(pending_stage2_targets, start=1):
        latest_close = None
        matched = stage1_result["ranked_candidates_stage1"][stage1_result["ranked_candidates_stage1"]["ts_code"] == ts_code]
        if not matched.empty:
            latest_close = to_float(matched.iloc[0].get("latest_close_qfq_calc"))
        log_step(f"[stage2 {idx}/{len(pending_stage2_targets)}] cyq fetch {ts_code}")
        cyq_row = fetch_single_stock_cyq_metrics(
            pro,
            ts_code=ts_code,
            end_date=config.end_date,
            cyq_lookback_days=config.cyq_lookback_days,
            sleep_sec=config.cyq_sleep_sec,
            latest_close=latest_close,
        )
        stage2_cyq_metrics = upsert_stage_row(stage2_cyq_metrics, cyq_row)
        if export_results:
            write_csv_checkpoint(stage2_cyq_metrics, export_dir / "stage2_cyq_metrics.csv")
            write_json_checkpoint(
                build_stage_progress_payload(
                    config=config,
                    status="stage2_running",
                    trade_date=config.end_date,
                    deep_dive_targets=deep_dive_targets,
                    deep_metrics_stage1=deep_metrics_stage1,
                    stage2_targets=stage2_target_budget,
                    stage2_cyq_metrics=stage2_cyq_metrics,
                ),
                progress_path,
            )
        if cyq_row.get("cyq_rate_limited"):
            log_step("cyq_perf hit rate limit, stop stage2 fetch")
            break

    completed_stage2_codes = completed_stage_codes(stage2_cyq_metrics, stage2_target_budget, complete_col="stage2_complete")
    pending_stage2_targets = [code for code in stage2_target_budget if code not in completed_stage2_codes]
    if require_complete and pending_stage2_targets:
        error_message = f"stage2 incomplete pending={','.join(pending_stage2_targets[:10])}"
        if export_results:
            write_json_checkpoint(
                build_stage_progress_payload(
                    config=config,
                    status="stage2_incomplete",
                    trade_date=config.end_date,
                    deep_dive_targets=deep_dive_targets,
                    deep_metrics_stage1=deep_metrics_stage1,
                    stage2_targets=stage2_target_budget,
                    stage2_cyq_metrics=stage2_cyq_metrics,
                    error=error_message,
                ),
                progress_path,
            )
        raise RuntimeError(error_message)

    stage2_result = apply_holder_stage2(stage1_result["ranked_candidates_stage1"], stage2_cyq_metrics, config, base_result["market_regime"])

    if export_results:
        write_csv_checkpoint(deep_metrics_stage1, export_dir / "deep_metrics_stage1.csv")
        write_csv_checkpoint(stage1_result["screened_stage1"], export_dir / "screened_candidates_stage1.csv")
        write_csv_checkpoint(stage1_result["ranked_candidates_stage1"], export_dir / "ranked_candidates_stage1.csv")
        write_csv_checkpoint(stage2_cyq_metrics, export_dir / "stage2_cyq_metrics.csv")
        write_csv_checkpoint(stage2_result["reranked_candidates"], export_dir / "reranked_candidates_stage2.csv")
        write_csv_checkpoint(stage2_result["stable_candidates"], export_dir / "stable_candidates.csv")
        write_csv_checkpoint(stage2_result["aggressive_candidates"], export_dir / "aggressive_candidates.csv")
        write_csv_checkpoint(stage2_result["final_candidates"], export_dir / "final_candidates.csv")
        write_csv_checkpoint(stage2_result["best_pick_candidate"], export_dir / "best_pick_candidate.csv")
        screen_summary = build_screen_summary(
            config=config,
            export_dir=export_dir,
            base_result=base_result,
            stage1_result=stage1_result,
            stage2_result=stage2_result,
            stage2_targets=stage2_target_budget,
            stage2_cyq_metrics=stage2_cyq_metrics,
        )
        write_json_checkpoint(screen_summary, export_dir / "screen_summary.json")
        write_json_checkpoint(
            build_stage_progress_payload(
                config=config,
                status="completed",
                trade_date=config.end_date,
                deep_dive_targets=deep_dive_targets,
                deep_metrics_stage1=deep_metrics_stage1,
                stage2_targets=stage2_target_budget,
                stage2_cyq_metrics=stage2_cyq_metrics,
            ),
            progress_path,
        )
    else:
        screen_summary = build_screen_summary(
            config=config,
            export_dir=export_dir,
            base_result=base_result,
            stage1_result=stage1_result,
            stage2_result=stage2_result,
            stage2_targets=stage2_target_budget,
            stage2_cyq_metrics=stage2_cyq_metrics,
        )
    return {
        **base_result,
        **stage1_result,
        **stage2_result,
        "holdertrade_raw": holdertrade_raw,
        "share_float_schedule": share_float_schedule,
        "deep_dive_targets": deep_dive_targets,
        "stage2_targets": stage2_targets,
        "stage2_cyq_metrics": stage2_cyq_metrics,
        "screen_summary": screen_summary,
        "export_dir": export_dir,
        "best_columns": [c for c in display_columns() if c in stage2_result["best_pick_candidate"].columns],
    }
