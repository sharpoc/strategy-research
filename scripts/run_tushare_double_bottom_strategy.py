from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    from double_bottom_strategy import build_double_bottom_snapshot
except ImportError:
    from scripts.double_bottom_strategy import build_double_bottom_snapshot


STRATEGY_NAME = "玄枢双底反转臻选"


def log_step(message: str) -> None:
    print(f"[double_bottom] {message}", flush=True)


def cache_enabled() -> bool:
    return os.getenv("DOUBLE_BOTTOM_USE_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def repo_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def cache_root_dir() -> Path:
    cache_dir = repo_root_dir() / "output" / "cache" / "double_bottom_api"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def fallback_cache_dirs() -> list[Path]:
    root = repo_root_dir() / "output" / "cache"
    return [
        root / "platform_breakout_api",
        root / "limitup_l1l2_api",
    ]


def discover_cached_trade_dates(end_date: str, count: int) -> list[str]:
    pattern = re.compile(r"^daily_all_(\d{8})_[0-9a-f]{12}\.csv$")
    dates: set[str] = set()
    for cache_dir in [cache_root_dir(), *fallback_cache_dirs()]:
        if not cache_dir.exists():
            continue
        for path in cache_dir.iterdir():
            match = pattern.match(path.name)
            if not match:
                continue
            trade_date = match.group(1)
            if trade_date <= end_date:
                dates.add(trade_date)
    ordered = sorted(dates)
    if not ordered:
        return [end_date]
    return ordered[-count:]


def cache_file_name(label: str, kwargs: dict) -> str:
    normalized = json.dumps(json_safe(kwargs), ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(f"{label}|{normalized}".encode("utf-8")).hexdigest()[:12]
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    return f"{safe_label}_{digest}.csv"


def load_cached_frame(file_name: str) -> Optional[pd.DataFrame]:
    search_paths = [cache_root_dir() / file_name]
    search_paths.extend(path / file_name for path in fallback_cache_dirs())
    for path in search_paths:
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
    path = cache_root_dir() / file_name
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        log_step(f"cache write failed {path.name} error={exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen the market for a double-bottom reversal structure and keep the strongest stock.")
    parser.add_argument("--end-date", default="", help="Requested end date in YYYYMMDD. Default: today with 20:00 cutoff.")
    parser.add_argument("--history-bars", type=int, default=170, help="Recent open-market bars used for structure reconstruction.")
    parser.add_argument("--min-score", type=float, default=62.0, help="Minimum strategy score to keep as a candidate.")
    parser.add_argument("--cutoff-hour", type=int, default=20, help="Use previous trading day before this hour.")
    parser.add_argument("--include-star", action="store_true", help="Include STAR board stocks.")
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


def ensure_columns(df: pd.DataFrame, columns: list[str], fill_value=np.nan) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column not in work.columns:
            work[column] = fill_value
    return work


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
            if sleep_sec:
                time.sleep(sleep_sec)
            if df is None:
                return pd.DataFrame()
            if cache_enabled() and not df.empty:
                save_cached_frame(file_name, df)
            return df.copy()
        except Exception as exc:
            last_exc = exc
            log_step(f"{label} failed attempt={attempt + 1} error={exc}")
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
    print(f"[{label}] 调用失败: {last_exc}")
    return pd.DataFrame()


def choose_screen_end_date(now_ts: pd.Timestamp, end_date: str, today_str: str, cutoff_hour: int = 20) -> str:
    requested = end_date or today_str
    if requested == today_str and int(now_ts.hour) < int(cutoff_hour):
        return (now_ts.normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    return requested


def get_recent_open_trade_dates(pro, end_date: str, count: int = 10) -> list[str]:
    cached_dates = discover_cached_trade_dates(end_date, count)
    if cached_dates and cached_dates[-1] == end_date and len(cached_dates) >= min(count, 40):
        log_step(f"trade_cal direct cache dates={len(cached_dates)}")
        return cached_dates

    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=max(30, count * 4))).strftime("%Y%m%d")
    cal = safe_call(
        "trade_cal",
        getattr(pro, "trade_cal", None),
        start_date=start_date,
        end_date=end_date,
        is_open="1",
    )
    if cal.empty:
        log_step(f"trade_cal cache fallback dates={len(cached_dates)}")
        return cached_dates
    date_col = "cal_date" if "cal_date" in cal.columns else "trade_date"
    dates = sorted(cal[date_col].dropna().astype(str).unique().tolist())
    if dates:
        return dates[-count:]
    log_step(f"trade_cal empty fallback dates={len(cached_dates)}")
    return cached_dates


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


def display_columns() -> list[str]:
    return [
        "ts_code",
        "name",
        "industry",
        "strategy_name",
        "strategy_rank_score",
        "double_bottom_score",
        "double_bottom_buy_type",
        "double_bottom_breakout_status",
        "double_bottom_reason",
        "double_bottom_l1_date",
        "double_bottom_h_date",
        "double_bottom_l2_date",
        "double_bottom_pre_down_pct",
        "double_bottom_rebound_pct",
        "double_bottom_l2_vs_l1_pct",
        "double_bottom_spacing_bars",
        "double_bottom_pullback_volume_ratio",
        "double_bottom_breakout_volume_ratio",
        "double_bottom_current_vs_ma20_pct",
        "double_bottom_space_to_120_high_pct",
        "close",
        "amount",
        "trade_date",
    ]


def run_double_bottom_screen(
    end_date: str = "",
    history_bars: int = 170,
    min_score: float = 62.0,
    cutoff_hour: int = 20,
    include_star: bool = False,
) -> dict:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    ensure_token(token)
    socket.setdefaulttimeout(int(os.getenv("DOUBLE_BOTTOM_SOCKET_TIMEOUT", "45")))
    ts.set_token(token)
    pro = ts.pro_api(token)
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "").strip()
    pro = configure_tushare_client(
        pro,
        token=token,
        use_custom_http_endpoint=bool(custom_http_url),
        custom_http_url=custom_http_url,
    )

    now_ts = pd.Timestamp.now()
    today_str = now_ts.strftime("%Y%m%d")
    requested_end_date = str(end_date or "").strip() or today_str
    screen_end_date = choose_screen_end_date(now_ts, requested_end_date, today_str, cutoff_hour)
    log_step(f"screen_end_date={screen_end_date} requested_end_date={requested_end_date}")

    pattern_trade_dates = get_recent_open_trade_dates(pro, screen_end_date, count=max(history_bars, 140))
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

    market_daily_history = fetch_market_daily_history(pro, pattern_trade_dates, sleep_sec=0.0)
    if market_daily_history.empty:
        raise SystemExit("daily history returned no data.")
    log_step(f"market_daily_history rows={len(market_daily_history)}")

    pattern_snapshot = build_double_bottom_snapshot(
        market_daily_history,
        stock_basic_df=stock_basic_all,
        config={
            "candidate_score_threshold": min_score,
            "include_star": include_star,
        },
    )

    latest_market = (
        market_daily_history.sort_values(["ts_code", "trade_date"]).groupby("ts_code", as_index=False).tail(1).copy()
        if not market_daily_history.empty
        else pd.DataFrame()
    )
    latest_trade_date = str(latest_market["trade_date"].max()) if not latest_market.empty else screen_end_date

    candidates = stock_basic_all.merge(latest_market, on="ts_code", how="left")
    if not pattern_snapshot.empty:
        candidates = candidates.merge(pattern_snapshot, on="ts_code", how="left")
    candidates = ensure_columns(
        candidates,
        [
            "double_bottom_signal",
            "double_bottom_score",
            "double_bottom_buy_type",
            "double_bottom_breakout_status",
            "double_bottom_reason",
            "double_bottom_breakout_volume_ratio",
            "double_bottom_space_to_120_high_pct",
        ],
    )
    candidates["double_bottom_signal"] = candidates["double_bottom_signal"].apply(lambda value: bool(value) if pd.notna(value) else False)
    candidates["double_bottom_score"] = pd.to_numeric(candidates["double_bottom_score"], errors="coerce").fillna(0.0)
    candidates["strategy_name"] = np.where(candidates["double_bottom_signal"], STRATEGY_NAME, "")
    candidates["strategy_rank_score"] = candidates["double_bottom_score"]

    buy_rank_map = {"B": 3, "A": 2, "C": 1}
    candidates["buy_type_rank"] = candidates["double_bottom_buy_type"].map(buy_rank_map).fillna(0).astype(int)
    strategy_candidates = candidates[candidates["double_bottom_signal"]].copy()
    if not strategy_candidates.empty:
        strategy_candidates = strategy_candidates.sort_values(
            [
                "strategy_rank_score",
                "buy_type_rank",
                "double_bottom_breakout_volume_ratio",
                "double_bottom_space_to_120_high_pct",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    best_pick_candidate = strategy_candidates.head(1).copy()

    output_root = repo_root_dir() / "output" / "jupyter-notebook" / "tushare_double_bottom_exports"
    output_root.mkdir(parents=True, exist_ok=True)
    run_tag = f"double_bottom_pick_{screen_end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = output_root / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)

    pattern_snapshot.to_csv(export_dir / "pattern_snapshot.csv", index=False)
    strategy_candidates.to_csv(export_dir / "strategy_candidates.csv", index=False)
    best_pick_candidate.to_csv(export_dir / "best_pick_candidate.csv", index=False)

    summary = {
        "strategy_name": STRATEGY_NAME,
        "requested_end_date": requested_end_date,
        "screen_end_date": screen_end_date,
        "latest_trade_date": latest_trade_date,
        "history_bars": history_bars,
        "min_score": min_score,
        "include_star": include_star,
        "pattern_snapshot_stocks": int(len(pattern_snapshot)),
        "strategy_candidates": int(len(strategy_candidates)),
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
    result = run_double_bottom_screen(
        end_date=args.end_date,
        history_bars=args.history_bars,
        min_score=args.min_score,
        cutoff_hour=args.cutoff_hour,
        include_star=args.include_star,
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
