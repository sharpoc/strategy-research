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
    from platform_breakout_retest_strategy import build_platform_breakout_snapshot
except ImportError:
    from scripts.platform_breakout_retest_strategy import build_platform_breakout_snapshot


def log_step(message: str) -> None:
    print(f"[platform_breakout] {message}", flush=True)


def cache_enabled() -> bool:
    return os.getenv("PLATFORM_BREAKOUT_USE_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def cache_root_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = repo_root / "output" / "cache" / "platform_breakout_api"
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


def safe_call(label: str, fn, sleep_sec: float = 0.0, retries: int = 2, **kwargs) -> pd.DataFrame:
    if fn is None:
        return pd.DataFrame()
    cache_path = cache_file_path(label, kwargs) if cache_enabled() else None
    if cache_path is not None:
        cached = load_cached_frame(cache_path)
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
            if cache_path is not None and not df.empty:
                save_cached_frame(cache_path, df)
            return df.copy()
        except Exception as exc:
            last_exc = exc
            log_step(f"{label} failed attempt={attempt + 1} error={exc}")
            if attempt < retries:
                time.sleep(0.7 * (attempt + 1))
    print(f"[{label}] 调用失败: {last_exc}")
    return pd.DataFrame()


def ensure_token(token: str) -> None:
    if not token or token.startswith("PASTE_"):
        raise SystemExit("Missing TUSHARE_TOKEN in environment.")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen the market for the platform-breakout pullback-restart strategy and pick the strongest stock.")
    parser.add_argument("--end-date", default="", help="Requested end date in YYYYMMDD. Default: today with 20:00 cutoff.")
    parser.add_argument("--history-bars", type=int, default=60, help="Recent open-market bars used for strategy reconstruction.")
    parser.add_argument("--min-score", type=float, default=60.0, help="Minimum strategy score to keep as a candidate.")
    parser.add_argument("--cutoff-hour", type=int, default=20, help="Use previous trading day before this hour.")
    parser.add_argument("--show-top", type=int, default=10, help="Rows to print from the strategy ranking.")
    return parser.parse_args()


def display_columns() -> list[str]:
    return [
        "ts_code",
        "name",
        "industry",
        "strategy_name",
        "strategy_rank_score",
        "platform_breakout_score",
        "platform_breakout_board",
        "platform_breakout_reason",
        "platform_breakout_limit_date",
        "platform_breakout_platform_start",
        "platform_breakout_platform_end",
        "platform_breakout_pullback_low_date",
        "platform_breakout_strength_date",
        "platform_breakout_platform_days",
        "platform_breakout_platform_amp_pct",
        "platform_breakout_limit_volume_ratio",
        "platform_breakout_pullback_ratio",
        "platform_breakout_pullback_avg_vol_ratio",
        "platform_breakout_current_volume_ratio",
        "platform_breakout_current_close_to_high_pct",
        "platform_breakout_pre20_runup_pct",
        "close",
        "vol",
        "amount",
        "trade_date",
    ]


def run_platform_breakout_screen(
    end_date: str = "",
    history_bars: int = 60,
    min_score: float = 60.0,
    cutoff_hour: int = 20,
) -> dict:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    ensure_token(token)
    socket.setdefaulttimeout(int(os.getenv("PLATFORM_BREAKOUT_SOCKET_TIMEOUT", "45")))
    ts.set_token(token)
    pro = ts.pro_api(token)

    now_ts = pd.Timestamp.now()
    today_str = now_ts.strftime("%Y%m%d")
    requested_end_date = str(end_date or "").strip() or today_str
    screen_end_date = choose_screen_end_date(now_ts, requested_end_date, today_str, cutoff_hour)
    log_step(f"screen_end_date={screen_end_date} requested_end_date={requested_end_date}")

    pattern_trade_dates = get_recent_open_trade_dates(pro, screen_end_date, count=max(history_bars, 40))
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

    pattern_snapshot = build_platform_breakout_snapshot(
        market_daily_history,
        stock_basic_df=stock_basic_all,
        config={"candidate_score_threshold": min_score},
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
            "platform_breakout_signal",
            "platform_breakout_score",
            "platform_breakout_reason",
            "platform_breakout_board",
            "platform_breakout_limit_date",
            "platform_breakout_platform_start",
            "platform_breakout_platform_end",
            "platform_breakout_pullback_low_date",
            "platform_breakout_strength_date",
            "platform_breakout_platform_days",
            "platform_breakout_platform_amp_pct",
            "platform_breakout_limit_volume_ratio",
            "platform_breakout_pullback_ratio",
            "platform_breakout_pullback_avg_vol_ratio",
            "platform_breakout_current_volume_ratio",
            "platform_breakout_current_close_to_high_pct",
            "platform_breakout_pre20_runup_pct",
        ],
    )
    candidates["platform_breakout_signal"] = candidates["platform_breakout_signal"].apply(
        lambda value: bool(value) if pd.notna(value) else False
    )
    candidates["platform_breakout_score"] = pd.to_numeric(candidates["platform_breakout_score"], errors="coerce").fillna(0.0)
    candidates["strategy_name"] = np.where(candidates["platform_breakout_signal"], "天衡回踩转强臻选", "")
    candidates["preferred_pool"] = np.where(candidates["platform_breakout_signal"], "stable", "")
    candidates["priority_score"] = candidates["platform_breakout_score"]
    candidates["final_score"] = candidates["platform_breakout_score"]
    candidates["strategy_rank_score"] = candidates["platform_breakout_score"]

    strategy_candidates = candidates[candidates["platform_breakout_signal"]].copy()
    strategy_candidates = strategy_candidates.sort_values(
        [
            "strategy_rank_score",
            "platform_breakout_current_volume_ratio",
            "platform_breakout_pullback_ratio",
            "platform_breakout_limit_volume_ratio",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best_pick_candidate = strategy_candidates.head(1).copy()

    repo_root = Path(__file__).resolve().parent.parent
    output_root = repo_root / "output" / "jupyter-notebook" / "tushare_platform_breakout_exports"
    output_root.mkdir(parents=True, exist_ok=True)
    run_tag = f"platform_breakout_pick_{screen_end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = output_root / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)

    pattern_snapshot.to_csv(export_dir / "pattern_snapshot.csv", index=False)
    strategy_candidates.to_csv(export_dir / "strategy_candidates.csv", index=False)
    best_pick_candidate.to_csv(export_dir / "best_pick_candidate.csv", index=False)

    summary = {
        "strategy_name": "天衡回踩转强臻选",
        "requested_end_date": requested_end_date,
        "screen_end_date": screen_end_date,
        "latest_trade_date": latest_trade_date,
        "history_bars": history_bars,
        "min_score": min_score,
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
    result = run_platform_breakout_screen(
        end_date=args.end_date,
        history_bars=args.history_bars,
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
