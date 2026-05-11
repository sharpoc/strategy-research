#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY_ID = "tail_rebound_screening"
TOKEN_HEADER = "X-Strategy-Lab-Token"


def env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def normalize_path(value: str | None, default: Path) -> Path:
    raw = str(value or "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def normalize_trade_day(value: str) -> str:
    trade_day = str(value or "").strip().replace("-", "")
    if not trade_day:
        return ""
    if len(trade_day) != 8 or not trade_day.isdigit():
        raise SystemExit(f"invalid trade day: {value}")
    return trade_day


def parse_args() -> argparse.Namespace:
    timezone = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Shanghai").strip() or "Asia/Shanghai")
    today = datetime.now(timezone).strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Run rebound stock picker and push result to strategy-lab internal API.")
    parser.add_argument("--trade-date", default=os.getenv("TRADE_DATE", today), help="Trade day in YYYYMMDD.")
    parser.add_argument("--strategy-id", default=os.getenv("REBOUND_STRATEGY_ID", DEFAULT_STRATEGY_ID))
    parser.add_argument("--top", type=int, default=int(os.getenv("REBOUND_TOP_N", "5")))
    parser.add_argument("--summary-json", default=os.getenv("REBOUND_SUMMARY_JSON", ""), help="Use an existing summary.json instead of running the picker.")
    parser.add_argument("--no-push", action="store_true", default=env_bool("REBOUND_NO_PUSH", False))
    parser.add_argument("--historical", action="store_true", default=env_bool("REBOUND_FORCE_HISTORICAL", False))
    parser.add_argument("--no-cache", action="store_true", default=env_bool("REBOUND_NO_CACHE", False))
    parser.add_argument("--api-base-url", default=os.getenv("APP_INTERNAL_API_BASE_URL", "") or os.getenv("TRACKING_API_BASE_URL", ""))
    parser.add_argument("--api-token", default=os.getenv("APP_INTERNAL_API_TOKEN", "") or os.getenv("TRACKING_API_TOKEN", ""))
    parser.add_argument("--timeout-seconds", type=int, default=int(os.getenv("APP_INTERNAL_API_TIMEOUT_SECONDS", "30")))
    parser.add_argument("--http-max-attempts", type=int, default=int(os.getenv("APP_INTERNAL_API_MAX_ATTEMPTS", "3")))
    parser.add_argument("--http-retry-sleep-seconds", type=float, default=float(os.getenv("APP_INTERNAL_API_RETRY_SLEEP_SECONDS", "2")))
    return parser.parse_args()


def is_transient_http_failure(message: str) -> bool:
    markers = (
        "HTTP 502",
        "HTTP 503",
        "HTTP 504",
        "Empty reply",
        "Remote end closed connection without response",
        "Connection reset by peer",
        "timed out",
        "Bad Gateway",
    )
    return any(marker in str(message or "") for marker in markers)


def http_json(
    method: str,
    url: str,
    token: str,
    payload: Dict[str, Any],
    timeout_seconds: int,
    *,
    max_attempts: int,
    retry_sleep_seconds: float,
) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        TOKEN_HEADER: token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error = SystemExit(f"HTTP {exc.code} for {url}: {detail}")
        except Exception as exc:
            error = SystemExit(f"request failed for {url}: {exc}")
        if attempt >= attempts or not is_transient_http_failure(str(error)):
            raise error
        time.sleep(max(float(retry_sleep_seconds), 0.0))
    raise SystemExit(f"request failed for {url}")


def run_picker(args: argparse.Namespace, trade_day: str) -> Path:
    project_root = normalize_path(
        os.getenv("REBOUND_RATER_PROJECT_ROOT"),
        REPO_ROOT / "vendor" / "tushare-stock-rater",
    )
    if not project_root.exists():
        raise SystemExit(f"REBOUND_RATER_PROJECT_ROOT does not exist: {project_root}")

    reports_dir = normalize_path(os.getenv("REBOUND_REPORT_ROOT"), REPO_ROOT / "output" / "rebound_reports")
    cache_dir = normalize_path(os.getenv("REBOUND_CACHE_DIR"), project_root / "data" / "cache")
    config_path = normalize_path(os.getenv("REBOUND_CONFIG_PATH"), project_root / "configs" / "scoring_weights.yaml")
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    python_bin = os.getenv("PYTHON_BIN", sys.executable or "python3")
    cmd = [
        python_bin,
        "-m",
        "tushare_stock_rater",
        "pick-rebound",
        "--top",
        str(args.top),
        "--out",
        str(reports_dir),
        "--config",
        str(config_path),
        "--cache-dir",
        str(cache_dir),
    ]
    timezone = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Shanghai").strip() or "Asia/Shanghai")
    today = datetime.now(timezone).strftime("%Y%m%d")
    if args.historical or trade_day != today:
        cmd.extend(["--as-of", trade_day, "--historical"])
    if args.no_cache:
        cmd.append("--no-cache")

    env = os.environ.copy()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=str(project_root), env=env, check=True)

    summary_paths = sorted(reports_dir.glob(f"rebound_{trade_day}_*/summary.json"))
    if not summary_paths:
        raise SystemExit(f"summary.json not found under {reports_dir} for trade_day={trade_day}")
    return summary_paths[-1]


def map_candidate(candidate: Dict[str, Any], trade_day: str) -> Dict[str, Any]:
    mapped = dict(candidate)
    score = candidate.get("score")
    mapped["trade_date"] = str(candidate.get("trade_date") or trade_day)
    mapped["latest_change_date"] = mapped["trade_date"]
    mapped["priority_score"] = score
    mapped["final_score"] = score
    mapped["preferred_pool"] = "tail_rebound"
    mapped["pct_change"] = candidate.get("pct_chg")
    return mapped


def build_result_payload(summary_payload: Dict[str, Any], trade_day: str) -> Dict[str, Any]:
    candidates = summary_payload.get("candidates") or []
    mapped_candidates = [map_candidate(row, trade_day) for row in candidates if isinstance(row, dict)]
    return {
        "summary": {
            "screen_end_date": trade_day,
            "latest_trade_date": trade_day,
            "today_direction": "尾盘反抽候选",
            "strategy_mode": summary_payload.get("mode"),
            "candidate_count": len(mapped_candidates),
            "warnings": summary_payload.get("warnings") or [],
        },
        "best_pick": mapped_candidates[0] if mapped_candidates else None,
        "final_candidates": mapped_candidates,
        "stage1_candidates": mapped_candidates,
    }


def main() -> int:
    args = parse_args()
    trade_day = normalize_trade_day(args.trade_date)
    strategy_id = str(args.strategy_id or DEFAULT_STRATEGY_ID).strip()
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else run_picker(args, trade_day)
    if not summary_path.exists():
        raise SystemExit(f"summary.json does not exist: {summary_path}")

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    result_trade_day = normalize_trade_day(str(summary_payload.get("trade_date") or trade_day))
    if result_trade_day != trade_day:
        raise SystemExit(f"summary trade day {result_trade_day} does not match requested trade day {trade_day}")
    result_payload = build_result_payload(summary_payload, trade_day)
    payload = {
        "trade_day": trade_day,
        "strategy_id": strategy_id,
        "trigger_source": f"mac-mini-rebound-{trade_day}",
        "export_dir": str(summary_path.parent),
        "result": result_payload,
    }

    response: Dict[str, Any] | None = None
    if not args.no_push:
        if not str(args.api_base_url or "").strip():
            raise SystemExit("APP_INTERNAL_API_BASE_URL is required")
        if not str(args.api_token or "").strip():
            raise SystemExit("APP_INTERNAL_API_TOKEN is required")
        response = http_json(
            "POST",
            args.api_base_url.rstrip("/") + f"/api/internal/strategies/{strategy_id}/imports",
            args.api_token.strip(),
            payload,
            args.timeout_seconds,
            max_attempts=args.http_max_attempts,
            retry_sleep_seconds=args.http_retry_sleep_seconds,
        )

    print(
        json.dumps(
            {
                "trade_day": trade_day,
                "strategy_id": strategy_id,
                "summary_json": str(summary_path),
                "candidate_count": len(result_payload["final_candidates"]),
                "pushed": not args.no_push,
                "response": response,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
