#!/usr/bin/env python3

from __future__ import annotations

import json
import time
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_REPO_ROOT = Path(
    os.getenv("LAB_REPO_ROOT", str(REPO_ROOT.parent / "strategy-lab"))
).expanduser().resolve()
if str(LAB_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_REPO_ROOT))

from server.market_data import TushareTrackingClient  # type: ignore  # noqa: E402


DEFAULT_STRATEGY_IDS = (
    "holder_increase_screening",
    "holder_chip_enhanced_screening",
    "event_conviction_signal",
    "tail_rebound_screening",
)
TOKEN_HEADER = "X-Strategy-Lab-Token"


@dataclass(frozen=True)
class TrackingSettings:
    tushare_token: str
    tushare_http_url: str
    timezone: ZoneInfo


def ensure_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def now_local(tz: ZoneInfo) -> datetime:
    return datetime.now(tz)


def http_json(
    method: str,
    url: str,
    token: str,
    payload: Dict[str, Any] | None = None,
    timeout: int = 30,
    max_attempts: int = 1,
    retry_sleep_seconds: float = 0.0,
) -> Dict[str, Any]:
    body = None
    headers = {
        TOKEN_HEADER: token,
        "Accept": "application/json",
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
    attempts = max(1, int(max_attempts))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = SystemExit(f"HTTP {exc.code} for {url}: {detail}")
        except Exception as exc:
            last_error = SystemExit(f"request failed for {url}: {exc}")
        if attempt < attempts:
            time.sleep(max(float(retry_sleep_seconds), 0.0))
    raise last_error if last_error is not None else SystemExit(f"request failed for {url}")


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def configured_strategy_ids() -> List[str]:
    raw_many = os.getenv("TRACKING_STRATEGY_IDS", "").strip()
    if raw_many:
        ids = [item.strip() for item in raw_many.replace(";", ",").split(",") if item.strip()]
        if ids:
            return list(dict.fromkeys(ids))
    raw_single = os.getenv("TRACKING_STRATEGY_ID", "").strip()
    if raw_single:
        return [raw_single]
    return list(DEFAULT_STRATEGY_IDS)


def push_single_stock_snapshot(
    *,
    api_base_url: str,
    api_token: str,
    strategy_id: str,
    trade_day: str,
    refresh_time: str,
    snapshot: Dict[str, Any],
    timeout: int,
    max_attempts: int,
    retry_sleep_seconds: float,
) -> Dict[str, Any]:
    ts_code = str(snapshot.get("ts_code") or "").strip()
    return http_json(
        "POST",
        api_base_url.rstrip("/") + f"/api/internal/strategies/{strategy_id}/tracking-snapshots",
        api_token,
        payload={
            "trade_day": trade_day,
            "trigger_source": f"mac-mini-tracking-{refresh_time}-{ts_code}",
            "snapshots": [snapshot],
        },
        timeout=timeout,
        max_attempts=max_attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    )


def main() -> int:
    api_base_url = os.getenv("TRACKING_API_BASE_URL", "").strip() or os.getenv("APP_INTERNAL_API_BASE_URL", "").strip()
    api_token = os.getenv("TRACKING_API_TOKEN", "").strip() or os.getenv("APP_INTERNAL_API_TOKEN", "").strip()
    tushare_token = os.getenv("TUSHARE_TOKEN", "").strip()
    tushare_http_url = os.getenv("TUSHARE_HTTP_URL", "").strip()
    if not api_base_url:
        raise SystemExit("TRACKING_API_BASE_URL or APP_INTERNAL_API_BASE_URL is required.")
    if not api_token:
        raise SystemExit("TRACKING_API_TOKEN or APP_INTERNAL_API_TOKEN is required.")
    if not tushare_token:
        raise SystemExit("TUSHARE_TOKEN is required.")

    timezone = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Shanghai").strip() or "Asia/Shanghai")
    timeout = int(os.getenv("TRACKING_API_TIMEOUT_SECONDS", "30"))
    http_max_attempts = int(os.getenv("TRACKING_HTTP_MAX_ATTEMPTS", "3"))
    http_retry_sleep_seconds = float(os.getenv("TRACKING_HTTP_RETRY_SLEEP_SECONDS", "2"))
    state_dir = ensure_dir(os.getenv("TRACKING_STATE_DIR", "ops/state"))
    strategy_ids = configured_strategy_ids()

    tracked_by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    tracked_codes: List[str] = []
    for strategy_id in strategy_ids:
        tracked_payload = http_json(
            "GET",
            api_base_url.rstrip("/") + f"/api/internal/strategies/{strategy_id}/tracked-stocks",
            api_token,
            timeout=timeout,
            max_attempts=http_max_attempts,
            retry_sleep_seconds=http_retry_sleep_seconds,
        )
        tracked_stocks = tracked_payload.get("tracked_stocks") or []
        if not tracked_stocks:
            print(f"No tracked stocks returned for {strategy_id}; skip this strategy.")
            tracked_by_strategy[strategy_id] = []
            continue
        clean_codes = [
            str(row.get("ts_code") or "").strip()
            for row in tracked_stocks
            if str(row.get("ts_code") or "").strip()
        ]
        tracked_by_strategy[strategy_id] = tracked_stocks
        tracked_codes.extend(clean_codes)

    tracked_codes = list(dict.fromkeys(tracked_codes))
    if not tracked_codes:
        print("No tracked stocks returned for configured strategies; skip snapshot refresh.")
        return 0

    client = TushareTrackingClient(
        TrackingSettings(
            tushare_token=tushare_token,
            tushare_http_url=tushare_http_url,
            timezone=timezone,
        )
    )
    live_snapshots = client.fetch_live_snapshots(tracked_codes)
    if not live_snapshots:
        raise SystemExit("live snapshots are empty.")

    now_dt = now_local(timezone)
    live_trade_days = [
        str(snapshot.get("trade_day") or "")
        for snapshot in live_snapshots.values()
        if str(snapshot.get("trade_day") or "")
    ]
    trade_day = max(live_trade_days) if live_trade_days else now_dt.strftime("%Y%m%d")
    refresh_time = now_dt.strftime("%H:%M")
    summary_by_strategy: Dict[str, Any] = {}
    total_pushed = 0
    total_tracked = 0
    for strategy_id in strategy_ids:
        strategy_rows = tracked_by_strategy.get(strategy_id) or []
        strategy_codes = [
            str(row.get("ts_code") or "").strip()
            for row in strategy_rows
            if str(row.get("ts_code") or "").strip()
        ]
        total_tracked += len(strategy_codes)
        rows_for_push: List[Dict[str, Any]] = []
        missing_codes: List[str] = []
        for ts_code in strategy_codes:
            snapshot = live_snapshots.get(ts_code)
            if not snapshot:
                missing_codes.append(ts_code)
                continue
            enriched = dict(snapshot)
            enriched["trade_day"] = trade_day
            if not str(enriched.get("trade_time") or "").strip():
                enriched["trade_time"] = now_dt.strftime("%Y%m%d%H%M%S")
            rows_for_push.append(enriched)

        responses: Dict[str, Any] = {}
        for row in rows_for_push:
            ts_code = str(row.get("ts_code") or "").strip()
            if not ts_code:
                continue
            responses[ts_code] = push_single_stock_snapshot(
                api_base_url=api_base_url,
                api_token=api_token,
                strategy_id=strategy_id,
                trade_day=trade_day,
                refresh_time=refresh_time,
                snapshot=row,
                timeout=timeout,
                max_attempts=http_max_attempts,
                retry_sleep_seconds=http_retry_sleep_seconds,
            )
        total_pushed += len(rows_for_push)
        state_path = state_dir / f"{strategy_id}_tracking_state.json"
        save_state(
            state_path,
            {
                "last_trade_day": trade_day,
                "last_refresh_time": refresh_time,
                "push_mode": "single-stock",
                "snapshots": {
                    row["ts_code"]: {
                        "close": row.get("close"),
                        "pct_change": row.get("pct_change"),
                        "trade_time": row.get("trade_time"),
                    }
                    for row in rows_for_push
                },
            },
        )
        summary_by_strategy[strategy_id] = {
            "responses": responses,
            "tracked_count": len(strategy_codes),
            "pushed_count": len(rows_for_push),
            "missing_codes": missing_codes,
            "state_path": str(state_path),
        }

    print(
        json.dumps(
            {
                "strategies": summary_by_strategy,
                "tracking_push_ok": True,
                "push_mode": "single-stock",
                "strategy_count": len(strategy_ids),
                "tracked_count": total_tracked,
                "pushed_count": total_pushed,
                "trade_day": trade_day,
                "refresh_time": refresh_time,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
