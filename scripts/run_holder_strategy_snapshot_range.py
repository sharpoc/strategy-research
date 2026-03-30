from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
from typing import Any

import pandas as pd

from holder_strategy_core import (
    HolderStrategyConfig,
    configure_tushare_client,
    run_holder_strategy_screening,
)
from research_backtest_utils import get_open_trade_dates, json_safe, log_step, repo_root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build resumable holder-strategy daily snapshots for a historical date range."
    )
    parser.add_argument("--start-date", required=True, help="Signal start date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Signal end date YYYYMMDD.")
    parser.add_argument(
        "--config-file",
        default="",
        help="Optional JSON file with HolderStrategyConfig override keys.",
    )
    parser.add_argument(
        "--config-json",
        default="",
        help="Optional inline JSON object with HolderStrategyConfig override keys.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dates whose snapshot export already looks complete.",
    )
    parser.add_argument(
        "--max-trade-days",
        type=int,
        default=0,
        help="Limit trade days for smoke tests.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the batch on the first failed trade date.",
    )
    parser.add_argument(
        "--snapshot-root",
        default="",
        help="Optional root directory for holder_increase_screen_<date> snapshots. Defaults to isolated research path.",
    )
    parser.add_argument(
        "--per-date-timeout-sec",
        type=int,
        default=300,
        help="Fail and kill a single trade-date worker after this many seconds. Set 0 to disable.",
    )
    return parser.parse_args()


def load_json_file(path_str: str) -> dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a JSON object: {path}")
    return data


def load_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    file_overrides = load_json_file(args.config_file)
    if file_overrides:
        config.update(file_overrides)
    if args.config_json.strip():
        try:
            inline = json.loads(args.config_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --config-json: {exc}") from exc
        if not isinstance(inline, dict):
            raise SystemExit("--config-json must be a JSON object.")
        config.update(inline)
    return config


def range_export_root_dir() -> Path:
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot_export_root_dir(path_str: str = "") -> Path:
    if path_str.strip():
        path = Path(path_str).expanduser().resolve()
    else:
        path = repo_root_dir() / "output" / "research_backtests" / "holder_snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_dir_for_trade_date(base_dir: Path, trade_date: str) -> Path:
    return base_dir / f"holder_increase_screen_{trade_date}"


def snapshot_is_complete(base_dir: Path, trade_date: str) -> bool:
    export_dir = export_dir_for_trade_date(base_dir, trade_date)
    required = [
        export_dir / "candidate_base.csv",
        export_dir / "deep_metrics_stage1.csv",
        export_dir / "reranked_candidates_stage2.csv",
        export_dir / "best_pick_candidate.csv",
        export_dir / "screen_summary.json",
    ]
    return export_dir.exists() and all(path.exists() for path in required)


def _run_trade_date_worker(
    trade_date: str,
    config_overrides: dict[str, Any],
    token: str,
    custom_http_url: str,
    snapshot_root: str,
    result_queue: mp.Queue,
) -> None:
    try:
        pro = configure_tushare_client(token, custom_http_url=custom_http_url)
        config = HolderStrategyConfig.for_end_date(trade_date, **config_overrides)
        result = run_holder_strategy_screening(
            config=config,
            pro=pro,
            custom_http_url=custom_http_url,
            export_results=True,
            export_root=Path(snapshot_root),
        )
        best_pick = result["best_pick_candidate"]
        payload = {
            "ok": True,
            "trade_date": trade_date,
            "market_regime": result["screen_summary"].get("market_regime"),
            "today_direction": result["screen_summary"].get("today_direction"),
            "best_pick_ts_code": None if best_pick.empty else str(best_pick.iloc[0].get("ts_code", "")),
            "best_pick_name": None if best_pick.empty else str(best_pick.iloc[0].get("name", "")),
            "export_dir": str(result["export_dir"]),
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "trade_date": trade_date,
            "error": repr(exc),
        }
    result_queue.put(payload)


def run_trade_date_with_timeout(
    trade_date: str,
    config_overrides: dict[str, Any],
    token: str,
    custom_http_url: str,
    snapshot_root: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    if timeout_sec <= 0:
        pro = configure_tushare_client(token, custom_http_url=custom_http_url)
        config = HolderStrategyConfig.for_end_date(trade_date, **config_overrides)
        result = run_holder_strategy_screening(
            config=config,
            pro=pro,
            custom_http_url=custom_http_url,
            export_results=True,
            export_root=snapshot_root,
        )
        best_pick = result["best_pick_candidate"]
        return {
            "ok": True,
            "trade_date": trade_date,
            "market_regime": result["screen_summary"].get("market_regime"),
            "today_direction": result["screen_summary"].get("today_direction"),
            "best_pick_ts_code": None if best_pick.empty else str(best_pick.iloc[0].get("ts_code", "")),
            "best_pick_name": None if best_pick.empty else str(best_pick.iloc[0].get("name", "")),
            "export_dir": str(result["export_dir"]),
        }

    result_queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_run_trade_date_worker,
        args=(trade_date, config_overrides, token, custom_http_url, str(snapshot_root), result_queue),
    )
    process.start()
    process.join(timeout=timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        return {
            "ok": False,
            "trade_date": trade_date,
            "error": f"TimeoutError('trade_date {trade_date} exceeded {timeout_sec}s')",
        }
    try:
        return result_queue.get_nowait()
    except Empty:
        return {
            "ok": False,
            "trade_date": trade_date,
            "error": f"RuntimeError('worker exited without result for {trade_date}')",
        }


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing TUSHARE_TOKEN in environment.")
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    config_overrides = load_config_overrides(args)
    snapshot_root = snapshot_export_root_dir(args.snapshot_root)
    pro = configure_tushare_client(token, custom_http_url=custom_http_url)
    trade_dates = get_open_trade_dates(pro, args.start_date, args.end_date)
    if args.max_trade_days > 0:
        trade_dates = trade_dates[: args.max_trade_days]
    if not trade_dates:
        raise SystemExit("No trade dates found in the requested range.")

    run_tag = f"holder_snapshot_range_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = range_export_root_dir() / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)

    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    total = len(trade_dates)

    for index, trade_date in enumerate(trade_dates, start=1):
        if args.skip_existing and snapshot_is_complete(snapshot_root, trade_date):
            log_step(f"holder snapshot {index}/{total} trade_date={trade_date} skipped existing export")
            skipped_rows.append({"trade_date": trade_date, "reason": "existing_snapshot"})
            continue
        try:
            log_step(f"holder snapshot {index}/{total} trade_date={trade_date} start")
            outcome = run_trade_date_with_timeout(
                trade_date=trade_date,
                config_overrides=config_overrides,
                token=token,
                custom_http_url=custom_http_url,
                snapshot_root=snapshot_root,
                timeout_sec=args.per_date_timeout_sec,
            )
            if outcome.get("ok"):
                success_rows.append(
                    {
                        "trade_date": trade_date,
                        "market_regime": outcome.get("market_regime"),
                        "today_direction": outcome.get("today_direction"),
                        "best_pick_ts_code": outcome.get("best_pick_ts_code"),
                        "best_pick_name": outcome.get("best_pick_name"),
                        "export_dir": outcome.get("export_dir"),
                    }
                )
            else:
                log_step(f"holder snapshot {index}/{total} trade_date={trade_date} failed error={outcome.get('error')}")
                failure_rows.append({"trade_date": trade_date, "error": str(outcome.get("error", "unknown error"))})
                if args.stop_on_error:
                    break
        except Exception as exc:
            log_step(f"holder snapshot {index}/{total} trade_date={trade_date} failed error={exc}")
            failure_rows.append({"trade_date": trade_date, "error": repr(exc)})
            if args.stop_on_error:
                break

    pd.DataFrame(success_rows).to_csv(export_dir / "success_rows.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(export_dir / "failure_rows.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(export_dir / "skipped_rows.csv", index=False)
    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "requested_trade_days": int(total),
        "success_count": int(len(success_rows)),
        "failure_count": int(len(failure_rows)),
        "skipped_count": int(len(skipped_rows)),
        "config_overrides": json_safe(config_overrides),
        "per_date_timeout_sec": int(args.per_date_timeout_sec),
        "snapshot_root": str(snapshot_root.resolve()),
        "export_dir": str(export_dir.resolve()),
    }
    with (export_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    if success_rows:
        print("===== success rows =====")
        print(pd.DataFrame(success_rows).to_string(index=False))
    if failure_rows:
        print("===== failure rows =====")
        print(pd.DataFrame(failure_rows).to_string(index=False))
    if skipped_rows:
        print("===== skipped rows =====")
        print(pd.DataFrame(skipped_rows).to_string(index=False))
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
