from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from holder_strategy_core import (
    HolderStrategyConfig,
    configure_tushare_client,
    display_columns,
    ensure_token,
    ensure_columns,
    run_holder_strategy_screening,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pure Python holder strategy core without notebook execution.")
    parser.add_argument("--end-date", default="", help="Requested end date in YYYYMMDD. Default: today with 20:00 cutoff.")
    parser.add_argument("--ann-start-date", default="", help="Announcement start date in YYYYMMDD. Default: end-date minus 45 days.")
    parser.add_argument("--show-top", type=int, default=5, help="How many rows to print from the stable/aggressive pools.")
    parser.add_argument("--config-file", default="", help="Optional JSON file with HolderStrategyConfig override keys.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON object with HolderStrategyConfig override keys.")
    parser.add_argument("--resume-existing", action="store_true", help="Resume from partial export files for the same trade date.")
    parser.add_argument("--require-complete", action="store_true", help="Fail instead of exporting a partial result.")
    return parser.parse_args()


def load_json_file(path_str: str) -> dict:
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


def load_config_overrides(args: argparse.Namespace) -> dict:
    config: dict = {}
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


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    config_overrides = load_config_overrides(args)
    ensure_token(token)
    now_ts = pd.Timestamp.now()
    today_str = now_ts.strftime("%Y%m%d")
    requested_end_date = str(args.end_date or "").strip() or today_str
    screen_end_date = requested_end_date if not args.end_date else requested_end_date
    config = HolderStrategyConfig.for_end_date(screen_end_date, ann_start_date=args.ann_start_date, **config_overrides)
    pro = configure_tushare_client(token, custom_http_url=custom_http_url)
    result = run_holder_strategy_screening(
        config,
        pro=pro,
        custom_http_url=custom_http_url,
        export_results=True,
        resume_existing=args.resume_existing,
        require_complete=args.require_complete,
    )

    stable_candidates = result["stable_candidates"]
    aggressive_candidates = result["aggressive_candidates"]
    best_pick_candidate = result["best_pick_candidate"]
    cols = [c for c in display_columns() if c in stable_candidates.columns]

    print(json.dumps(result["screen_summary"], ensure_ascii=False, indent=2))
    print("===== stable candidates =====")
    print(ensure_columns(stable_candidates, cols)[cols].head(args.show_top).to_string(index=False) if not stable_candidates.empty else "(empty)")
    print("===== aggressive candidates =====")
    print(ensure_columns(aggressive_candidates, cols)[cols].head(args.show_top).to_string(index=False) if not aggressive_candidates.empty else "(empty)")
    print("===== best pick =====")
    print(ensure_columns(best_pick_candidate, cols)[cols].head(1).to_string(index=False) if not best_pick_candidate.empty else "(empty)")
    print(f"export_dir={result['export_dir']}")


if __name__ == "__main__":
    main()
