from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from financial_report_catalyst_strategy import (
    FinancialReportCatalystConfig,
    configure_tushare_client,
    ensure_token,
    run_financial_report_catalyst_screening,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the financial report catalyst strategy.")
    parser.add_argument("--end-date", default="", help="Screen trade date in YYYYMMDD. Default: today.")
    parser.add_argument("--show-top", type=int, default=5, help="Rows to print from final candidates.")
    parser.add_argument("--config-file", default="", help="Optional JSON file with config override keys.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON object with config override keys.")
    parser.add_argument("--export-root", default="", help="Optional base directory for strategy exports.")
    parser.add_argument("--export-prefix", default="financial_report_catalyst_", help="Export directory prefix.")
    return parser.parse_args()


def load_json_file(path_str: str) -> dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a JSON object: {path}")
    return data


def load_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    config.update(load_json_file(args.config_file))
    if args.config_json.strip():
        inline = json.loads(args.config_json)
        if not isinstance(inline, dict):
            raise SystemExit("--config-json must be a JSON object.")
        config.update(inline)
    return config


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    ensure_token(token)

    screen_end_date = str(args.end_date or "").strip() or pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
    config = FinancialReportCatalystConfig.for_end_date(screen_end_date, **load_config_overrides(args))
    pro = configure_tushare_client(token, custom_http_url=custom_http_url)
    result = run_financial_report_catalyst_screening(
        config,
        pro=pro,
        export_results=True,
        export_root=Path(args.export_root).expanduser().resolve() if args.export_root.strip() else None,
        export_prefix=args.export_prefix,
    )
    candidates = result["candidates"]
    print(json.dumps(result["screen_summary"], ensure_ascii=False, indent=2))
    print("===== financial report catalyst candidates =====")
    if candidates.empty:
        print("(empty)")
    else:
        cols = [
            "ts_code",
            "name",
            "industry",
            "report_period",
            "report_catalyst_date",
            "report_catalyst_type",
            "close",
            "pe_ttm",
            "pb",
            "quality_score",
            "financial_report_score",
        ]
        cols = [col for col in cols if col in candidates.columns]
        print(candidates[cols].head(args.show_top).to_string(index=False))
    print("===== best pick =====")
    best_pick = result["best_pick_candidate"]
    if best_pick.empty:
        print("(empty)")
    else:
        cols = [col for col in ("ts_code", "name", "close", "report_catalyst_date", "financial_report_score") if col in best_pick.columns]
        print(best_pick[cols].head(1).to_string(index=False))
    print(f"export_dir={result['export_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
