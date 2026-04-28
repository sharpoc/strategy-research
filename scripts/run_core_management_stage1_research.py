from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import run_core_management_final_review as review
from research_backtest_utils import json_safe, repo_root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified 6m stage1 baseline for core-management accumulation research."
    )
    parser.add_argument("--start-date", default="20250925", help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", default="20260324", help="End date in YYYYMMDD.")
    parser.add_argument("--export-root", default="", help="Optional export directory. Defaults to output/research_backtests.")
    parser.add_argument("--hold-days", default="3,5,10", help="Comma-separated holding windows.")
    parser.add_argument("--config-file", default="", help="Optional JSON file with config overrides.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON with config overrides.")
    parser.add_argument("--api-sleep-sec", type=float, default=0.12, help="Sleep between API calls.")
    return parser.parse_args()


def export_root_dir(path_str: str = "") -> Path:
    if path_str.strip():
        path = Path(path_str).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_stage1_report(summary_payload: dict[str, Any], stage1_df: pd.DataFrame, progress_df: pd.DataFrame) -> str:
    stage1_summary = summary_payload["stage1_summary"]
    duplicate_payload = summary_payload["stage1_ts_code_duplicates"]
    lines: list[str] = []
    lines.append("# 核心高管连增臻选 Stage1 基线")
    lines.append("")
    lines.append("## 样本概览")
    lines.append(
        f"- 扫描区间：`{summary_payload['range']['start_date']} ~ {summary_payload['range']['end_date']}`"
    )
    lines.append(f"- 扫描交易日：`{summary_payload['range']['trade_days_scanned']}`")
    lines.append(f"- 候选交易日：`{summary_payload['candidate_trade_day_count']}`")
    lines.append(f"- Stage1 总样本：`{stage1_summary.get('rows', 0)}`")
    lines.append(f"- Stage1 唯一股票：`{summary_payload.get('stage1_unique_stock_count', 0)}`")
    lines.append("")
    lines.append("## 收益总表")
    lines.append("| 样本 | 3日均值 | 5日均值 | 10日均值 | 3日胜率 | 5日胜率 | 10日胜率 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| Stage1 | {stage1_summary.get('avg_3d_pct')} | {stage1_summary.get('avg_5d_pct')} | "
        f"{stage1_summary.get('avg_10d_pct')} | {stage1_summary.get('win_rate_3d_pct')}% | "
        f"{stage1_summary.get('win_rate_5d_pct')}% | {stage1_summary.get('win_rate_10d_pct')}% |"
    )
    lines.append("")
    lines.append("## 重复分布")
    lines.append(
        f"- 重复股票行数：`{duplicate_payload.get('duplicate_row_count', 0)}` / "
        f"重复股票数：`{duplicate_payload.get('duplicate_value_count', 0)}`"
    )
    top_duplicates = duplicate_payload.get("top_duplicates", [])
    if top_duplicates:
        for item in top_duplicates[:10]:
            lines.append(f"- `{item.get('value')}` 出现 `{item.get('count')}` 次")
    else:
        lines.append("- 当前没有重复股票。")
    lines.append("")
    lines.append("## 最近样本")
    if stage1_df.empty:
        lines.append("- 当前没有 Stage1 样本。")
    else:
        lines.append("| 日期 | 股票 | 分数 | 3日 | 5日 | 10日 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for _, row in stage1_df.tail(20).iterrows():
            lines.append(
                f"| {row.get('signal_date')} | {row.get('ts_code')} {row.get('name', '')} | "
                f"{row.get('preliminary_score')} | {row.get('return_open_to_close_3d_pct')} | "
                f"{row.get('return_open_to_close_5d_pct')} | {row.get('return_open_to_close_10d_pct')} |"
            )
    if not progress_df.empty and "stage1_rows" in progress_df.columns:
        lines.append("")
        lines.append("## 每日 Stage1 行数")
        lines.append("| 日期 | Stage1 行数 |")
        lines.append("| --- | ---: |")
        for _, row in progress_df.iterrows():
            lines.append(f"| {row.get('trade_date')} | {row.get('stage1_rows')} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    review.ensure_token(token)

    config_overrides = review.load_config_overrides(args)
    hold_days = review.parse_hold_days(args.hold_days)
    export_dir = export_root_dir(args.export_root) / f"core_management_stage1_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    trade_cal = review.safe_call(
        "core_mgmt_trade_cal_range",
        getattr(review.configure_tushare_client(token, custom_http_url=custom_http_url), "trade_cal", None),
        sleep_sec=0.0,
        start_date=args.start_date,
        end_date=args.end_date,
        exchange="SSE",
    )
    if trade_cal.empty:
        raise SystemExit("Failed to load trade calendar for requested range.")
    trade_cal["cal_date"] = trade_cal["cal_date"].astype(str)
    trade_days = sorted(
        trade_cal.loc[
            pd.to_numeric(trade_cal["is_open"], errors="coerce").fillna(0).astype(int) == 1,
            "cal_date",
        ].tolist()
    )
    if not trade_days:
        raise SystemExit("No open trade dates found in requested range.")

    pro = review.configure_tushare_client(token, custom_http_url=custom_http_url)
    stock_basic_all = review.fetch_stock_basic_all(pro)
    price_history_start_date = (pd.Timestamp(trade_days[0]) - pd.Timedelta(days=max(420, 250 * 2))).strftime("%Y%m%d")

    price_bundle_cache: dict[str, dict[str, pd.DataFrame]] = {}
    stage1_signal_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []

    for idx, trade_date in enumerate(trade_days, start=1):
        review.log_step(f"stage1 baseline {idx}/{len(trade_days)} trade_date={trade_date}")
        try:
            progress_row, stage1_candidates, _ = review.evaluate_trade_date(
                pro=pro,
                trade_date=trade_date,
                config_overrides=config_overrides,
                api_sleep_sec=args.api_sleep_sec,
                stock_basic_all=stock_basic_all,
                price_bundle_cache=price_bundle_cache,
                price_history_start_date=price_history_start_date,
                global_end_date=args.end_date,
                recent_final_signals_df=pd.DataFrame(),
            )
        except Exception as exc:
            progress_rows.append(
                {
                    "trade_date": trade_date,
                    "latest_trade_date": None,
                    "stage1_rows": None,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        progress_rows.append(
            {
                "trade_date": progress_row.get("trade_date"),
                "latest_trade_date": progress_row.get("latest_trade_date"),
                "stage1_rows": progress_row.get("stage1_rows"),
                "status": progress_row.get("status"),
            }
        )
        if stage1_candidates.empty:
            continue

        latest_trade_date = str(progress_row.get("latest_trade_date") or "")
        for _, row in stage1_candidates.iterrows():
            row_dict = row.to_dict()
            ts_code = str(row_dict.get("ts_code") or "")
            if ts_code not in price_bundle_cache:
                price_bundle_cache[ts_code] = review.fetch_full_price_bundle(
                    pro=pro,
                    ts_code=ts_code,
                    start_date=price_history_start_date,
                    end_date=args.end_date,
                    sleep_sec=args.api_sleep_sec,
                )
            bundle = price_bundle_cache[ts_code]
            forward_payload = review.compute_forward_returns(bundle.get("daily_df", pd.DataFrame()), latest_trade_date, hold_days)
            stage1_signal_rows.append(
                {
                    "signal_date": latest_trade_date,
                    "screen_trade_date": trade_date,
                    "ts_code": ts_code,
                    "name": row_dict.get("name"),
                    "preliminary_score": row_dict.get("preliminary_score"),
                    "wave_signature": row_dict.get("wave_signature") or review.build_wave_signature(row_dict),
                    "wave_first_date": row_dict.get("wave_first_date"),
                    "wave_last_date": row_dict.get("wave_last_date"),
                    "wave_trade_days": row_dict.get("wave_trade_days"),
                    "wave_event_count": row_dict.get("wave_event_count"),
                    "wave_core_holder_count": row_dict.get("wave_core_holder_count"),
                    "wave_core_management_event_count": row_dict.get("wave_core_management_event_count"),
                    "wave_total_amount": row_dict.get("wave_total_amount"),
                    "current_to_cost_ratio": row_dict.get("current_to_cost_ratio"),
                    **forward_payload,
                }
            )

    stage1_df = pd.DataFrame(stage1_signal_rows)
    progress_df = pd.DataFrame(progress_rows)
    candidate_trade_days = sorted(stage1_df["screen_trade_date"].dropna().astype(str).unique().tolist()) if not stage1_df.empty else []
    summary_payload = {
        "strategy_id": "core_management_accumulation",
        "strategy_name": review.STRATEGY_NAME,
        "range": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "trade_days_scanned": len(trade_days),
        },
        "config_overrides": config_overrides,
        "hold_days": hold_days,
        "candidate_trade_day_count": len(candidate_trade_days),
        "candidate_trade_days": candidate_trade_days,
        "stage1_summary": review.summarize_returns("stage1_baseline", stage1_df, hold_days),
        "stage1_unique_stock_count": int(stage1_df["ts_code"].dropna().astype(str).nunique()) if not stage1_df.empty else 0,
        "stage1_ts_code_duplicates": review.duplicate_summary(stage1_df, "ts_code"),
        "stage1_wave_signature_duplicates": review.duplicate_summary(stage1_df, "wave_signature"),
        "export_dir": str(export_dir),
    }

    (export_dir / "stage1_summary.json").write_text(
        json.dumps(json_safe(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage1_df.to_csv(export_dir / "stage1_signals.csv", index=False)
    progress_df.to_csv(export_dir / "progress.csv", index=False)
    (export_dir / "stage1_report.md").write_text(
        build_stage1_report(summary_payload, stage1_df, progress_df),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary_payload), ensure_ascii=False))
    print(f"export_dir={export_dir}")
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
