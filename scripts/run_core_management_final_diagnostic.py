from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_core_management_final_review as review
from core_management_accumulation_strategy import (
    CoreManagementAccumulationConfig,
    _build_aux_flow_health_score,
    _build_continuity_score,
    _build_cost_zone_score,
    _build_freshness_penalty_score,
    _build_identity_strength_score,
    _build_retrigger_quality_score,
    build_final_candidate_flags,
    build_repeat_signal_state,
)
from research_backtest_utils import json_safe, repo_root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why stage1 candidates fail the light-final hard filters."
    )
    parser.add_argument("--trade-date", required=True, help="Trade date in YYYYMMDD.")
    parser.add_argument("--config-file", default="", help="Optional JSON config overrides.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON config overrides.")
    parser.add_argument("--export-root", default="", help="Optional export directory.")
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


def build_diagnostic_frame(
    merged: pd.DataFrame,
    config: CoreManagementAccumulationConfig,
) -> pd.DataFrame:
    work = merged.copy()
    if work.empty:
        return work
    work["identity_strength_score"] = work.apply(lambda row: _build_identity_strength_score(row.to_dict()), axis=1)
    work["continuity_score"] = work.apply(lambda row: _build_continuity_score(row.to_dict()), axis=1)
    work["cost_zone_score"] = work.apply(lambda row: _build_cost_zone_score(row.to_dict()), axis=1)
    work["aux_flow_health_score"] = work.apply(lambda row: _build_aux_flow_health_score(row.to_dict()), axis=1)
    work["post_wave_structure_score"] = pd.to_numeric(work.get("post_wave_structure_score"), errors="coerce").fillna(0.0).round(2)
    repeat_state = work.apply(
        lambda row: build_repeat_signal_state(row.to_dict(), config, recent_final_signals=None),
        axis=1,
        result_type="expand",
    )
    work = pd.concat([work, repeat_state], axis=1)
    work["freshness_penalty_score"] = work.apply(lambda row: _build_freshness_penalty_score(row.to_dict(), config), axis=1)
    work["retrigger_quality_score"] = work.apply(lambda row: _build_retrigger_quality_score(row.to_dict(), config), axis=1)
    work["final_confirmation_score"] = (
        pd.to_numeric(work["retrigger_quality_score"], errors="coerce").fillna(0.0)
        + np.where(
            pd.to_numeric(work["post_wave_structure_score"], errors="coerce").fillna(0.0) >= config.min_retrigger_structure_score,
            1.5,
            0.0,
        )
    ).round(2)
    work["base_total_score"] = (
        pd.to_numeric(work["identity_strength_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["continuity_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["aux_flow_health_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["cost_zone_score"], errors="coerce").fillna(0.0) * 0.4
        + pd.to_numeric(work["post_wave_structure_score"], errors="coerce").fillna(0.0) * 1.1
        + pd.to_numeric(work["final_confirmation_score"], errors="coerce").fillna(0.0) * 1.2
    ).round(2)
    work["adjusted_total_score"] = (
        pd.to_numeric(work["base_total_score"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["freshness_penalty_score"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["repeat_penalty_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["retrigger_quality_score"], errors="coerce").fillna(0.0)
    ).round(2)
    flags = work.apply(lambda row: build_final_candidate_flags(row.to_dict(), config), axis=1, result_type="expand")
    work = pd.concat([work, flags], axis=1)
    failure_columns = ["avg_amount_ok", "structure_ok", "freshness_ok", "repeat_ok", "confirmation_ok"]

    def _failure_reasons(row: pd.Series) -> str:
        reasons = [name for name in failure_columns if not bool(row.get(name))]
        score_ok = float(row.get("adjusted_total_score") or 0.0) >= float(config.min_total_score)
        if not score_ok:
            reasons.append("min_total_score")
        return ",".join(reasons)

    work["failure_reasons"] = work.apply(_failure_reasons, axis=1)
    return work


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    custom_http_url = os.getenv("TUSHARE_HTTP_URL", "http://lianghua.nanyangqiankun.top").strip()
    review.ensure_token(token)

    config_overrides = review.load_config_overrides(args)
    config = CoreManagementAccumulationConfig.for_end_date(args.trade_date, **config_overrides)
    export_dir = export_root_dir(args.export_root) / f"core_management_final_diagnostic_{args.trade_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    pro = review.configure_tushare_client(token, custom_http_url=custom_http_url)
    stock_basic_all = review.fetch_stock_basic_all(pro)
    price_history_start_date = (pd.Timestamp(args.trade_date) - pd.Timedelta(days=max(420, 250 * 2))).strftime("%Y%m%d")
    progress_row, stage1_candidates, _ = review.evaluate_trade_date(
        pro=pro,
        trade_date=args.trade_date,
        config_overrides=config_overrides,
        api_sleep_sec=args.api_sleep_sec,
        stock_basic_all=stock_basic_all,
        price_bundle_cache={},
        price_history_start_date=price_history_start_date,
        global_end_date=args.trade_date,
        recent_final_signals_df=pd.DataFrame(),
    )
    if stage1_candidates.empty:
        summary_payload = {
            "trade_date": args.trade_date,
            "config_overrides": config_overrides,
            "stage1_rows": 0,
            "final_pass_rows": 0,
            "export_dir": str(export_dir),
        }
        (export_dir / "diagnostic_summary.json").write_text(
            json.dumps(json_safe(summary_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(json_safe(summary_payload), ensure_ascii=False))
        print(f"export_dir={export_dir}")
        return

    price_bundle_cache: dict[str, dict[str, pd.DataFrame]] = {}
    deep_rows: list[dict[str, Any]] = []
    for _, row in stage1_candidates.iterrows():
        ts_code = str(row["ts_code"])
        if ts_code not in price_bundle_cache:
            price_bundle_cache[ts_code] = review.fetch_full_price_bundle(
                pro,
                ts_code=ts_code,
                start_date=price_history_start_date,
                end_date=args.trade_date,
                sleep_sec=args.api_sleep_sec,
            )
        bundle = price_bundle_cache[ts_code]
        deep_rows.append(
            {
                "ts_code": ts_code,
                **review.build_post_wave_structure_metrics(
                    daily_df=bundle["daily_df"],
                    adj_df=bundle["adj_df"],
                    end_date=str(progress_row.get("latest_trade_date") or args.trade_date),
                    wave_first_date=str(row.get("wave_first_date", "")),
                    wave_last_date=str(row.get("wave_last_date", "")),
                    weighted_cost=row.get("wave_buy_avg_price_weighted"),
                    config=config,
                ),
            }
        )
    merged = stage1_candidates.merge(pd.DataFrame(deep_rows), on="ts_code", how="left")
    diagnostic_df = build_diagnostic_frame(merged, config)
    passed_df = diagnostic_df[diagnostic_df["all_hard_filters_ok"].fillna(False)].copy()

    summary_payload = {
        "trade_date": args.trade_date,
        "latest_trade_date": progress_row.get("latest_trade_date"),
        "config_overrides": config_overrides,
        "stage1_rows": int(len(stage1_candidates)),
        "final_pass_rows": int(len(passed_df)),
        "failure_reason_counts": diagnostic_df["failure_reasons"].fillna("").astype(str).value_counts().head(20).to_dict(),
        "export_dir": str(export_dir),
    }
    diagnostic_df.to_csv(export_dir / "final_diagnostic.csv", index=False)
    passed_df.to_csv(export_dir / "final_pass_candidates.csv", index=False)
    (export_dir / "diagnostic_summary.json").write_text(
        json.dumps(json_safe(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary_payload), ensure_ascii=False))
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
