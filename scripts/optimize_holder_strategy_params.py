from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from holder_strategy_core import (
    STRATEGY_NAME,
    HolderStrategyConfig,
    apply_holder_stage1,
    apply_holder_stage2,
    json_safe,
)
from research_backtest_utils import (
    discover_cached_trade_dates,
    load_cached_market_daily_history,
    log_step,
    repo_root_dir,
)
from strategy_exit_rules import apply_exit_rules, build_price_path_map

warnings.filterwarnings("ignore", category=FutureWarning, message="Downcasting behavior in Series and DataFrame methods.*")


PARAM_SPACE: dict[str, list[Any]] = {
    "min_volume_ratio": [1.1, 1.2, 1.3],
    "max_price_position": [0.40, 0.45, 0.50],
    "max_industry_pb_pct": [0.65, 0.70, 0.75],
    "min_final_score": [58.0, 60.0, 62.0],
    "min_aggressive_score": [50.0, 52.0, 54.0],
    "top_n_stage1": [8, 10],
    "top_n_final": [3, 5],
    "enable_stage2_cyq": [True],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize holder-strategy thresholds from existing export snapshots without touching live/test environment.")
    parser.add_argument("--start-date", required=True, help="Replay start date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Replay end date YYYYMMDD.")
    parser.add_argument("--min-trades", type=int, default=3, help="Soft minimum trade count.")
    parser.add_argument("--show-top", type=int, default=20, help="Rows to print from ranked trials.")
    parser.add_argument(
        "--snapshot-root",
        default="",
        help="Optional directory containing holder_increase_screen_<date> snapshots. Defaults to notebook export path.",
    )
    return parser.parse_args()


def export_root_dir() -> Path:
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def holder_export_root_dir(path_str: str = "") -> Path:
    if path_str.strip():
        return Path(path_str).expanduser().resolve()
    return repo_root_dir() / "output" / "jupyter-notebook" / "tushare_screen_exports"


def load_export_snapshots(start_date: str, end_date: str, snapshot_root: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(holder_export_root_dir(snapshot_root).glob("holder_increase_screen_*")):
        trade_date = path.name.rsplit("_", 1)[-1]
        if not (len(trade_date) == 8 and start_date <= trade_date <= end_date):
            continue
        candidate_base_path = path / "candidate_base.csv"
        deep_metrics_path = path / "deep_metrics_stage1.csv"
        stage2_cyq_path = path / "stage2_cyq_metrics.csv"
        if not candidate_base_path.exists() or not deep_metrics_path.exists():
            continue
        candidate_base = pd.read_csv(candidate_base_path)
        deep_metrics_stage1 = pd.read_csv(deep_metrics_path)
        stage2_cyq_metrics = pd.read_csv(stage2_cyq_path) if stage2_cyq_path.exists() else pd.DataFrame()
        market_regime = str(candidate_base["market_regime"].dropna().astype(str).iloc[0]) if "market_regime" in candidate_base.columns and not candidate_base["market_regime"].dropna().empty else "neutral"
        rows.append(
            {
                "trade_date": trade_date,
                "export_dir": path,
                "candidate_base": candidate_base,
                "deep_metrics_stage1": deep_metrics_stage1,
                "stage2_cyq_metrics": stage2_cyq_metrics,
                "market_regime": market_regime,
            }
        )
    return rows


def load_price_context(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    cached_dates = discover_cached_trade_dates(start_date, end_date)
    if not cached_dates:
        return pd.DataFrame(), pd.DataFrame(), {}
    history = load_cached_market_daily_history(cached_dates)
    if history.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    history = history.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    forward = history.copy()
    for column in ["open", "high", "low", "close"]:
        forward[column] = pd.to_numeric(forward[column], errors="coerce")
    grouped = forward.groupby("ts_code", sort=False)
    forward["entry_trade_date"] = grouped["trade_date"].shift(-1)
    forward["entry_open"] = grouped["open"].shift(-1)
    keep_cols = ["ts_code", "trade_date", "entry_trade_date", "entry_open"]
    price_path_map = build_price_path_map(history)
    return history, forward[keep_cols].copy(), price_path_map


def evaluate_daily_results(daily_results: pd.DataFrame, min_trades: int) -> dict[str, Any]:
    empty_metrics = {
        "selection_score": -999.0,
        "signal_days": 0,
        "filled_trades": 0,
        "avg_exit_return_pct": None,
        "median_exit_return_pct": None,
        "exit_win_rate_pct": None,
        "max_drawdown_pct": None,
        "avg_exit_hold_days": None,
        "positive_signal_ratio_pct": None,
    }
    if daily_results.empty:
        return empty_metrics
    frame = daily_results[daily_results["has_signal"]].copy()
    frame["exit_return_pct"] = pd.to_numeric(frame.get("exit_return_pct"), errors="coerce")
    frame["exit_hold_days"] = pd.to_numeric(frame.get("exit_hold_days"), errors="coerce")
    valid = frame.dropna(subset=["exit_return_pct"]).copy()
    if valid.empty:
        metrics = dict(empty_metrics)
        metrics["signal_days"] = int(len(frame))
        return metrics

    returns = valid["exit_return_pct"]
    equity = (1.0 + returns / 100.0).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = abs(float(drawdown.min()) * 100.0) if not drawdown.empty else None
    avg_return = float(returns.mean())
    win_rate = float((returns > 0).mean() * 100.0)
    avg_hold = float(valid["exit_hold_days"].dropna().mean()) if not valid["exit_hold_days"].dropna().empty else None
    positive_ratio = float((returns > 0).mean() * 100.0)
    trades_penalty = max(min_trades - len(valid), 0) * 4.0
    selection_score = (
        avg_return * 5.0
        + win_rate * 0.28
        + len(valid) * 1.0
        - (max_drawdown or 0.0) * 0.55
        - trades_penalty
    )
    return {
        "selection_score": round(float(selection_score), 4),
        "signal_days": int(len(frame)),
        "filled_trades": int(len(valid)),
        "avg_exit_return_pct": round(avg_return, 4),
        "median_exit_return_pct": round(float(returns.median()), 4),
        "exit_win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_drawdown, 4) if max_drawdown is not None else None,
        "avg_exit_hold_days": round(avg_hold, 2) if avg_hold is not None else None,
        "positive_signal_ratio_pct": round(positive_ratio, 2),
    }


def simulate_trial(
    snapshots: list[dict[str, Any]],
    forward_table: pd.DataFrame,
    price_path_map: dict[str, pd.DataFrame],
    config_overrides: dict[str, Any],
    min_trades: int,
) -> dict[str, Any]:
    daily_rows: list[dict[str, Any]] = []
    merge_cols = ["ts_code", "trade_date"]
    forward_lookup = forward_table.copy()
    if not forward_lookup.empty:
        forward_lookup["trade_date"] = forward_lookup["trade_date"].astype(str)
        forward_lookup["ts_code"] = forward_lookup["ts_code"].astype(str)
    for snapshot in snapshots:
        trade_date = snapshot["trade_date"]
        cfg = HolderStrategyConfig.for_end_date(trade_date, **config_overrides)
        candidate_base = snapshot["candidate_base"]
        deep_metrics_stage1 = snapshot["deep_metrics_stage1"]
        stage2_cyq_metrics = snapshot["stage2_cyq_metrics"] if cfg.enable_stage2_cyq else pd.DataFrame()
        market_regime = snapshot["market_regime"]

        stage1_result = apply_holder_stage1(candidate_base, deep_metrics_stage1, cfg, market_regime)
        stage2_result = apply_holder_stage2(stage1_result["ranked_candidates_stage1"], stage2_cyq_metrics, cfg, market_regime)
        best = stage2_result["best_pick_candidate"].head(1).copy()
        if best.empty:
            daily_rows.append(
                {
                    "trade_date": trade_date,
                    "strategy_id": "holder_increase",
                    "strategy_name": STRATEGY_NAME,
                    "market_regime": market_regime,
                    "has_signal": False,
                }
            )
            continue

        row = best.iloc[0].to_dict()
        row.update(
            {
                "trade_date": trade_date,
                "strategy_id": "holder_increase",
                "strategy_name": STRATEGY_NAME,
                "has_signal": True,
            }
        )
        daily_rows.append(row)

    daily_results = pd.DataFrame(daily_rows)
    if not daily_results.empty:
        daily_results["trade_date"] = daily_results["trade_date"].astype(str)
        if not forward_lookup.empty and {"ts_code", "trade_date"}.issubset(daily_results.columns):
            daily_results["ts_code"] = daily_results["ts_code"].astype(str)
            daily_results = daily_results.merge(forward_lookup, on=merge_cols, how="left")
        daily_results = apply_exit_rules(daily_results, price_path_map)
    metrics = evaluate_daily_results(daily_results, min_trades=min_trades)
    return {
        "daily_results": daily_results,
        **metrics,
    }


def main() -> None:
    args = parse_args()
    snapshots = load_export_snapshots(args.start_date, args.end_date, snapshot_root=args.snapshot_root)
    if not snapshots:
        raise SystemExit("No holder export snapshots found in the requested range.")

    available_trade_dates = discover_cached_trade_dates(args.start_date, "20991231")
    forward_end_date = available_trade_dates[-1] if available_trade_dates else args.end_date
    _, forward_table, price_path_map = load_price_context(args.start_date, forward_end_date)
    if forward_table.empty:
        raise SystemExit("No cached market daily history available for replay evaluation.")

    param_keys = list(PARAM_SPACE.keys())
    trial_rows: list[dict[str, Any]] = []

    all_configs = itertools.product(*(PARAM_SPACE[key] for key in param_keys))
    for idx, values in enumerate(all_configs, start=1):
        if idx == 1 or idx % 100 == 0:
            log_step(f"holder export replay trial {idx}")
        config_overrides = dict(zip(param_keys, values))
        if int(config_overrides["top_n_final"]) > int(config_overrides["top_n_stage1"]):
            continue
        result = simulate_trial(
            snapshots=snapshots,
            forward_table=forward_table,
            price_path_map=price_path_map,
            config_overrides=config_overrides,
            min_trades=args.min_trades,
        )
        row = {
            "trial_index": idx,
            "config_json": json.dumps(json_safe(config_overrides), ensure_ascii=False, sort_keys=True),
        }
        row.update({key: value for key, value in result.items() if key != "daily_results"})
        trial_rows.append(row)

    trial_results = pd.DataFrame(trial_rows).sort_values(
        by=["selection_score", "avg_exit_return_pct", "exit_win_rate_pct", "filled_trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    if trial_results.empty:
        raise SystemExit("No valid holder optimization trials were produced.")

    best_row = trial_results.iloc[0].to_dict()
    best_config = json.loads(best_row["config_json"])
    best_daily_results = simulate_trial(
        snapshots=snapshots,
        forward_table=forward_table,
        price_path_map=price_path_map,
        config_overrides=best_config,
        min_trades=args.min_trades,
    )["daily_results"]
    run_tag = f"optimize_holder_exports_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = export_root_dir() / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    trial_results.to_csv(export_dir / "trial_results.csv", index=False)
    best_daily_results.to_csv(export_dir / "best_daily_results.csv", index=False)

    summary = {
        "strategy_id": "holder_increase",
        "strategy_name": STRATEGY_NAME,
        "mode": "export_replay",
        "range": [args.start_date, args.end_date],
        "snapshot_count": int(len(snapshots)),
        "best_config": best_config,
        "best_selection_score": best_row.get("selection_score"),
        "best_avg_exit_return_pct": best_row.get("avg_exit_return_pct"),
        "best_exit_win_rate_pct": best_row.get("exit_win_rate_pct"),
        "best_filled_trades": best_row.get("filled_trades"),
        "export_dir": str(export_dir.resolve()),
    }
    with (export_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print("===== top trials =====")
    print(trial_results.head(args.show_top).to_string(index=False))
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
