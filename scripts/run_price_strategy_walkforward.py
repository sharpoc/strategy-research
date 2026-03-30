from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_strategy_registry import build_price_strategy_registry
from optimize_price_strategy_params import (
    PARAM_SPACE,
    config_signature,
    evaluate_backtest_frame,
    export_root_dir,
    sample_strategy_config,
)
from research_config_presets import (
    exit_config_preset_help,
    load_exit_config_with_preset,
    load_json_file,
    load_strategy_overrides_with_preset,
    strategy_config_preset_help,
)
from research_backtest_utils import json_safe, log_step
from run_price_strategy_regime_backtest import (
    prepare_backtest_dataset,
    run_backtest_on_dataset,
    slice_backtest_dataset,
)


STRATEGY_REGISTRY = build_price_strategy_registry()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for one price strategy using isolated local research datasets.")
    parser.add_argument("--strategy-id", required=True, choices=sorted(PARAM_SPACE.keys()))
    parser.add_argument("--start-date", required=True, help="Overall signal start date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Overall signal end date YYYYMMDD.")
    parser.add_argument("--train-trade-days", type=int, default=60, help="Training trade days per fold.")
    parser.add_argument("--validation-trade-days", type=int, default=20, help="Validation trade days per fold.")
    parser.add_argument("--step-trade-days", type=int, default=20, help="Fold step in trade days.")
    parser.add_argument("--trials", type=int, default=40, help="Random-search trials per fold, excluding baseline.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hold-days", default="1,3,5,10", help="Comma-separated forward horizons for reporting.")
    parser.add_argument("--regime-filter", default="", choices=["", "上涨趋势", "下跌趋势", "震荡趋势"], help="Optional market regime filter.")
    parser.add_argument(
        "--strategy-config-preset",
        default="",
        help=f"Named research preset for strategy overrides. Available: {strategy_config_preset_help()}",
    )
    parser.add_argument("--strategy-config-file", default="", help="JSON base overrides before random sampling.")
    parser.add_argument(
        "--exit-config-preset",
        default="",
        help=f"Named exit preset. Available: {exit_config_preset_help()}",
    )
    parser.add_argument("--exit-config-file", default="", help="JSON sell-rule overrides.")
    parser.add_argument("--min-trades", type=int, default=8, help="Soft minimum trade count in training folds.")
    parser.add_argument("--validation-min-trades", type=int, default=3, help="Soft minimum trade count in validation folds.")
    parser.add_argument("--max-folds", type=int, default=0, help="Limit walk-forward folds for smoke tests.")
    parser.add_argument("--future-padding-days", type=int, default=45, help="Calendar days added after end date to support future exit evaluation.")
    parser.add_argument("--show-top", type=int, default=10, help="Rows to print from fold results.")
    return parser.parse_args()


def _shift_calendar_date(date_str: str, calendar_days: int) -> str:
    return (pd.Timestamp(date_str) + pd.Timedelta(days=int(calendar_days))).strftime("%Y%m%d")


def build_trial_configs(strategy_id: str, base_config: dict[str, Any], trials: int, seed: int, fold_index: int) -> list[dict[str, Any]]:
    rng = __import__("random").Random(seed + fold_index * 1009)
    configs: list[dict[str, Any]] = [dict(base_config)]
    seen = {config_signature(base_config)}
    attempts = 0
    max_attempts = max(trials * 20, 100)
    while len(configs) < trials + 1 and attempts < max_attempts:
        attempts += 1
        sampled = sample_strategy_config(strategy_id, rng, base_config)
        signature = config_signature(sampled)
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(sampled)
    return configs


def run_dataset_trial(
    dataset,
    strategy_id: str,
    hold_days: list[int],
    strategy_config: dict[str, Any],
    regime_filter: str,
    min_trades: int,
    exit_config: dict[str, Any],
) -> dict[str, Any]:
    result = run_backtest_on_dataset(
        dataset=dataset,
        strategy_ids=[strategy_id],
        hold_days=hold_days,
        strategy_overrides={strategy_id: strategy_config},
        exit_config=exit_config,
        export_results=False,
        use_signal_cache=False,
    )
    metrics = evaluate_backtest_frame(result["daily_results"], regime_filter=regime_filter, min_trades=min_trades)
    metrics["daily_results"] = result["daily_results"]
    return metrics


def build_fold_ranges(trade_dates: list[str], train_trade_days: int, validation_trade_days: int, step_trade_days: int) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    if len(trade_dates) < train_trade_days + validation_trade_days:
        return folds
    fold_index = 0
    train_end_idx = train_trade_days - 1
    while train_end_idx + validation_trade_days < len(trade_dates):
        valid_start_idx = train_end_idx + 1
        valid_end_idx = train_end_idx + validation_trade_days
        folds.append(
            {
                "fold_index": fold_index,
                "train_start": trade_dates[train_end_idx - train_trade_days + 1],
                "train_end": trade_dates[train_end_idx],
                "valid_start": trade_dates[valid_start_idx],
                "valid_end": trade_dates[valid_end_idx],
            }
        )
        fold_index += 1
        train_end_idx += step_trade_days
    return folds


def aggregate_walkforward_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    if fold_results.empty:
        return pd.DataFrame()
    work = fold_results.copy()
    numeric_cols = [
        "best_valid_avg_exit_return_pct",
        "best_valid_exit_win_rate_pct",
        "best_valid_filled_trades",
        "best_valid_max_drawdown_pct",
        "baseline_valid_avg_exit_return_pct",
        "baseline_valid_exit_win_rate_pct",
        "baseline_valid_filled_trades",
    ]
    for column in numeric_cols:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    beat_baseline = (
        (work["best_valid_avg_exit_return_pct"] > work["baseline_valid_avg_exit_return_pct"])
        if {"best_valid_avg_exit_return_pct", "baseline_valid_avg_exit_return_pct"}.issubset(work.columns)
        else pd.Series(dtype=bool)
    )
    summary = {
        "fold_count": int(len(work)),
        "positive_valid_fold_ratio_pct": round(float((work["best_valid_avg_exit_return_pct"] > 0).mean() * 100.0), 2)
        if "best_valid_avg_exit_return_pct" in work.columns and not work["best_valid_avg_exit_return_pct"].dropna().empty
        else None,
        "avg_best_valid_return_pct": round(float(work["best_valid_avg_exit_return_pct"].mean()), 4)
        if "best_valid_avg_exit_return_pct" in work.columns and not work["best_valid_avg_exit_return_pct"].dropna().empty
        else None,
        "median_best_valid_return_pct": round(float(work["best_valid_avg_exit_return_pct"].median()), 4)
        if "best_valid_avg_exit_return_pct" in work.columns and not work["best_valid_avg_exit_return_pct"].dropna().empty
        else None,
        "avg_best_valid_win_rate_pct": round(float(work["best_valid_exit_win_rate_pct"].mean()), 2)
        if "best_valid_exit_win_rate_pct" in work.columns and not work["best_valid_exit_win_rate_pct"].dropna().empty
        else None,
        "total_best_valid_trades": int(work["best_valid_filled_trades"].fillna(0).sum()) if "best_valid_filled_trades" in work.columns else 0,
        "beat_baseline_ratio_pct": round(float(beat_baseline.mean() * 100.0), 2) if not beat_baseline.empty else None,
    }
    return pd.DataFrame([summary])


def main() -> None:
    args = parse_args()
    if args.strategy_id not in STRATEGY_REGISTRY:
        raise SystemExit(f"Unsupported strategy: {args.strategy_id}")

    hold_days = sorted({int(item.strip()) for item in args.hold_days.split(",") if item.strip()})
    base_overrides, strategy_config_meta = load_strategy_overrides_with_preset(
        preset_name=args.strategy_config_preset,
        config_file=args.strategy_config_file,
    )
    base_strategy_config = dict(base_overrides.get(args.strategy_id, {}))
    exit_config, exit_config_meta = load_exit_config_with_preset(
        preset_name=args.exit_config_preset,
        config_file=args.exit_config_file,
    )
    extended_end_date = _shift_calendar_date(args.end_date, args.future_padding_days)

    log_step("prepare master dataset")
    master_dataset = prepare_backtest_dataset(
        start_date=args.start_date,
        end_date=extended_end_date,
        strategy_ids=[args.strategy_id],
        hold_days=hold_days,
    )
    signal_trade_dates = [trade_date for trade_date in master_dataset.trade_dates if args.start_date <= trade_date <= args.end_date]
    folds = build_fold_ranges(
        trade_dates=signal_trade_dates,
        train_trade_days=args.train_trade_days,
        validation_trade_days=args.validation_trade_days,
        step_trade_days=args.step_trade_days,
    )
    if args.max_folds > 0:
        folds = folds[: args.max_folds]
    if not folds:
        raise SystemExit("No valid walk-forward folds for the requested range.")

    fold_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_index = int(fold["fold_index"])
        log_step(f"walk-forward fold {fold_index + 1}/{len(folds)}")
        train_dataset = slice_backtest_dataset(master_dataset, fold["train_start"], fold["train_end"])
        valid_dataset = slice_backtest_dataset(master_dataset, fold["valid_start"], fold["valid_end"])
        trial_configs = build_trial_configs(
            strategy_id=args.strategy_id,
            base_config=base_strategy_config,
            trials=args.trials,
            seed=args.seed,
            fold_index=fold_index,
        )

        train_rank_rows: list[dict[str, Any]] = []
        best_train_row: dict[str, Any] | None = None
        best_train_metrics: dict[str, Any] | None = None
        for trial_index, strategy_config in enumerate(trial_configs):
            metrics = run_dataset_trial(
                dataset=train_dataset,
                strategy_id=args.strategy_id,
                hold_days=hold_days,
                strategy_config=strategy_config,
                regime_filter=args.regime_filter,
                min_trades=args.min_trades,
                exit_config=exit_config,
            )
            trial_row = {
                "fold_index": fold_index,
                "trial_index": trial_index,
                "is_baseline": trial_index == 0,
                "config_json": json.dumps(json_safe(strategy_config), ensure_ascii=False, sort_keys=True),
            }
            trial_row.update({f"train_{key}": value for key, value in metrics.items() if key != "daily_results"})
            train_rank_rows.append(trial_row)
            if best_train_row is None or float(trial_row.get("train_selection_score") or -999.0) > float(best_train_row.get("train_selection_score") or -999.0):
                best_train_row = trial_row
                best_train_metrics = metrics

        if best_train_row is None or best_train_metrics is None:
            continue

        best_config = json.loads(best_train_row["config_json"])
        best_valid_metrics = run_dataset_trial(
            dataset=valid_dataset,
            strategy_id=args.strategy_id,
            hold_days=hold_days,
            strategy_config=best_config,
            regime_filter=args.regime_filter,
            min_trades=args.validation_min_trades,
            exit_config=exit_config,
        )
        baseline_valid_metrics = run_dataset_trial(
            dataset=valid_dataset,
            strategy_id=args.strategy_id,
            hold_days=hold_days,
            strategy_config=dict(base_strategy_config),
            regime_filter=args.regime_filter,
            min_trades=args.validation_min_trades,
            exit_config=exit_config,
        )

        fold_row = {
            "fold_index": fold_index,
            "strategy_id": args.strategy_id,
            "strategy_name": STRATEGY_REGISTRY[args.strategy_id].strategy_name,
            "regime_filter": args.regime_filter or "全部",
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "valid_start": fold["valid_start"],
            "valid_end": fold["valid_end"],
            "best_config_json": best_train_row["config_json"],
        }
        fold_row.update({f"best_{key}": value for key, value in best_train_row.items() if key.startswith("train_")})
        fold_row.update({f"best_valid_{key}": value for key, value in best_valid_metrics.items() if key != "daily_results"})
        fold_row.update({f"baseline_valid_{key}": value for key, value in baseline_valid_metrics.items() if key != "daily_results"})
        beat_baseline = (
            best_valid_metrics.get("avg_exit_return_pct") is not None
            and baseline_valid_metrics.get("avg_exit_return_pct") is not None
            and float(best_valid_metrics["avg_exit_return_pct"]) > float(baseline_valid_metrics["avg_exit_return_pct"])
        )
        fold_row["valid_beat_baseline"] = bool(beat_baseline)
        fold_rows.append(fold_row)

        for trial_row in train_rank_rows:
            trial_row = dict(trial_row)
            trial_row["fold_valid_start"] = fold["valid_start"]
            trial_row["fold_valid_end"] = fold["valid_end"]
            trial_rows.append(trial_row)

    fold_results = pd.DataFrame(fold_rows)
    trial_results = pd.DataFrame(trial_rows)
    aggregate_summary = aggregate_walkforward_results(fold_results)

    run_tag = f"walkforward_{args.strategy_id}_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = export_root_dir() / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(export_dir / "fold_results.csv", index=False)
    trial_results.to_csv(export_dir / "fold_trial_results.csv", index=False)
    aggregate_summary.to_csv(export_dir / "aggregate_summary.csv", index=False)

    summary = {
        "strategy_id": args.strategy_id,
        "strategy_name": STRATEGY_REGISTRY[args.strategy_id].strategy_name,
        "range": [args.start_date, args.end_date],
        "train_trade_days": args.train_trade_days,
        "validation_trade_days": args.validation_trade_days,
        "step_trade_days": args.step_trade_days,
        "fold_count": int(len(fold_results)),
        "trials_per_fold": args.trials + 1,
        "regime_filter": args.regime_filter or "全部",
        "hold_days": hold_days,
        "strategy_config_preset": strategy_config_meta["strategy_config_preset"],
        "strategy_config_file": strategy_config_meta["strategy_config_file"],
        "preset_strategy_config_file": strategy_config_meta["preset_strategy_config_file"],
        "exit_config_preset": exit_config_meta["exit_config_preset"],
        "exit_config_file": exit_config_meta["exit_config_file"],
        "preset_exit_config_file": exit_config_meta["preset_exit_config_file"],
        "export_dir": str(export_dir.resolve()),
    }
    with (export_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print("===== aggregate summary =====")
    print(aggregate_summary.to_string(index=False) if not aggregate_summary.empty else "(empty)")
    print("===== fold results =====")
    print(fold_results.head(args.show_top).to_string(index=False) if not fold_results.empty else "(empty)")
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
