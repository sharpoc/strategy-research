from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from holder_replay_utils import build_holder_daily_results, export_root_dir, load_export_snapshots, load_price_context
from optimize_holder_strategy_params import PARAM_SPACE
from optimize_price_strategy_params import config_signature, evaluate_backtest_frame
from research_config_presets import exit_config_preset_help, load_exit_config_with_preset
from research_backtest_utils import json_safe, log_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for holder strategy snapshot replay.")
    parser.add_argument("--start-date", required=True, help="Overall replay start date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Overall replay end date YYYYMMDD.")
    parser.add_argument("--train-trade-days", type=int, default=4, help="Training snapshot days per fold.")
    parser.add_argument("--validation-trade-days", type=int, default=1, help="Validation snapshot days per fold.")
    parser.add_argument("--step-trade-days", type=int, default=1, help="Fold step in snapshot days.")
    parser.add_argument("--trials", type=int, default=40, help="Random-search trials per fold, excluding baseline.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hold-days", default="1,3,5,10", help="Comma-separated forward horizons for reporting.")
    parser.add_argument("--config-file", default="", help="Optional JSON file with HolderStrategyConfig override keys.")
    parser.add_argument("--config-json", default="", help="Optional inline JSON object with HolderStrategyConfig override keys.")
    parser.add_argument(
        "--exit-config-preset",
        default="",
        help=f"Named exit preset. Available: {exit_config_preset_help()}",
    )
    parser.add_argument("--exit-config-file", default="", help="Optional JSON sell-rule override file.")
    parser.add_argument("--snapshot-root", default="", help="Optional root directory containing holder_increase_screen_<date> snapshots.")
    parser.add_argument("--min-trades", type=int, default=3, help="Soft minimum trade count in training folds.")
    parser.add_argument("--validation-min-trades", type=int, default=1, help="Soft minimum trade count in validation folds.")
    parser.add_argument("--max-folds", type=int, default=0, help="Limit fold count for smoke tests.")
    parser.add_argument("--show-top", type=int, default=10, help="Rows to print from fold results.")
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


def load_config_overrides(config_file: str, config_json: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if config_file:
        config.update(load_json_file(config_file))
    if config_json.strip():
        try:
            inline = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --config-json: {exc}") from exc
        if not isinstance(inline, dict):
            raise SystemExit("--config-json must be a JSON object.")
        config.update(inline)
    return config


def set_nested_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    if "." not in dotted_key:
        config[dotted_key] = value
        return
    current = config
    parts = [part for part in dotted_key.split(".") if part]
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value


def sample_holder_config(base_config: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    sampled = copy.deepcopy(base_config)
    for key, values in PARAM_SPACE.items():
        set_nested_config_value(sampled, key, rng.choice(values))
    return sampled


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
    for column in [
        "best_valid_avg_exit_return_pct",
        "best_valid_exit_win_rate_pct",
        "best_valid_filled_trades",
        "baseline_valid_avg_exit_return_pct",
        "baseline_valid_exit_win_rate_pct",
        "baseline_valid_filled_trades",
    ]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    beat_baseline = (
        work["best_valid_avg_exit_return_pct"] > work["baseline_valid_avg_exit_return_pct"]
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
        "avg_best_valid_win_rate_pct": round(float(work["best_valid_exit_win_rate_pct"].mean()), 2)
        if "best_valid_exit_win_rate_pct" in work.columns and not work["best_valid_exit_win_rate_pct"].dropna().empty
        else None,
        "total_best_valid_trades": int(work["best_valid_filled_trades"].fillna(0).sum()) if "best_valid_filled_trades" in work.columns else 0,
        "beat_baseline_ratio_pct": round(float(beat_baseline.mean() * 100.0), 2) if not beat_baseline.empty else None,
    }
    return pd.DataFrame([summary])


def subset_snapshots(snapshots: list[dict[str, Any]], start_date: str, end_date: str) -> list[dict[str, Any]]:
    return [snapshot for snapshot in snapshots if start_date <= snapshot["trade_date"] <= end_date]


def run_snapshot_trial(
    snapshots: list[dict[str, Any]],
    forward_table: pd.DataFrame,
    price_path_map: dict[str, pd.DataFrame],
    config_overrides: dict[str, Any],
    exit_config: dict[str, Any],
    min_trades: int,
) -> dict[str, Any]:
    daily_results = build_holder_daily_results(
        snapshots=snapshots,
        forward_table=forward_table,
        price_path_map=price_path_map,
        config_overrides=config_overrides,
        exit_config=exit_config,
        apply_exit=True,
    )
    metrics = evaluate_backtest_frame(daily_results, regime_filter="", min_trades=min_trades)
    metrics["daily_results"] = daily_results
    return metrics


def main() -> None:
    args = parse_args()
    hold_days = sorted({int(item.strip()) for item in args.hold_days.split(",") if item.strip()})
    base_config = load_config_overrides(args.config_file, args.config_json)
    exit_config, exit_config_meta = load_exit_config_with_preset(
        preset_name=args.exit_config_preset,
        config_file=args.exit_config_file,
    )
    rng = random.Random(args.seed)

    all_snapshots = load_export_snapshots(args.start_date, args.end_date, snapshot_root=args.snapshot_root)
    if not all_snapshots:
        raise SystemExit("No holder snapshots found in the requested range.")
    signal_trade_dates = [snapshot["trade_date"] for snapshot in all_snapshots]
    _, forward_table, price_path_map, _ = load_price_context(signal_start_date=signal_trade_dates[0], hold_days=hold_days)
    if forward_table.empty or not price_path_map:
        raise SystemExit("No cached market daily history available for holder replay.")

    folds = build_fold_ranges(signal_trade_dates, args.train_trade_days, args.validation_trade_days, args.step_trade_days)
    if args.max_folds > 0:
        folds = folds[: args.max_folds]
    if not folds:
        raise SystemExit("No valid walk-forward folds for the requested holder snapshot range.")

    fold_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_index = int(fold["fold_index"])
        log_step(f"holder walk-forward fold {fold_index + 1}/{len(folds)}")
        train_snapshots = subset_snapshots(all_snapshots, fold["train_start"], fold["train_end"])
        valid_snapshots = subset_snapshots(all_snapshots, fold["valid_start"], fold["valid_end"])
        if not train_snapshots or not valid_snapshots:
            continue

        trial_configs: list[dict[str, Any]] = [dict(base_config)]
        seen_signatures = {config_signature(base_config)}
        max_attempts = max(args.trials * 20, 100)
        attempts = 0
        while len(trial_configs) < args.trials + 1 and attempts < max_attempts:
            attempts += 1
            sampled = sample_holder_config(base_config, random.Random(rng.randint(0, 10**9)))
            signature = config_signature(sampled)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            trial_configs.append(sampled)

        best_train_row: dict[str, Any] | None = None
        for trial_index, strategy_config in enumerate(trial_configs):
            metrics = run_snapshot_trial(
                snapshots=train_snapshots,
                forward_table=forward_table,
                price_path_map=price_path_map,
                config_overrides=strategy_config,
                exit_config=exit_config,
                min_trades=args.min_trades,
            )
            trial_row = {
                "fold_index": fold_index,
                "trial_index": trial_index,
                "is_baseline": trial_index == 0,
                "config_json": json.dumps(json_safe(strategy_config), ensure_ascii=False, sort_keys=True),
            }
            trial_row.update({f"train_{key}": value for key, value in metrics.items() if key != "daily_results"})
            trial_row["fold_valid_start"] = fold["valid_start"]
            trial_row["fold_valid_end"] = fold["valid_end"]
            trial_rows.append(trial_row)
            if best_train_row is None or float(trial_row.get("train_selection_score") or -999.0) > float(best_train_row.get("train_selection_score") or -999.0):
                best_train_row = trial_row

        if best_train_row is None:
            continue

        best_config = json.loads(best_train_row["config_json"])
        best_valid_metrics = run_snapshot_trial(
            snapshots=valid_snapshots,
            forward_table=forward_table,
            price_path_map=price_path_map,
            config_overrides=best_config,
            exit_config=exit_config,
            min_trades=args.validation_min_trades,
        )
        baseline_valid_metrics = run_snapshot_trial(
            snapshots=valid_snapshots,
            forward_table=forward_table,
            price_path_map=price_path_map,
            config_overrides=dict(base_config),
            exit_config=exit_config,
            min_trades=args.validation_min_trades,
        )

        fold_row = {
            "fold_index": fold_index,
            "strategy_id": "holder_increase",
            "strategy_name": "星曜增持臻选",
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

    fold_results = pd.DataFrame(fold_rows)
    trial_results = pd.DataFrame(trial_rows)
    aggregate_summary = aggregate_walkforward_results(fold_results)

    run_tag = f"walkforward_holder_increase_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = export_root_dir() / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(export_dir / "fold_results.csv", index=False)
    trial_results.to_csv(export_dir / "fold_trial_results.csv", index=False)
    aggregate_summary.to_csv(export_dir / "aggregate_summary.csv", index=False)

    summary = {
        "strategy_id": "holder_increase",
        "strategy_name": "星曜增持臻选",
        "range": [args.start_date, args.end_date],
        "train_trade_days": args.train_trade_days,
        "validation_trade_days": args.validation_trade_days,
        "step_trade_days": args.step_trade_days,
        "trials": args.trials,
        "folds": int(len(fold_results)),
        "hold_days": hold_days,
        "config_overrides": json_safe(base_config),
        "exit_config_preset": exit_config_meta["exit_config_preset"],
        "exit_config_file": exit_config_meta["exit_config_file"],
        "preset_exit_config_file": exit_config_meta["preset_exit_config_file"],
        "export_dir": str(export_dir.resolve()),
    }
    with (export_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print("===== fold results =====")
    print(fold_results.head(args.show_top).to_string(index=False) if not fold_results.empty else "(empty)")
    print("===== aggregate summary =====")
    print(aggregate_summary.to_string(index=False) if not aggregate_summary.empty else "(empty)")
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
