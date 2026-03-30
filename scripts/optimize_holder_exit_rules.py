from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from holder_replay_utils import build_holder_daily_results, export_root_dir, load_export_snapshots, load_price_context
from optimize_price_strategy_params import config_signature, evaluate_backtest_frame
from research_config_presets import exit_config_preset_help, load_exit_config_with_preset
from research_backtest_utils import json_safe, log_step


EXIT_PARAM_SPACE: dict[str, list[Any]] = {
    "common.intraday_conflict_mode": ["conservative", "nearest_open", "target_first"],
    "holder_increase.max_hold_days": [8, 10, 12, 15],
    "holder_increase.hard_stop_pct": [4.5, 5.5, 6.5, 7.5],
    "holder_increase.ma20_stop_buffer_pct": [0.8, 1.2, 1.6, 2.0],
    "holder_increase.breakeven_trigger_pct": [5.0, 7.0, 9.0],
    "holder_increase.breakeven_buffer_pct": [0.3, 0.5, 0.8],
    "holder_increase.trail_arm_pct": [9.0, 11.0, 12.5, 14.0],
    "holder_increase.trail_from_peak_pct": [3.8, 4.8, 5.8, 6.8],
    "holder_increase.trend_exit_arm_pct": [4.0, 5.0, 6.5],
    "holder_increase.trend_exit_min_hold_days": [2, 3, 4],
    "holder_increase.min_target_pct": [8.0, 10.0, 12.0, 15.0],
}

EXIT_COLUMNS = [
    "exit_trade_date",
    "exit_price",
    "exit_reason",
    "exit_rule",
    "exit_hold_days",
    "exit_return_pct",
    "exit_target_price",
    "exit_structure_stop",
    "exit_active_stop",
    "exit_peak_price",
    "exit_mfe_pct",
    "exit_mae_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize exit rules for the holder snapshot replay strategy.")
    parser.add_argument("--start-date", required=True, help="Training start date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Training end date YYYYMMDD.")
    parser.add_argument("--validation-start-date", default="", help="Optional validation start date YYYYMMDD.")
    parser.add_argument("--validation-end-date", default="", help="Optional validation end date YYYYMMDD.")
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
    parser.add_argument("--trials", type=int, default=80, help="Random-search trials, excluding the baseline exit config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-trades", type=int, default=3, help="Soft minimum trade count for training.")
    parser.add_argument("--validation-min-trades", type=int, default=2, help="Soft minimum trade count for validation.")
    parser.add_argument("--show-top", type=int, default=15, help="Rows to print from ranked results.")
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


def sample_exit_config(base_config: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    sampled = copy.deepcopy(base_config)
    for dotted_key, values in EXIT_PARAM_SPACE.items():
        set_nested_config_value(sampled, dotted_key, rng.choice(values))
    return sampled


def build_base_frame(
    snapshots: list[dict[str, Any]],
    config_overrides: dict[str, Any],
    hold_days: list[int],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    signal_trade_dates = [snapshot["trade_date"] for snapshot in snapshots]
    _, forward_table, price_path_map, _ = load_price_context(signal_start_date=signal_trade_dates[0], hold_days=hold_days)
    if forward_table.empty or not price_path_map:
        raise SystemExit("No cached market daily history available for holder replay.")
    base_frame = build_holder_daily_results(
        snapshots=snapshots,
        forward_table=forward_table,
        price_path_map=price_path_map,
        config_overrides=config_overrides,
        exit_config={},
        apply_exit=False,
    )
    drop_cols = [column for column in EXIT_COLUMNS if column in base_frame.columns]
    if drop_cols:
        base_frame = base_frame.drop(columns=drop_cols)
    return base_frame, price_path_map


def run_exit_trial(
    base_frame: pd.DataFrame,
    price_path_map: dict[str, pd.DataFrame],
    exit_config: dict[str, Any],
    min_trades: int,
) -> dict[str, Any]:
    from strategy_exit_rules import apply_exit_rules

    daily_results = apply_exit_rules(base_frame.copy(), price_path_map, config=exit_config)
    metrics = evaluate_backtest_frame(daily_results, regime_filter="", min_trades=min_trades)
    metrics["raw_daily_results"] = daily_results
    return metrics


def main() -> None:
    args = parse_args()
    hold_days = sorted({int(item.strip()) for item in args.hold_days.split(",") if item.strip()})
    config_overrides = load_config_overrides(args.config_file, args.config_json)
    base_exit_config, exit_config_meta = load_exit_config_with_preset(
        preset_name=args.exit_config_preset,
        config_file=args.exit_config_file,
    )
    rng = random.Random(args.seed)

    train_snapshots = load_export_snapshots(args.start_date, args.end_date, snapshot_root=args.snapshot_root)
    if not train_snapshots:
        raise SystemExit("No holder snapshots found in the training range.")
    train_base_frame, train_price_path_map = build_base_frame(train_snapshots, config_overrides, hold_days)

    valid_base_frame = None
    valid_price_path_map = None
    if args.validation_start_date and args.validation_end_date:
        valid_snapshots = load_export_snapshots(args.validation_start_date, args.validation_end_date, snapshot_root=args.snapshot_root)
        if valid_snapshots:
            valid_base_frame, valid_price_path_map = build_base_frame(valid_snapshots, config_overrides, hold_days)

    trial_configs: list[dict[str, Any]] = [copy.deepcopy(base_exit_config)]
    seen_signatures = {config_signature(base_exit_config)}
    max_attempts = max(args.trials * 25, 150)
    attempts = 0
    while len(trial_configs) < args.trials + 1 and attempts < max_attempts:
        attempts += 1
        sampled = sample_exit_config(base_exit_config, rng)
        signature = config_signature(sampled)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        trial_configs.append(sampled)

    rows: list[dict[str, Any]] = []
    for idx, exit_config in enumerate(trial_configs, start=1):
        log_step(f"holder exit trial {idx}/{len(trial_configs)}")
        train_metrics = run_exit_trial(
            base_frame=train_base_frame,
            price_path_map=train_price_path_map,
            exit_config=exit_config,
            min_trades=args.min_trades,
        )
        row = {
            "trial_index": idx - 1,
            "strategy_id": "holder_increase",
            "strategy_name": "星曜增持臻选",
            "exit_config_json": json.dumps(json_safe(exit_config), ensure_ascii=False, sort_keys=True),
            "is_baseline": idx == 1,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items() if key != "raw_daily_results"})

        if valid_base_frame is not None and valid_price_path_map is not None:
            valid_metrics = run_exit_trial(
                base_frame=valid_base_frame,
                price_path_map=valid_price_path_map,
                exit_config=exit_config,
                min_trades=args.validation_min_trades,
            )
            row.update({f"valid_{key}": value for key, value in valid_metrics.items() if key != "raw_daily_results"})
            row["selection_score"] = round(
                float(row.get("train_selection_score", -999.0)) * 0.40
                + float(row.get("valid_selection_score", -999.0)) * 0.60,
                4,
            )
        else:
            row["selection_score"] = row.get("train_selection_score", -999.0)
        rows.append(row)

    results = pd.DataFrame(rows)
    if not results.empty:
        for column in ["selection_score", "train_avg_exit_return_pct", "train_exit_win_rate_pct", "train_filled_trades"]:
            if column in results.columns:
                results[column] = pd.to_numeric(results[column], errors="coerce")
        results = results.sort_values(
            ["selection_score", "train_avg_exit_return_pct", "train_exit_win_rate_pct", "train_filled_trades"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    run_tag = f"optimize_exit_holder_increase_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = export_root_dir() / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(export_dir / "trial_results.csv", index=False)
    top_results = results.head(args.show_top).copy()
    top_results.to_csv(export_dir / "top_results.csv", index=False)

    summary = {
        "strategy_id": "holder_increase",
        "strategy_name": "星曜增持臻选",
        "train_range": [args.start_date, args.end_date],
        "validation_range": [args.validation_start_date, args.validation_end_date] if valid_base_frame is not None else None,
        "hold_days": hold_days,
        "config_overrides": json_safe(config_overrides),
        "trials_requested": args.trials,
        "trials_completed": int(len(results)),
        "exit_config_preset": exit_config_meta["exit_config_preset"],
        "exit_config_file": exit_config_meta["exit_config_file"],
        "preset_exit_config_file": exit_config_meta["preset_exit_config_file"],
        "export_dir": str(export_dir.resolve()),
    }
    with (export_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print("===== top exit trials =====")
    print(top_results.to_string(index=False) if not top_results.empty else "(empty)")
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
