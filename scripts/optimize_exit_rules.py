from __future__ import annotations

import argparse
import copy
import json
import random
from typing import Any

import pandas as pd

from backtest_strategy_registry import build_price_strategy_registry
from optimize_price_strategy_params import (
    config_signature,
    evaluate_backtest_frame,
    export_root_dir,
    set_nested_config_value,
)
from research_config_presets import (
    exit_config_preset_help,
    load_exit_config_with_preset,
    load_strategy_overrides_with_preset,
    strategy_config_preset_help,
)
from research_backtest_utils import json_safe, log_step
from run_price_strategy_regime_backtest import prepare_backtest_dataset, run_backtest_on_dataset
from strategy_exit_rules import apply_exit_rules


STRATEGY_REGISTRY = build_price_strategy_registry()

EXIT_PARAM_SPACE: dict[str, dict[str, list[Any]]] = {
    "limitup_l1l2": {
        "common.intraday_conflict_mode": ["conservative", "nearest_open", "target_first"],
        "limitup_l1l2.max_hold_days": [8, 10, 12, 15],
        "limitup_l1l2.stop_below_l2_pct": [0.8, 1.2, 1.6, 2.0],
        "limitup_l1l2.breakeven_trigger_pct": [5.0, 6.0, 7.5, 9.0],
        "limitup_l1l2.breakeven_buffer_pct": [0.3, 0.5, 0.8],
        "limitup_l1l2.trail_arm_pct": [8.0, 9.0, 10.5, 12.0],
        "limitup_l1l2.trail_from_peak_pct": [3.5, 4.5, 5.5, 6.5],
        "limitup_l1l2.trend_exit_arm_pct": [4.5, 5.5, 7.0, 8.5],
        "limitup_l1l2.trend_exit_min_hold_days": [2, 3, 4, 5],
        "limitup_l1l2.target_from_impulse_multiple": [0.6, 0.8, 1.0, 1.2],
        "limitup_l1l2.min_target_pct": [6.0, 8.0, 10.0, 12.0],
    },
    "platform_breakout": {
        "common.intraday_conflict_mode": ["conservative", "nearest_open", "target_first"],
        "platform_breakout.max_hold_days": [10, 12, 15],
        "platform_breakout.stop_below_support_pct": [1.0, 1.2, 1.6, 2.0],
        "platform_breakout.breakeven_trigger_pct": [6.0, 7.0, 8.5, 10.0],
        "platform_breakout.breakeven_buffer_pct": [0.4, 0.6, 0.8],
        "platform_breakout.trail_arm_pct": [9.0, 10.0, 11.5, 13.0],
        "platform_breakout.trail_from_peak_pct": [4.0, 5.0, 5.8, 6.5],
        "platform_breakout.trend_exit_arm_pct": [5.5, 6.5, 8.0],
        "platform_breakout.trend_exit_min_hold_days": [2, 3, 4, 5],
        "platform_breakout.target_from_pattern_multiple": [0.6, 0.8, 1.0, 1.2],
        "platform_breakout.min_target_pct": [8.0, 10.0, 12.0],
    },
    "double_bottom": {
        "common.intraday_conflict_mode": ["conservative", "nearest_open", "target_first"],
        "double_bottom.max_hold_days": [12, 15, 18],
        "double_bottom.bottom_stop_buffer_pct": [1.0, 1.5, 2.0, 2.5],
        "double_bottom.neckline_retest_break_pct": [1.5, 2.0, 2.5, 3.0],
        "double_bottom.breakeven_trigger_pct": [7.0, 8.0, 10.0],
        "double_bottom.breakeven_buffer_pct": [0.5, 0.8, 1.0],
        "double_bottom.trail_arm_pct": [10.0, 12.0, 14.0],
        "double_bottom.trail_from_peak_pct": [4.5, 5.5, 6.5],
        "double_bottom.trend_exit_arm_pct": [6.5, 7.5, 9.0],
        "double_bottom.trend_exit_min_hold_days": [3, 4, 5],
        "double_bottom.target_measured_move_multiple": [0.8, 1.0, 1.2, 1.4],
        "double_bottom.min_target_pct": [8.0, 10.0, 12.0],
    },
    "real_breakout": {
        "common.intraday_conflict_mode": ["conservative", "nearest_open", "target_first"],
        "real_breakout.max_hold_days": [10, 12, 15],
        "real_breakout.stop_below_support_pct": [1.2, 1.5, 2.0, 2.5],
        "real_breakout.breakeven_trigger_pct": [6.0, 6.5, 8.0, 9.5],
        "real_breakout.breakeven_buffer_pct": [0.3, 0.5, 0.8, 1.0],
        "real_breakout.trail_arm_pct": [8.5, 9.5, 11.0, 12.5],
        "real_breakout.trail_from_peak_pct": [4.0, 4.8, 5.5, 6.2],
        "real_breakout.trend_exit_arm_pct": [5.0, 6.0, 7.5, 9.0],
        "real_breakout.trend_exit_min_hold_days": [3, 4, 5],
        "real_breakout.target_from_pattern_multiple": [0.6, 0.8, 1.0, 1.2],
        "real_breakout.min_target_pct": [8.0, 9.0, 10.0, 12.0],
    },
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
    parser = argparse.ArgumentParser(description="Optimize exit rules for a single local research strategy.")
    parser.add_argument("--strategy-id", required=True, choices=sorted(EXIT_PARAM_SPACE.keys()))
    parser.add_argument("--start-date", required=True, help="Training start date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Training end date YYYYMMDD.")
    parser.add_argument("--validation-start-date", default="", help="Optional validation start date YYYYMMDD.")
    parser.add_argument("--validation-end-date", default="", help="Optional validation end date YYYYMMDD.")
    parser.add_argument("--trials", type=int, default=80, help="Random-search trials, excluding the baseline exit config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hold-days", default="1,3,5,10", help="Comma-separated forward horizons for reporting.")
    parser.add_argument("--regime-filter", default="", choices=["", "上涨趋势", "下跌趋势", "震荡趋势"], help="Optional market regime filter.")
    parser.add_argument(
        "--strategy-config-preset",
        default="",
        help=f"Named research preset for strategy overrides. Available: {strategy_config_preset_help()}",
    )
    parser.add_argument("--strategy-config-file", default="", help="JSON strategy overrides.")
    parser.add_argument(
        "--exit-config-preset",
        default="",
        help=f"Named exit preset. Available: {exit_config_preset_help()}",
    )
    parser.add_argument("--exit-config-file", default="", help="JSON base sell-rule overrides.")
    parser.add_argument("--max-trade-days", type=int, default=0, help="Limit training trade days for smoke tests.")
    parser.add_argument("--validation-max-trade-days", type=int, default=0, help="Limit validation trade days for smoke tests.")
    parser.add_argument("--min-trades", type=int, default=8, help="Soft minimum trade count per sample.")
    parser.add_argument("--show-top", type=int, default=15, help="Rows to print from ranked results.")
    return parser.parse_args()


def build_base_frame(
    *,
    start_date: str,
    end_date: str,
    strategy_id: str,
    hold_days: list[int],
    strategy_config: dict[str, Any],
    max_trade_days: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dataset = prepare_backtest_dataset(
        start_date=start_date,
        end_date=end_date,
        strategy_ids=[strategy_id],
        hold_days=hold_days,
        max_trade_days=max_trade_days,
    )
    result = run_backtest_on_dataset(
        dataset=dataset,
        strategy_ids=[strategy_id],
        hold_days=hold_days,
        strategy_overrides={strategy_id: strategy_config},
        exit_config={},
        export_results=False,
        use_signal_cache=True,
    )
    base_frame = result["daily_results"].copy()
    drop_cols = [column for column in EXIT_COLUMNS if column in base_frame.columns]
    if drop_cols:
        base_frame = base_frame.drop(columns=drop_cols)
    return base_frame, dataset.price_path_map


def sample_exit_config(strategy_id: str, rng: random.Random, base_config: dict[str, Any]) -> dict[str, Any]:
    sampled = copy.deepcopy(base_config)
    for dotted_key, values in EXIT_PARAM_SPACE[strategy_id].items():
        set_nested_config_value(sampled, dotted_key, rng.choice(values))
    return sampled


def run_exit_trial(
    base_frame: pd.DataFrame,
    price_path_map: dict[str, pd.DataFrame],
    exit_config: dict[str, Any],
    regime_filter: str,
    min_trades: int,
) -> dict[str, Any]:
    daily_results = apply_exit_rules(base_frame.copy(), price_path_map, config=exit_config)
    metrics = evaluate_backtest_frame(daily_results, regime_filter=regime_filter, min_trades=min_trades)
    metrics["raw_daily_results"] = daily_results
    return metrics


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
    base_exit_config, exit_config_meta = load_exit_config_with_preset(
        preset_name=args.exit_config_preset,
        config_file=args.exit_config_file,
    )
    rng = random.Random(args.seed)

    log_step("prepare training base frame")
    train_base_frame, train_price_path_map = build_base_frame(
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_id=args.strategy_id,
        hold_days=hold_days,
        strategy_config=base_strategy_config,
        max_trade_days=args.max_trade_days,
    )

    valid_base_frame = None
    valid_price_path_map = None
    if args.validation_start_date and args.validation_end_date:
        log_step("prepare validation base frame")
        valid_base_frame, valid_price_path_map = build_base_frame(
            start_date=args.validation_start_date,
            end_date=args.validation_end_date,
            strategy_id=args.strategy_id,
            hold_days=hold_days,
            strategy_config=base_strategy_config,
            max_trade_days=args.validation_max_trade_days,
        )

    trial_configs: list[dict[str, Any]] = [copy.deepcopy(base_exit_config)]
    seen_signatures = {config_signature(base_exit_config)}
    max_attempts = max(args.trials * 25, 150)
    attempts = 0
    while len(trial_configs) < args.trials + 1 and attempts < max_attempts:
        attempts += 1
        sampled = sample_exit_config(args.strategy_id, rng, base_exit_config)
        signature = config_signature(sampled)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        trial_configs.append(sampled)

    rows: list[dict[str, Any]] = []
    for idx, exit_config in enumerate(trial_configs, start=1):
        log_step(f"exit trial {idx}/{len(trial_configs)}")
        train_metrics = run_exit_trial(
            base_frame=train_base_frame,
            price_path_map=train_price_path_map,
            exit_config=exit_config,
            regime_filter=args.regime_filter,
            min_trades=args.min_trades,
        )
        row = {
            "trial_index": idx - 1,
            "strategy_id": args.strategy_id,
            "strategy_name": STRATEGY_REGISTRY[args.strategy_id].strategy_name,
            "regime_filter": args.regime_filter or "全部",
            "exit_config_json": json.dumps(json_safe(exit_config), ensure_ascii=False, sort_keys=True),
            "is_baseline": idx == 1,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items() if key != "raw_daily_results"})

        if valid_base_frame is not None and valid_price_path_map is not None:
            valid_metrics = run_exit_trial(
                base_frame=valid_base_frame,
                price_path_map=valid_price_path_map,
                exit_config=exit_config,
                regime_filter=args.regime_filter,
                min_trades=max(3, args.min_trades // 2),
            )
            row.update({f"valid_{key}": value for key, value in valid_metrics.items() if key != "raw_daily_results"})
            row["selection_score"] = round(
                float(row.get("train_selection_score", -999.0)) * 0.40 + float(row.get("valid_selection_score", -999.0)) * 0.60,
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

    run_tag = f"optimize_exit_{args.strategy_id}_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = export_root_dir() / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(export_dir / "trial_results.csv", index=False)
    top_results = results.head(args.show_top).copy()
    top_results.to_csv(export_dir / "top_results.csv", index=False)

    summary = {
        "strategy_id": args.strategy_id,
        "strategy_name": STRATEGY_REGISTRY[args.strategy_id].strategy_name,
        "train_range": [args.start_date, args.end_date],
        "validation_range": [args.validation_start_date, args.validation_end_date] if valid_base_frame is not None else None,
        "regime_filter": args.regime_filter or "全部",
        "hold_days": hold_days,
        "trials_requested": args.trials,
        "trials_completed": int(len(results)),
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
    print("===== top exit trials =====")
    print(top_results.to_string(index=False) if not top_results.empty else "(empty)")
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
