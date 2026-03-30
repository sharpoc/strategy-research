from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_strategy_registry import build_price_strategy_registry
from research_config_presets import (
    exit_config_preset_help,
    load_exit_config_with_preset,
    load_json_file,
    load_strategy_overrides_with_preset,
    strategy_config_preset_help,
)
from research_backtest_utils import json_safe, log_step, repo_root_dir
from run_price_strategy_regime_backtest import (
    prepare_backtest_dataset,
    run_backtest_on_dataset,
)


STRATEGY_REGISTRY = build_price_strategy_registry()


PARAM_SPACE: dict[str, dict[str, list[Any]]] = {
    "limitup_l1l2": {
        "zz_left": [2, 3, 4],
        "zz_right": [2, 3, 4],
        "l1_break_pct": [0.1, 0.2, 0.3, 0.5],
        "min_bars_between_lows": [2, 3, 4],
        "max_bars_between_lows": [25, 35, 45],
        "higher_low_min_pct": [0.0, 0.3, 0.5, 0.8],
        "higher_low_max_pct": [8.0, 12.0, 16.0],
        "rebound_max_bars": [15, 20, 25],
        "candidate_score_threshold": [50.0, 55.0, 60.0, 65.0],
        "_research_entry_gate.min_impulse_pct": [5.5, 6.0, 6.5, 7.0],
        "_research_entry_gate.max_pullback_pct": [7.0, 8.0, 9.0],
        "_research_entry_gate.min_l2_above_l1_pct": [0.8, 1.0, 1.2, 1.5],
        "_research_entry_gate.min_hold_buffer_pct": [0.8, 1.0, 1.2],
        "_research_entry_gate.max_close_vs_l2_pct": [5.0, 6.0, 7.0],
        "_research_entry_gate.require_volume_ok": [False, True],
    },
    "platform_breakout": {
        "platform_min_bars": [3, 4, 5],
        "platform_max_bars": [8, 10, 12],
        "platform_amp_max_pct": [10.0, 12.0, 15.0],
        "breakout_close_buffer_pct": [1.0, 2.0, 3.0],
        "limit_volume_mult_min": [1.2, 1.3, 1.5],
        "main_pullback_min_ratio": [0.45, 0.50, 0.55],
        "main_pullback_max_ratio": [0.62, 0.67, 0.72],
        "pullback_avg_vol_ratio_max": [0.60, 0.70, 0.80],
        "platform_support_break_pct": [1.0, 2.0, 3.0],
        "ma20_break_pct": [1.0, 2.0, 3.0],
        "candidate_score_threshold": [55.0, 60.0, 65.0],
    },
    "double_bottom": {
        "pivot_left": [3, 4, 5],
        "pivot_right": [3, 4, 5],
        "min_pre_down_pct": [12.0, 15.0, 18.0, 20.0],
        "min_rebound_pct": [8.0, 10.0, 12.0],
        "min_bars_between_bottoms": [8, 10, 12],
        "max_bars_between_bottoms": [30, 40, 50],
        "max_l2_deviation_pct": [2.0, 3.0, 4.0],
        "pullback_volume_ratio_max": [0.75, 0.85, 0.95],
        "neckline_breakout_volume_mult": [1.2, 1.3, 1.5],
        "retest_max_break_pct": [1.0, 2.0, 3.0],
        "pattern_stale_max_bars_after_l2": [20, 25, 30],
        "candidate_score_threshold": [50.0, 55.0, 60.0, 65.0],
    },
    "real_breakout": {
        "pre_runup_min_pct": [10.0, 12.0, 15.0],
        "platform_min_bars": [5, 6, 7],
        "platform_max_bars": [10, 12, 14],
        "platform_amp_max_pct": [8.0, 10.0, 12.0],
        "platform_shrink_vol_ratio_max": [0.72, 0.82, 0.90],
        "breakout_close_buffer_pct": [0.8, 1.2, 1.8],
        "breakout_volume_ratio_min": [1.25, 1.35, 1.50],
        "breakout_upper_shadow_pct_max": [1.5, 2.0, 2.5],
        "support_hold_break_pct": [1.2, 1.8, 2.4],
        "candidate_score_threshold": [56.0, 60.0, 64.0],
        "_research_real_breakout_tuning.min_breakout_volume_ratio": [1.2, 1.3, 1.4],
        "_research_real_breakout_tuning.max_breakout_volume_ratio": [2.1, 2.4, 2.8],
        "_research_real_breakout_tuning.max_platform_vol_ratio": [0.78, 0.82, 0.88],
        "_research_real_breakout_tuning.min_base_score": [56.0, 58.0, 60.0, 62.0],
        "_research_entry_gate.allowed_market_regimes": [["下跌趋势"], ["下跌趋势", "震荡趋势"]],
        "_research_entry_gate.max_breakout_volume_ratio": [2.1, 2.4, 2.8],
        "_research_entry_gate.max_current_buffer_pct": [8.5, 10.0, 12.0],
        "_research_entry_gate.max_platform_vol_ratio": [0.82, 0.88, 0.92],
        "_research_entry_gate.min_ma20_slope_pct": [0.0, -0.2],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a single price strategy with isolated local research datasets.")
    parser.add_argument("--strategy-id", required=True, choices=sorted(PARAM_SPACE.keys()))
    parser.add_argument("--start-date", required=True, help="Training start date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Training end date YYYYMMDD.")
    parser.add_argument("--validation-start-date", default="", help="Optional validation start date YYYYMMDD.")
    parser.add_argument("--validation-end-date", default="", help="Optional validation end date YYYYMMDD.")
    parser.add_argument("--trials", type=int, default=80, help="Random-search trials, excluding the baseline config.")
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
    parser.add_argument("--max-trade-days", type=int, default=0, help="Limit training trade days for smoke tests.")
    parser.add_argument("--validation-max-trade-days", type=int, default=0, help="Limit validation trade days for smoke tests.")
    parser.add_argument("--min-trades", type=int, default=8, help="Soft minimum trade count per sample.")
    parser.add_argument("--show-top", type=int, default=15, help="Rows to print from ranked results.")
    return parser.parse_args()


def export_root_dir() -> Path:
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_nested_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    if "." not in dotted_key:
        config[dotted_key] = value
        return
    current: dict[str, Any] = config
    parts = [part for part in dotted_key.split(".") if part]
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value


def sample_strategy_config(strategy_id: str, rng: random.Random, base_config: dict[str, Any]) -> dict[str, Any]:
    sampled = copy.deepcopy(base_config)
    for key, values in PARAM_SPACE[strategy_id].items():
        set_nested_config_value(sampled, key, rng.choice(values))
    return sampled


def config_signature(config: dict[str, Any]) -> str:
    return json.dumps(json_safe(config), ensure_ascii=False, sort_keys=True)


def max_drawdown_pct(returns_pct: pd.Series) -> float | None:
    if returns_pct.empty:
        return None
    equity = (1.0 + returns_pct / 100.0).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    return abs(float(drawdown.min()) * 100.0)


def profit_factor(returns_pct: pd.Series) -> float | None:
    if returns_pct.empty:
        return None
    gains = returns_pct[returns_pct > 0].sum()
    losses = returns_pct[returns_pct < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else None
    return float(gains / abs(losses))


def evaluate_backtest_frame(
    daily_results: pd.DataFrame,
    regime_filter: str,
    min_trades: int,
) -> dict[str, Any]:
    empty_metrics = {
        "selection_score": -999.0,
        "filled_trades": 0,
        "signal_days": 0,
        "avg_exit_return_pct": None,
        "median_exit_return_pct": None,
        "exit_win_rate_pct": None,
        "profit_factor": None,
        "profit_factor_label": None,
        "avg_exit_mfe_pct": None,
        "avg_exit_mae_pct": None,
        "avg_exit_hold_days": None,
        "max_drawdown_pct": None,
        "positive_month_ratio_pct": None,
        "months": 0,
    }
    if daily_results is None or daily_results.empty:
        return dict(empty_metrics)

    frame = daily_results[daily_results["has_signal"]].copy()
    if regime_filter:
        frame = frame[frame["market_regime"] == regime_filter].copy()
    frame["exit_return_pct"] = pd.to_numeric(frame.get("exit_return_pct"), errors="coerce")
    frame["exit_mfe_pct"] = pd.to_numeric(frame.get("exit_mfe_pct"), errors="coerce")
    frame["exit_mae_pct"] = pd.to_numeric(frame.get("exit_mae_pct"), errors="coerce")
    frame["exit_hold_days"] = pd.to_numeric(frame.get("exit_hold_days"), errors="coerce")

    valid = frame.dropna(subset=["exit_return_pct"]).copy()
    if valid.empty:
        metrics = dict(empty_metrics)
        metrics["signal_days"] = int(len(frame))
        return metrics

    returns = valid["exit_return_pct"]
    valid["trade_month"] = valid["trade_date"].astype(str).str.slice(0, 6)
    monthly_returns = valid.groupby("trade_month")["exit_return_pct"].mean()
    pf = profit_factor(returns)
    pf_score = 0.0
    if pf is not None:
        if np.isinf(pf):
            pf_score = 8.0
        else:
            pf_score = max(min((pf - 1.0) * 6.0, 8.0), -6.0)
    avg_return = float(returns.mean())
    win_rate = float((returns > 0).mean() * 100.0)
    avg_mfe = float(valid["exit_mfe_pct"].dropna().mean()) if not valid["exit_mfe_pct"].dropna().empty else 0.0
    avg_mae = float(valid["exit_mae_pct"].dropna().mean()) if not valid["exit_mae_pct"].dropna().empty else 0.0
    avg_hold = float(valid["exit_hold_days"].dropna().mean()) if not valid["exit_hold_days"].dropna().empty else None
    dd = max_drawdown_pct(returns)
    positive_month_ratio = float((monthly_returns > 0).mean() * 100.0) if not monthly_returns.empty else 0.0
    trades_penalty = max(min_trades - len(valid), 0) * 4.0

    selection_score = (
        avg_return * 5.0
        + win_rate * 0.30
        + pf_score
        + positive_month_ratio * 0.08
        + avg_mfe * 0.12
        + avg_mae * 0.35
        - (dd or 0.0) * 0.60
        - trades_penalty
    )
    metrics = {
        "selection_score": round(float(selection_score), 4),
        "signal_days": int(len(frame)),
        "filled_trades": int(len(valid)),
        "avg_exit_return_pct": round(avg_return, 4),
        "median_exit_return_pct": round(float(returns.median()), 4),
        "exit_win_rate_pct": round(win_rate, 2),
        "profit_factor": None if pf is None or np.isinf(pf) else round(float(pf), 4),
        "profit_factor_label": "inf" if pf is not None and np.isinf(pf) else (round(float(pf), 4) if pf is not None else None),
        "avg_exit_mfe_pct": round(avg_mfe, 4),
        "avg_exit_mae_pct": round(avg_mae, 4),
        "avg_exit_hold_days": round(avg_hold, 2) if avg_hold is not None else None,
        "max_drawdown_pct": round(dd, 4) if dd is not None else None,
        "positive_month_ratio_pct": round(positive_month_ratio, 2),
        "months": int(monthly_returns.size),
    }
    return {**empty_metrics, **metrics}


def run_single_trial(
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
    metrics["raw_daily_results"] = result["daily_results"]
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
    exit_config, exit_config_meta = load_exit_config_with_preset(
        preset_name=args.exit_config_preset,
        config_file=args.exit_config_file,
    )
    rng = random.Random(args.seed)

    log_step("prepare training dataset")
    train_dataset = prepare_backtest_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_ids=[args.strategy_id],
        hold_days=hold_days,
        max_trade_days=args.max_trade_days,
    )

    validation_dataset = None
    if args.validation_start_date and args.validation_end_date:
        log_step("prepare validation dataset")
        validation_dataset = prepare_backtest_dataset(
            start_date=args.validation_start_date,
            end_date=args.validation_end_date,
            strategy_ids=[args.strategy_id],
            hold_days=hold_days,
            max_trade_days=args.validation_max_trade_days,
        )

    trial_configs: list[dict[str, Any]] = [dict(base_strategy_config)]
    seen_signatures = {config_signature(base_strategy_config)}
    max_attempts = max(args.trials * 20, 100)
    attempts = 0
    while len(trial_configs) < args.trials + 1 and attempts < max_attempts:
        attempts += 1
        sampled = sample_strategy_config(args.strategy_id, rng, base_strategy_config)
        signature = config_signature(sampled)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        trial_configs.append(sampled)

    rows: list[dict[str, Any]] = []
    for idx, strategy_config in enumerate(trial_configs, start=1):
        log_step(f"trial {idx}/{len(trial_configs)}")
        train_metrics = run_single_trial(
            dataset=train_dataset,
            strategy_id=args.strategy_id,
            hold_days=hold_days,
            strategy_config=strategy_config,
            regime_filter=args.regime_filter,
            min_trades=args.min_trades,
            exit_config=exit_config,
        )
        row = {
            "trial_index": idx - 1,
            "strategy_id": args.strategy_id,
            "strategy_name": STRATEGY_REGISTRY[args.strategy_id].strategy_name,
            "regime_filter": args.regime_filter or "全部",
            "config_json": json.dumps(json_safe(strategy_config), ensure_ascii=False, sort_keys=True),
            "is_baseline": idx == 1,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items() if key != "raw_daily_results"})

        if validation_dataset is not None:
            validation_metrics = run_single_trial(
                dataset=validation_dataset,
                strategy_id=args.strategy_id,
                hold_days=hold_days,
                strategy_config=strategy_config,
                regime_filter=args.regime_filter,
                min_trades=max(3, args.min_trades // 2),
                exit_config=exit_config,
            )
            row.update({f"valid_{key}": value for key, value in validation_metrics.items() if key != "raw_daily_results"})
            row["selection_score"] = round(
                float(row.get("train_selection_score", -999.0)) * 0.40 + float(row.get("valid_selection_score", -999.0)) * 0.60,
                4,
            )
        else:
            row["selection_score"] = row.get("train_selection_score", -999.0)
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        by=["selection_score", "train_avg_exit_return_pct", "train_exit_win_rate_pct", "train_filled_trades"],
        ascending=[False, False, False, False],
    ) if rows else pd.DataFrame()
    if not results.empty:
        for column in ["selection_score", "train_avg_exit_return_pct", "train_exit_win_rate_pct", "train_filled_trades"]:
            if column in results.columns:
                results[column] = pd.to_numeric(results[column], errors="coerce")
        results = results.sort_values(
            ["selection_score", "train_avg_exit_return_pct", "train_exit_win_rate_pct", "train_filled_trades"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    run_tag = f"optimize_{args.strategy_id}_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = export_root_dir() / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(export_dir / "trial_results.csv", index=False)

    top_results = results.head(args.show_top).copy()
    top_results.to_csv(export_dir / "top_results.csv", index=False)
    summary = {
        "strategy_id": args.strategy_id,
        "strategy_name": STRATEGY_REGISTRY[args.strategy_id].strategy_name,
        "train_range": [args.start_date, args.end_date],
        "validation_range": [args.validation_start_date, args.validation_end_date] if validation_dataset is not None else None,
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
    print("===== top trials =====")
    print(top_results.to_string(index=False))
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
