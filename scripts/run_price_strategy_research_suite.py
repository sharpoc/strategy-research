from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from optimize_price_strategy_params import evaluate_backtest_frame
from research_config_presets import (
    exit_config_preset_help,
    load_exit_config_with_preset,
    load_json_file,
    load_strategy_overrides_with_preset,
    strategy_config_preset_help,
)
from research_backtest_utils import json_safe, log_step, repo_root_dir
from run_price_strategy_regime_backtest import run_backtest


DEFAULT_STRATEGIES = ["limitup_l1l2", "platform_breakout", "double_bottom"]
DEFAULT_REGIMES = ["上涨趋势", "震荡趋势", "下跌趋势"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a consolidated local research suite for price-based strategies.")
    parser.add_argument("--start-date", required=True, help="Backtest start date YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Backtest end date YYYYMMDD.")
    parser.add_argument("--strategies", default="limitup_l1l2,platform_breakout,double_bottom", help="Comma-separated strategy ids.")
    parser.add_argument("--hold-days", default="1,3,5,10", help="Comma-separated fixed forward horizons.")
    parser.add_argument(
        "--strategy-config-preset",
        default="",
        help=f"Named research preset for strategy overrides. Available: {strategy_config_preset_help()}",
    )
    parser.add_argument("--strategy-config-file", default="", help="JSON strategy override file.")
    parser.add_argument("--regime-config-file", default="", help="JSON market regime override file.")
    parser.add_argument(
        "--exit-config-preset",
        default="",
        help=f"Named exit preset. Available: {exit_config_preset_help()}",
    )
    parser.add_argument("--exit-config-file", default="", help="JSON exit-rule override file.")
    parser.add_argument("--max-trade-days", type=int, default=0, help="Limit trade days for smoke tests.")
    parser.add_argument("--min-trades", type=int, default=5, help="Soft minimum trade count when evaluating regime suitability.")
    parser.add_argument("--top-trades", type=int, default=10, help="How many best/worst trades to export.")
    return parser.parse_args()

def export_root_dir() -> Path:
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_strategy_compare(daily_results: pd.DataFrame, strategies: list[str], min_trades: int) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy_id in strategies:
        sub = daily_results[daily_results["strategy_id"] == strategy_id].copy()
        if sub.empty:
            continue
        metrics = evaluate_backtest_frame(sub, regime_filter="", min_trades=min_trades)
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": sub["strategy_name"].iloc[0],
                **metrics,
            }
        )
    compare = pd.DataFrame(rows)
    if not compare.empty:
        compare = compare.sort_values(
            ["selection_score", "avg_exit_return_pct", "exit_win_rate_pct", "filled_trades"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    return compare


def build_regime_compare(daily_results: pd.DataFrame, strategies: list[str], min_trades: int) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy_id in strategies:
        sub = daily_results[daily_results["strategy_id"] == strategy_id].copy()
        if sub.empty:
            continue
        strategy_name = sub["strategy_name"].iloc[0]
        for regime in DEFAULT_REGIMES:
            metrics = evaluate_backtest_frame(sub, regime_filter=regime, min_trades=min_trades)
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "market_regime": regime,
                    **metrics,
                }
            )
    compare = pd.DataFrame(rows)
    if not compare.empty:
        compare = compare.sort_values(
            ["strategy_id", "selection_score", "avg_exit_return_pct", "filled_trades"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    return compare


def build_best_regime_recommendations(regime_compare: pd.DataFrame) -> pd.DataFrame:
    if regime_compare.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for strategy_id, sub in regime_compare.groupby("strategy_id", sort=False):
        ranked = sub.copy()
        for col in ["selection_score", "avg_exit_return_pct", "filled_trades"]:
            if col in ranked.columns:
                ranked[col] = pd.to_numeric(ranked[col], errors="coerce")
        ranked = ranked.sort_values(
            ["selection_score", "avg_exit_return_pct", "filled_trades"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        best = ranked.iloc[0]
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": best["strategy_name"],
                "recommended_regime": best["market_regime"],
                "selection_score": best.get("selection_score"),
                "avg_exit_return_pct": best.get("avg_exit_return_pct"),
                "exit_win_rate_pct": best.get("exit_win_rate_pct"),
                "filled_trades": best.get("filled_trades"),
            }
        )
    return pd.DataFrame(rows)


def build_top_trade_tables(daily_results: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if daily_results.empty or "exit_return_pct" not in daily_results.columns:
        return pd.DataFrame(), pd.DataFrame()
    valid = daily_results[daily_results["has_signal"]].copy()
    valid["exit_return_pct"] = pd.to_numeric(valid["exit_return_pct"], errors="coerce")
    valid = valid.dropna(subset=["exit_return_pct"]).copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    keep_cols = [
        column
        for column in [
            "trade_date",
            "strategy_id",
            "strategy_name",
            "market_regime",
            "ts_code",
            "name",
            "strategy_rank_score",
            "entry_trade_date",
            "entry_open",
            "exit_trade_date",
            "exit_price",
            "exit_reason",
            "exit_hold_days",
            "exit_return_pct",
            "exit_mfe_pct",
            "exit_mae_pct",
        ]
        if column in valid.columns
    ]
    best = valid.sort_values(["exit_return_pct", "exit_mfe_pct"], ascending=[False, False]).head(top_n)[keep_cols].reset_index(drop=True)
    worst = valid.sort_values(["exit_return_pct", "exit_mae_pct"], ascending=[True, True]).head(top_n)[keep_cols].reset_index(drop=True)
    return best, worst


def main() -> None:
    args = parse_args()
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    hold_days = sorted({int(item.strip()) for item in args.hold_days.split(",") if item.strip()})
    strategy_overrides, strategy_config_meta = load_strategy_overrides_with_preset(
        preset_name=args.strategy_config_preset,
        config_file=args.strategy_config_file,
    )
    regime_config = load_json_file(args.regime_config_file)
    exit_config, exit_config_meta = load_exit_config_with_preset(
        preset_name=args.exit_config_preset,
        config_file=args.exit_config_file,
    )

    log_step("run consolidated baseline backtest")
    result = run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_ids=strategies,
        hold_days=hold_days,
        strategy_overrides=strategy_overrides,
        regime_config=regime_config,
        exit_config=exit_config,
        max_trade_days=args.max_trade_days,
    )

    daily_results = result["daily_results"].copy()
    strategy_compare = build_strategy_compare(daily_results, strategies=strategies, min_trades=args.min_trades)
    regime_compare = build_regime_compare(daily_results, strategies=strategies, min_trades=args.min_trades)
    best_regime = build_best_regime_recommendations(regime_compare)
    top_trades, worst_trades = build_top_trade_tables(daily_results, top_n=args.top_trades)

    run_tag = f"research_suite_{args.start_date}_{args.end_date}_{pd.Timestamp.now().strftime('%H%M%S')}"
    export_dir = export_root_dir() / run_tag
    export_dir.mkdir(parents=True, exist_ok=True)
    daily_results.to_csv(export_dir / "daily_results.csv", index=False)
    result["strategy_summary"].to_csv(export_dir / "strategy_summary.csv", index=False)
    result["regime_summary"].to_csv(export_dir / "regime_summary.csv", index=False)
    result["monthly_summary"].to_csv(export_dir / "monthly_summary.csv", index=False)
    result["exit_reason_summary"].to_csv(export_dir / "exit_reason_summary.csv", index=False)
    strategy_compare.to_csv(export_dir / "strategy_compare.csv", index=False)
    regime_compare.to_csv(export_dir / "strategy_regime_compare.csv", index=False)
    best_regime.to_csv(export_dir / "best_regime_recommendations.csv", index=False)
    top_trades.to_csv(export_dir / "top_trades.csv", index=False)
    worst_trades.to_csv(export_dir / "worst_trades.csv", index=False)

    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "strategies": strategies,
        "hold_days": hold_days,
        "trade_days": int(result["summary"]["trade_days"]),
        "daily_rows": int(result["summary"]["daily_rows"]),
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
    print("===== strategy compare =====")
    print(strategy_compare.to_string(index=False) if not strategy_compare.empty else "(empty)")
    print("===== best regime recommendations =====")
    print(best_regime.to_string(index=False) if not best_regime.empty else "(empty)")
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
