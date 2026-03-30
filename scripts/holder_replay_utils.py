from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from holder_strategy_core import (
    STRATEGY_NAME,
    HolderStrategyConfig,
    apply_holder_stage1,
    apply_holder_stage2,
    json_safe,
)
from research_backtest_utils import (
    build_forward_return_table,
    discover_cached_trade_dates,
    load_cached_market_daily_history,
    log_step,
    repo_root_dir,
)
from strategy_exit_rules import apply_exit_rules, build_price_path_map


def export_root_dir() -> Path:
    path = repo_root_dir() / "output" / "research_backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_snapshot_roots() -> list[Path]:
    return [
        repo_root_dir() / "output" / "jupyter-notebook" / "tushare_screen_exports",
        repo_root_dir() / "output" / "research_backtests" / "holder_snapshots",
        repo_root_dir() / "output" / "research_backtests" / "holder_snapshots_research",
        repo_root_dir() / "output" / "research_backtests" / "holder_snapshots_smoke",
    ]


def snapshot_search_roots(snapshot_root: str = "") -> list[Path]:
    roots: list[Path] = []
    if snapshot_root.strip():
        roots.append(Path(snapshot_root).expanduser().resolve())
    roots.extend(default_snapshot_roots())

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        root_str = str(root.resolve())
        if root_str in seen:
            continue
        seen.add(root_str)
        deduped.append(root)
    return deduped


def available_snapshot_dirs(snapshot_root: str = "") -> dict[str, Path]:
    snapshot_map: dict[str, Path] = {}
    for root in snapshot_search_roots(snapshot_root):
        if not root.exists():
            continue
        for path in sorted(root.glob("holder_increase_screen_*")):
            trade_date = path.name.rsplit("_", 1)[-1]
            if len(trade_date) != 8 or not trade_date.isdigit():
                continue
            snapshot_map.setdefault(trade_date, path)
    return snapshot_map


def available_snapshot_trade_dates(snapshot_root: str = "") -> list[str]:
    return sorted(available_snapshot_dirs(snapshot_root).keys())


def _load_snapshot_summary(export_dir: Path) -> dict[str, Any]:
    summary_path = export_dir / "screen_summary.json"
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log_step(f"holder snapshot summary read failed path={summary_path} error={exc}")
        return {}


def load_export_snapshots(
    start_date: str,
    end_date: str,
    snapshot_root: str = "",
    max_trade_days: int = 0,
) -> list[dict[str, Any]]:
    snapshot_map = available_snapshot_dirs(snapshot_root)
    trade_dates = [trade_date for trade_date in sorted(snapshot_map.keys()) if start_date <= trade_date <= end_date]
    if max_trade_days > 0:
        trade_dates = trade_dates[-max_trade_days:]

    rows: list[dict[str, Any]] = []
    for trade_date in trade_dates:
        export_dir = snapshot_map[trade_date]
        candidate_base_path = export_dir / "candidate_base.csv"
        deep_metrics_path = export_dir / "deep_metrics_stage1.csv"
        stage2_cyq_path = export_dir / "stage2_cyq_metrics.csv"
        if not candidate_base_path.exists() or not deep_metrics_path.exists():
            continue

        candidate_base = pd.read_csv(candidate_base_path)
        deep_metrics_stage1 = pd.read_csv(deep_metrics_path)
        stage2_cyq_metrics = pd.read_csv(stage2_cyq_path) if stage2_cyq_path.exists() else pd.DataFrame()
        snapshot_summary = _load_snapshot_summary(export_dir)
        market_regime = str(snapshot_summary.get("market_regime") or "")
        if not market_regime and "market_regime" in candidate_base.columns and not candidate_base["market_regime"].dropna().empty:
            market_regime = str(candidate_base["market_regime"].dropna().astype(str).iloc[0])
        rows.append(
            {
                "trade_date": trade_date,
                "export_dir": export_dir,
                "candidate_base": candidate_base,
                "deep_metrics_stage1": deep_metrics_stage1,
                "stage2_cyq_metrics": stage2_cyq_metrics,
                "snapshot_summary": snapshot_summary,
                "market_regime": market_regime or "neutral",
            }
        )
    return rows


def load_price_context(
    signal_start_date: str,
    hold_days: list[int],
    forward_end_date: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    effective_end_date = forward_end_date or "20991231"
    history_start_date = (pd.Timestamp(signal_start_date) - pd.Timedelta(days=180)).strftime("%Y%m%d")
    cached_dates = discover_cached_trade_dates(history_start_date, effective_end_date)
    if not cached_dates:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    full_history = load_cached_market_daily_history(cached_dates)
    if full_history.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    full_history = full_history.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    full_history["trade_date"] = full_history["trade_date"].astype(str)
    forward_history = full_history[full_history["trade_date"] >= signal_start_date].copy()
    if forward_history.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    forward_table = build_forward_return_table(forward_history, hold_days)
    price_path_map = build_price_path_map(full_history)
    history_trade_dates = sorted(full_history["trade_date"].dropna().astype(str).unique().tolist())
    return full_history, forward_table, price_path_map, history_trade_dates


def _coalesce_duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for base_name in ["chip_score"]:
        left = f"{base_name}_x"
        right = f"{base_name}_y"
        if left not in work.columns or right not in work.columns:
            continue
        work[base_name] = work[left].where(work[left].notna(), work[right])
        work = work.drop(columns=[left, right])
    return work


def build_holder_daily_results(
    snapshots: list[dict[str, Any]],
    forward_table: pd.DataFrame,
    price_path_map: dict[str, pd.DataFrame],
    config_overrides: dict[str, Any],
    exit_config: Optional[dict[str, Any]] = None,
    apply_exit: bool = True,
) -> pd.DataFrame:
    daily_rows: list[dict[str, Any]] = []
    forward_lookup = forward_table.copy()
    if not forward_lookup.empty:
        forward_lookup["trade_date"] = forward_lookup["trade_date"].astype(str)
        forward_lookup["ts_code"] = forward_lookup["ts_code"].astype(str)

    for snapshot in snapshots:
        trade_date = str(snapshot["trade_date"])
        cfg = HolderStrategyConfig.for_end_date(trade_date, **config_overrides)
        candidate_base = snapshot["candidate_base"]
        deep_metrics_stage1 = snapshot["deep_metrics_stage1"]
        stage2_cyq_metrics = snapshot["stage2_cyq_metrics"] if cfg.enable_stage2_cyq else pd.DataFrame()
        market_regime = str(snapshot.get("market_regime") or "neutral")

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
                    "today_direction": stage2_result.get("today_direction"),
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
                "market_regime": market_regime,
                "today_direction": stage2_result.get("today_direction"),
                "has_signal": True,
                "strategy_rank_score": row.get("priority_score"),
            }
        )
        daily_rows.append(row)

    daily_results = pd.DataFrame(daily_rows)
    if daily_results.empty:
        return daily_results

    daily_results["trade_date"] = daily_results["trade_date"].astype(str)
    if "ts_code" in daily_results.columns:
        daily_results["ts_code"] = daily_results["ts_code"].astype(str)
    if not forward_lookup.empty and {"ts_code", "trade_date"}.issubset(daily_results.columns):
        daily_results = daily_results.merge(forward_lookup, on=["ts_code", "trade_date"], how="left")
    daily_results = _coalesce_duplicate_columns(daily_results)
    if apply_exit and price_path_map:
        daily_results = apply_exit_rules(daily_results, price_path_map, config=exit_config)
    return daily_results


def write_replay_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, ensure_ascii=False, indent=2)
