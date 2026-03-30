from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_strategy_registry import build_price_strategy_registry
from research_backtest_utils import json_safe, repo_root_dir


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    dataset_name: str
    roots: tuple[str, ...]
    entry_kind: str = "file"
    regex: str = ""
    date_mode: str = "none"
    notes: str = ""


@dataclass(frozen=True)
class StrategyDataSpec:
    strategy_id: str
    strategy_name: str
    strategy_kind: str
    recommended_history_years: float
    required_dataset_ids: tuple[str, ...]
    optional_dataset_ids: tuple[str, ...]
    notes: tuple[str, ...] = ()


def build_dataset_catalog() -> dict[str, DatasetSpec]:
    return {
        "trade_calendar": DatasetSpec(
            dataset_id="trade_calendar",
            dataset_name="交易日历缓存",
            roots=(
                "output/cache/research_api",
                "output/cache/holder_increase_api",
                "output/cache/double_bottom_api",
                "output/cache/platform_breakout_api",
                "output/cache/limitup_l1l2_api",
            ),
            regex=r"^trade_cal_[0-9a-f]{12}\.csv$",
            date_mode="trade_cal_csv",
            notes="逐日回测的基础交易日来源。",
        ),
        "stock_basic_all": DatasetSpec(
            dataset_id="stock_basic_all",
            dataset_name="全市场股票基础信息缓存",
            roots=("output/cache/research_api", "output/cache/holder_increase_api"),
            regex=r"^stock_basic_all_[0-9a-f]{12}\.csv$",
            notes="全市场股票池、上市日期、市场板块等基础字段。",
        ),
        "market_daily_all": DatasetSpec(
            dataset_id="market_daily_all",
            dataset_name="全市场日线缓存",
            roots=(
                "output/cache/research_api",
                "output/cache/double_bottom_api",
                "output/cache/platform_breakout_api",
                "output/cache/limitup_l1l2_api",
            ),
            regex=r"^daily_all_(\d{8})_[0-9a-f]{12}\.csv$",
            date_mode="single",
            notes="价格型策略研究的核心历史行情缓存。",
        ),
        "holder_snapshots": DatasetSpec(
            dataset_id="holder_snapshots",
            dataset_name="增持策略历史快照",
            roots=(
                "output/jupyter-notebook/tushare_screen_exports",
                "output/research_backtests/holder_snapshots_research",
                "output/research_backtests/holder_snapshots_smoke",
            ),
            entry_kind="dir",
            regex=r"^holder_increase_screen_(\d{8})$",
            date_mode="single",
            notes="星曜增持臻选做 export replay / 历史对拍的基础快照。",
        ),
        "holder_events": DatasetSpec(
            dataset_id="holder_events",
            dataset_name="增减持事件缓存",
            roots=("output/cache/holder_increase_api",),
            regex=r"^stk_holdertrade_(\d{8})_(\d{8})_[0-9a-f]{12}\.csv$",
            date_mode="range",
            notes="星曜增持臻选做完整 API 历史回放时的关键事件源。",
        ),
        "holder_share_float": DatasetSpec(
            dataset_id="holder_share_float",
            dataset_name="解禁与流通股本缓存",
            roots=("output/cache/holder_increase_api",),
            regex=r"^share_float_(\d{8})_(\d{8})_[0-9a-f]{12}\.csv$",
            date_mode="range",
            notes="用于解禁风控与流通股本判断。",
        ),
        "holder_moneyflow": DatasetSpec(
            dataset_id="holder_moneyflow",
            dataset_name="资金流缓存",
            roots=("output/cache/holder_increase_api",),
            regex=r"^moneyflow_(\d{8})_[0-9a-f]{12}\.csv$",
            date_mode="single",
            notes="增持策略里的主力资金辅助评分。",
        ),
        "holder_forecast": DatasetSpec(
            dataset_id="holder_forecast",
            dataset_name="业绩预告缓存",
            roots=("output/cache/holder_increase_api",),
            regex=r"^forecast_.*_[0-9a-f]{12}\.csv$",
            notes="增持策略的业绩安全过滤。",
        ),
        "holder_fina_indicator": DatasetSpec(
            dataset_id="holder_fina_indicator",
            dataset_name="财务指标缓存",
            roots=("output/cache/holder_increase_api",),
            regex=r"^fina_indicator_.*_[0-9a-f]{12}\.csv$",
            notes="增持策略的 ROE / 现金流 / 负债率过滤。",
        ),
        "holder_adj_factor": DatasetSpec(
            dataset_id="holder_adj_factor",
            dataset_name="复权因子缓存",
            roots=("output/cache/holder_increase_api",),
            regex=r"^adj_factor_.*_[0-9a-f]{12}\.csv$",
            notes="增持策略单股历史复权对齐。",
        ),
        "limitup_exports": DatasetSpec(
            dataset_id="limitup_exports",
            dataset_name="龙门双阶导出结果",
            roots=("output/jupyter-notebook/tushare_limitup_l1l2_exports",),
            entry_kind="dir",
            regex=r"^limitup_l1l2_pick_(\d{8})_\d{6}$",
            date_mode="single",
            notes="策略 runner 导出结果，可做对拍与回放留痕。",
        ),
        "platform_breakout_exports": DatasetSpec(
            dataset_id="platform_breakout_exports",
            dataset_name="天衡回踩导出结果",
            roots=("output/jupyter-notebook/tushare_platform_breakout_exports",),
            entry_kind="dir",
            regex=r"^platform_breakout_pick_(\d{8})_\d{6}$",
            date_mode="single",
            notes="策略 runner 导出结果，可做对拍与回放留痕。",
        ),
        "double_bottom_exports": DatasetSpec(
            dataset_id="double_bottom_exports",
            dataset_name="双底策略导出结果",
            roots=("output/jupyter-notebook/tushare_double_bottom_exports",),
            entry_kind="dir",
            regex=r"^double_bottom_pick_(\d{8})_\d{6}$",
            date_mode="single",
            notes="双底策略本地研究导出结果。",
        ),
    }


def build_strategy_data_specs() -> dict[str, StrategyDataSpec]:
    price_specs = {
        strategy_id: StrategyDataSpec(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            strategy_kind=spec.strategy_kind,
            recommended_history_years=spec.recommended_history_years,
            required_dataset_ids=spec.required_dataset_ids,
            optional_dataset_ids=spec.optional_dataset_ids,
            notes=("价格型策略默认至少依赖全市场日线 + 股票基础信息 + 交易日历。",),
        )
        for strategy_id, spec in build_price_strategy_registry().items()
    }
    price_specs["holder_increase"] = StrategyDataSpec(
        strategy_id="holder_increase",
        strategy_name="星曜增持臻选",
        strategy_kind="event_hybrid",
        recommended_history_years=2.5,
        required_dataset_ids=("trade_calendar", "stock_basic_all", "market_daily_all", "holder_snapshots"),
        optional_dataset_ids=(
            "holder_events",
            "holder_share_float",
            "holder_moneyflow",
            "holder_forecast",
            "holder_fina_indicator",
            "holder_adj_factor",
        ),
        notes=(
            "当前研究态优先走 export replay，因此 holder_snapshots 属于必需项。",
            "如果要升级到完整 API 历史回放，holder_events / holder_share_float / 财务与预告缓存也要尽量补齐。",
        ),
    )
    return price_specs


def _coverage_years(start_date: str | None, end_date: str | None) -> float | None:
    if not start_date or not end_date:
        return None
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        return None
    return round(float((end_ts - start_ts).days / 365.25), 2)


def _iter_root_entries(root_path: Path, entry_kind: str) -> list[Path]:
    if not root_path.exists():
        return []
    if entry_kind == "dir":
        return [entry for entry in root_path.iterdir() if entry.is_dir()]
    return [entry for entry in root_path.iterdir() if entry.is_file()]


def _collect_trade_cal_dates(paths: list[Path]) -> set[str]:
    open_dates: set[str] = set()
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        date_column = "cal_date" if "cal_date" in df.columns else "trade_date" if "trade_date" in df.columns else ""
        if not date_column:
            continue
        open_mask = pd.Series(True, index=df.index)
        if "is_open" in df.columns:
            open_mask = df["is_open"].astype(str).eq("1")
        values = df.loc[open_mask, date_column].dropna().astype(str).tolist()
        open_dates.update(values)
    return open_dates


def _scan_trade_cal_dates(paths: list[Path]) -> tuple[str | None, str | None, int, set[str]]:
    open_dates = _collect_trade_cal_dates(paths)
    if not open_dates:
        return None, None, 0, set()
    ordered = sorted(open_dates)
    return ordered[0], ordered[-1], len(ordered), open_dates


def scan_dataset_availability() -> dict[str, dict[str, Any]]:
    repo_root = repo_root_dir()
    catalog = build_dataset_catalog()
    availability: dict[str, dict[str, Any]] = {}
    dataset_date_points: dict[str, set[str]] = {}
    for dataset_id, spec in catalog.items():
        matched_paths: list[Path] = []
        missing_roots: list[str] = []
        date_points: set[str] = set()
        range_starts: list[str] = []
        range_ends: list[str] = []
        pattern = re.compile(spec.regex) if spec.regex else None
        for relative_root in spec.roots:
            root_path = repo_root / relative_root
            if not root_path.exists():
                missing_roots.append(relative_root)
                continue
            for entry in _iter_root_entries(root_path, spec.entry_kind):
                if pattern is not None:
                    match = pattern.match(entry.name)
                    if not match:
                        continue
                else:
                    match = None
                matched_paths.append(entry)
                if spec.date_mode == "single" and match is not None:
                    date_points.add(str(match.group(1)))
                elif spec.date_mode == "range" and match is not None:
                    range_starts.append(str(match.group(1)))
                    range_ends.append(str(match.group(2)))
        start_date = None
        end_date = None
        coverage_trade_days = 0
        if spec.date_mode == "single" and date_points:
            ordered_dates = sorted(date_points)
            start_date, end_date = ordered_dates[0], ordered_dates[-1]
            coverage_trade_days = len(ordered_dates)
        elif spec.date_mode == "range" and range_starts and range_ends:
            start_date = min(range_starts)
            end_date = max(range_ends)
        elif spec.date_mode == "trade_cal_csv":
            start_date, end_date, coverage_trade_days, date_points = _scan_trade_cal_dates(matched_paths)

        availability[dataset_id] = {
            "dataset_id": dataset_id,
            "dataset_name": spec.dataset_name,
            "available": bool(matched_paths),
            "roots": list(spec.roots),
            "missing_roots": missing_roots,
            "file_count": int(sum(1 for path in matched_paths if path.is_file())),
            "dir_count": int(sum(1 for path in matched_paths if path.is_dir())),
            "coverage_start_date": start_date,
            "coverage_end_date": end_date,
            "coverage_trade_days": int(coverage_trade_days),
            "coverage_years": _coverage_years(start_date, end_date),
            "sample_paths": [str(path.relative_to(repo_root)) for path in matched_paths[:5]],
            "notes": spec.notes,
        }
        if spec.date_mode in {"single", "trade_cal_csv"}:
            dataset_date_points[dataset_id] = set(date_points)

    trade_calendar_dates = dataset_date_points.get("trade_calendar", set())
    market_daily_dates = dataset_date_points.get("market_daily_all", set())
    if availability.get("trade_calendar") is not None and availability.get("market_daily_all") is not None:
        market_start = availability["market_daily_all"].get("coverage_start_date")
        market_end = availability["market_daily_all"].get("coverage_end_date")
        expected_trade_dates = {
            trade_date
            for trade_date in trade_calendar_dates
            if market_start is not None and market_end is not None and market_start <= trade_date <= market_end
        }
        available_trade_dates_in_range = {trade_date for trade_date in market_daily_dates if trade_date in expected_trade_dates}
        missing_trade_dates = sorted(expected_trade_dates - available_trade_dates_in_range)
        coverage_ratio_pct = round(float(len(available_trade_dates_in_range) / len(expected_trade_dates) * 100.0), 2) if expected_trade_dates else None
        availability["market_daily_all"]["expected_trade_days_in_range"] = int(len(expected_trade_dates))
        availability["market_daily_all"]["missing_trade_days_in_range"] = int(len(missing_trade_dates))
        availability["market_daily_all"]["coverage_ratio_pct"] = coverage_ratio_pct
        availability["market_daily_all"]["sample_missing_trade_dates"] = missing_trade_dates[:10]
    return availability


def build_strategy_data_inventory() -> dict[str, Any]:
    datasets = scan_dataset_availability()
    strategies = build_strategy_data_specs()
    strategy_rows: list[dict[str, Any]] = []
    for strategy_id, spec in strategies.items():
        missing_required = [dataset_id for dataset_id in spec.required_dataset_ids if not datasets.get(dataset_id, {}).get("available")]
        missing_optional = [dataset_id for dataset_id in spec.optional_dataset_ids if not datasets.get(dataset_id, {}).get("available")]
        market_daily_years = datasets.get("market_daily_all", {}).get("coverage_years")
        market_daily_missing_trade_days = int(datasets.get("market_daily_all", {}).get("missing_trade_days_in_range") or 0)
        market_daily_coverage_ratio_pct = datasets.get("market_daily_all", {}).get("coverage_ratio_pct")
        holder_snapshot_days = datasets.get("holder_snapshots", {}).get("coverage_trade_days")
        history_ready = (
            market_daily_years is not None
            and float(market_daily_years) >= float(spec.recommended_history_years)
            and market_daily_missing_trade_days == 0
        )
        row = {
            "strategy_id": strategy_id,
            "strategy_name": spec.strategy_name,
            "strategy_kind": spec.strategy_kind,
            "recommended_history_years": spec.recommended_history_years,
            "required_dataset_ids": list(spec.required_dataset_ids),
            "optional_dataset_ids": list(spec.optional_dataset_ids),
            "missing_required_dataset_ids": missing_required,
            "missing_optional_dataset_ids": missing_optional,
            "replay_ready": len(missing_required) == 0,
            "history_window_ready": bool(history_ready),
            "market_daily_coverage_years": market_daily_years,
            "market_daily_coverage_ratio_pct": market_daily_coverage_ratio_pct,
            "market_daily_missing_trade_days": market_daily_missing_trade_days,
            "holder_snapshot_trade_days": holder_snapshot_days,
            "notes": list(spec.notes),
        }
        if strategy_id == "holder_increase":
            row["api_replay_ready"] = len(missing_required) == 0 and len(missing_optional) == 0
            row["replay_mode"] = "export_replay"
        else:
            row["api_replay_ready"] = len(missing_required) == 0
            row["replay_mode"] = "price_daily_replay"
        strategy_rows.append(row)

    return {
        "repo_root": str(repo_root_dir()),
        "datasets": datasets,
        "strategies": {row["strategy_id"]: row for row in strategy_rows},
        "dataset_table": pd.DataFrame(list(datasets.values())).sort_values(["available", "dataset_id"], ascending=[False, True]),
        "strategy_table": pd.DataFrame(strategy_rows).sort_values(["replay_ready", "strategy_id"], ascending=[False, True]),
    }


def build_strategy_inventory_json_safe() -> dict[str, Any]:
    payload = build_strategy_data_inventory()
    return {
        "repo_root": payload["repo_root"],
        "datasets": json_safe(payload["datasets"]),
        "strategies": json_safe(payload["strategies"]),
    }


def dataset_specs_as_rows() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in build_dataset_catalog().values()]
