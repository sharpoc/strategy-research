from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the holder-increase screening notebook and print top picks.")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("output/jupyter-notebook/tushare-holder-increase-screening.ipynb"),
        help="Notebook path to execute.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=5,
        help="How many rows to print from the final candidates table.",
    )
    parser.add_argument("--end-date", default="", help="Requested end date in YYYYMMDD. Default: today with 20:00 cutoff.")
    parser.add_argument(
        "--ann-start-date",
        default="",
        help="Requested holder announcement start date in YYYYMMDD. Default: end-date minus 45 days.",
    )
    return parser.parse_args()


def display(obj=None) -> None:
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False))
            return
    except Exception:
        pass
    print(obj)


def _normalize_trade_day(value: str) -> str:
    raw = str(value or "").strip().replace("-", "")
    if not raw:
        return ""
    try:
        return datetime.strptime(raw, "%Y%m%d").strftime("%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"Invalid trade date: {value}") from exc


def _default_ann_start(end_date: str) -> str:
    if not end_date:
        return ""
    return (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=45)).strftime("%Y%m%d")


def _override_notebook_source(
    source: str,
    *,
    end_date: str,
    ann_start_date: str,
) -> str:
    if end_date and "TODAY = pd.Timestamp.today().normalize()" in source:
        override = (
            f'TODAY = pd.Timestamp("{end_date}")\n'
            'TODAY = TODAY.normalize()\n'
            'TODAY_STR = TODAY.strftime("%Y%m%d")\n'
            'DEFAULT_ANN_START = (TODAY - pd.Timedelta(days=45)).strftime("%Y%m%d")\n'
            'DEFAULT_PRICE_START = (TODAY - pd.Timedelta(days=420)).strftime("%Y%m%d")'
        )
        source = source.replace(
            'TODAY = pd.Timestamp.today().normalize()\n'
            'TODAY_STR = TODAY.strftime("%Y%m%d")\n'
            'DEFAULT_ANN_START = (TODAY - pd.Timedelta(days=45)).strftime("%Y%m%d")\n'
            'DEFAULT_PRICE_START = (TODAY - pd.Timedelta(days=420)).strftime("%Y%m%d")',
            override,
        )
    if ann_start_date:
        source = source.replace('ANN_START_DATE = DEFAULT_ANN_START', f'ANN_START_DATE = "{ann_start_date}"')
    if end_date:
        source = source.replace('END_DATE = TODAY_STR', f'END_DATE = "{end_date}"')
        source = source.replace(
            'screen_end_date = choose_screen_end_date(pd.Timestamp.now(), END_DATE, TODAY_STR, MARKET_DATA_CUTOFF_HOUR)',
            'screen_end_date = END_DATE',
        )
        source = source.replace(
            'export_dir = OUTPUT_DIR / f"holder_increase_screen_{TODAY_STR}"',
            'export_dir = OUTPUT_DIR / f"holder_increase_screen_{screen_end_date}"',
        )
    return source


def execute_notebook(notebook_path: Path, *, end_date: str, ann_start_date: str) -> dict:
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    context: dict = {"__name__": "__main__", "display": display}

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        source = _override_notebook_source(
            source,
            end_date=end_date,
            ann_start_date=ann_start_date,
        )
        print(f"===== running cell {index} =====")
        exec(compile(source, f"cell_{index}", "exec"), context)
    return context


def main() -> None:
    args = parse_args()
    end_date = _normalize_trade_day(args.end_date)
    ann_start_date = _normalize_trade_day(args.ann_start_date) or _default_ann_start(end_date)

    if not os.getenv("TUSHARE_TOKEN"):
        raise SystemExit("Missing TUSHARE_TOKEN in environment.")

    try:
        context = execute_notebook(args.notebook, end_date=end_date, ann_start_date=ann_start_date)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

    export_dir = context.get("export_dir")
    stable_candidates = context.get("stable_candidates")
    final_candidates = context.get("final_candidates")
    aggressive_candidates = context.get("aggressive_candidates")
    best_pick_candidate = context.get("best_pick_candidate")
    today_direction = context.get("today_direction")

    if stable_candidates is None:
        stable_candidates = final_candidates
    if stable_candidates is None:
        raise SystemExit("Notebook execution finished but stable/final candidates were not produced.")

    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit(f"pandas is required to print results: {exc}")

    if not isinstance(stable_candidates, pd.DataFrame):
        stable_candidates = pd.DataFrame(stable_candidates)
    if aggressive_candidates is not None and not isinstance(aggressive_candidates, pd.DataFrame):
        aggressive_candidates = pd.DataFrame(aggressive_candidates)
    if best_pick_candidate is not None and not isinstance(best_pick_candidate, pd.DataFrame):
        best_pick_candidate = pd.DataFrame(best_pick_candidate)

    if best_pick_candidate is None or best_pick_candidate.empty:
        best_pick_candidate = stable_candidates.head(1).copy()

    if export_dir:
        export_dir_path = Path(export_dir)
        export_dir_path.mkdir(parents=True, exist_ok=True)
        best_pick_candidate.to_csv(export_dir_path / "best_pick_candidate.csv", index=False)

    columns = [
        column
        for column in [
            "ts_code",
            "name",
            "industry",
            "market_regime",
            "preferred_pool",
            "priority_score",
            "preliminary_score",
            "stable_score",
            "aggressive_score",
            "final_score",
            "chip_score",
            "event_bonus_score",
            "earnings_score",
            "value_score",
            "reversal_score",
            "fund_flow_score",
            "overheat_penalty_score",
            "risk_penalty_score",
            "active_reduction_plan_flag",
            "unlock_risk_veto",
            "winner_rate",
            "winner_rate_change_5d",
        ]
        if column in stable_candidates.columns
    ]

    if today_direction:
        print(f"===== today direction: {today_direction} =====")
    print("===== stable candidates =====")
    print(stable_candidates[columns].head(args.show_top).to_string(index=False))
    if aggressive_candidates is not None and not aggressive_candidates.empty:
        print("===== aggressive candidates =====")
        print(aggressive_candidates[columns].head(args.show_top).to_string(index=False))
    print("===== best pick =====")
    print(best_pick_candidate[columns].to_string(index=False))

    if export_dir:
        print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
