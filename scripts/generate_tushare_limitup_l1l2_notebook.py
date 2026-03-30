from __future__ import annotations

import json
import textwrap
from pathlib import Path


def lines(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def build_notebook() -> dict:
    return {
        "cells": [
            markdown_cell(
                """
                # Experiment: 涨停后 L1 / L2 更高低点策略

                Objective:
                - 从全市场里找“之前出现过涨停，随后形成两次回调，且 `L2 > L1`”的股票。
                - 每天只输出这个策略里最强的一只。
                - 导出候选清单、最强个股和可复核的原始文件，方便复盘。
                """
            ),
            markdown_cell(
                """
                ## 使用方式

                1. 修改下面配置单元格后，直接 `Run All`。
                2. Notebook 会调用仓库里的同一套策略脚本逻辑，不会和命令行跑出两套结果。
                3. 结果会写到 `output/jupyter-notebook/tushare_limitup_l1l2_exports/`。
                """
            ),
            code_cell(
                """
                from __future__ import annotations

                import json
                import os
                import sys
                from pathlib import Path

                import pandas as pd

                pd.set_option("display.max_columns", 120)
                pd.set_option("display.width", 240)
                pd.set_option("display.max_colwidth", 120)

                CWD = Path.cwd().resolve()
                if CWD.name == "jupyter-notebook" and CWD.parent.name == "output":
                    REPO_ROOT = CWD.parent.parent
                else:
                    REPO_ROOT = CWD

                SCRIPT_DIR = REPO_ROOT / "scripts"
                if str(SCRIPT_DIR) not in sys.path:
                    sys.path.insert(0, str(SCRIPT_DIR))

                TODAY = pd.Timestamp.today().normalize()
                TODAY_STR = TODAY.strftime("%Y%m%d")
                print({"today": TODAY_STR, "repo_root": str(REPO_ROOT)})
                """
            ),
            code_cell(
                """
                # 只需要改这一格。
                END_DATE = TODAY_STR
                HISTORY_BARS = 100
                MONEYFLOW_LOOKBACK_DAYS = 5
                RECENT_BUY_WINDOW = 0
                MIN_SCORE = 55.0
                CUTOFF_HOUR = 20
                SHOW_TOP = 10

                if not os.getenv("TUSHARE_TOKEN", "").strip():
                    print("提示：当前环境未检测到 TUSHARE_TOKEN，运行时会直接报错。")

                {
                    "end_date": END_DATE,
                    "history_bars": HISTORY_BARS,
                    "moneyflow_lookback_days": MONEYFLOW_LOOKBACK_DAYS,
                    "recent_buy_window": RECENT_BUY_WINDOW,
                    "min_score": MIN_SCORE,
                    "cutoff_hour": CUTOFF_HOUR,
                    "show_top": SHOW_TOP,
                }
                """
            ),
            code_cell(
                """
                from run_tushare_limitup_l1l2_strategy import (
                    display_columns,
                    ensure_columns,
                    json_safe,
                    run_limitup_l1l2_screen,
                )

                result = run_limitup_l1l2_screen(
                    end_date=END_DATE,
                    history_bars=HISTORY_BARS,
                    moneyflow_lookback_days=MONEYFLOW_LOOKBACK_DAYS,
                    recent_buy_window=RECENT_BUY_WINDOW,
                    min_score=MIN_SCORE,
                    cutoff_hour=CUTOFF_HOUR,
                )

                summary = result["summary"]
                strategy_candidates = result["strategy_candidates"]
                best_pick_candidate = result["best_pick_candidate"]

                print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))

                candidate_cols = [c for c in display_columns() if c in strategy_candidates.columns]
                best_cols = [c for c in display_columns() if c in best_pick_candidate.columns]

                strategy_candidates = ensure_columns(strategy_candidates, candidate_cols)
                best_pick_candidate = ensure_columns(best_pick_candidate, best_cols)

                display(strategy_candidates[candidate_cols].head(SHOW_TOP))
                display(best_pick_candidate[best_cols].head(1))
                """
            ),
            markdown_cell(
                """
                ## 当前策略定义

                - 硬条件：先有涨停，再有两次回调低点，且 `L2 > L1`。
                - `Impulse / 放量 / EMA` 不再是硬门槛，只作为辅助排序项。
                - 当天最终只取这个策略池里排序第 1 的股票，也就是 `best_pick_candidate`。
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "output" / "jupyter-notebook" / "tushare-limitup-l1l2-screening.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
