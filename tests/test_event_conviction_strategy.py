from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from event_conviction_strategy import EventConvictionConfig, select_top1_candidate  # noqa: E402


class EventConvictionStrategyTest(unittest.TestCase):
    def test_selects_highest_scored_candidate_even_below_publish_threshold(self) -> None:
        config = EventConvictionConfig.for_end_date("20260428", min_publish_score=48.0)
        scored = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "高分候选", "total_score": 43.2},
                {"ts_code": "000002.SZ", "name": "次高候选", "total_score": 39.8},
            ]
        )

        selected = select_top1_candidate(scored, config)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["ts_code"], "000001.SZ")


if __name__ == "__main__":
    unittest.main()
