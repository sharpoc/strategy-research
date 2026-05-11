from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from financial_report_catalyst_strategy import (  # noqa: E402
    FinancialReportCatalystConfig,
    build_financial_report_candidates,
    run_financial_report_catalyst_screening,
)


class FinancialReportCatalystStrategyTest(unittest.TestCase):
    def test_empty_disclosure_frames_return_empty_result_instead_of_crashing(self) -> None:
        class EmptyDisclosurePro:
            def disclosure_date(self, **kwargs):
                return pd.DataFrame()

            def stock_basic(self, **kwargs):
                return pd.DataFrame(columns=["ts_code", "symbol", "name", "area", "industry", "market", "list_date", "list_status"])

            def daily_basic(self, **kwargs):
                return pd.DataFrame()

            def daily(self, **kwargs):
                return pd.DataFrame()

            def forecast(self, **kwargs):
                return pd.DataFrame()

            def express(self, **kwargs):
                return pd.DataFrame()

            def trade_cal(self, **kwargs):
                return pd.DataFrame()

            def fina_indicator(self, **kwargs):
                return pd.DataFrame()

        config = FinancialReportCatalystConfig.for_end_date("20260428", api_sleep_sec=0.0)

        result = run_financial_report_catalyst_screening(
            config=config,
            pro=EmptyDisclosurePro(),
            export_results=False,
        )

        self.assertEqual(result["screen_summary"]["candidate_count"], 0)
        self.assertTrue(result["candidates"].empty)

    def test_selects_nearby_profitable_low_position_stock(self) -> None:
        config = FinancialReportCatalystConfig.for_end_date(
            "20260428",
            report_lookahead_days=30,
            recent_confirmed_days=7,
            price_min=3.0,
            price_max=35.0,
            min_amount=40000.0,
        )
        stock_basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "好业绩", "industry": "制造", "list_status": "L"},
                {"ts_code": "000002.SZ", "name": "低质低价", "industry": "地产", "list_status": "L"},
                {"ts_code": "000003.SZ", "name": "负面预告", "industry": "化工", "list_status": "L"},
                {"ts_code": "000004.SZ", "name": "太贵高位", "industry": "软件", "list_status": "L"},
            ]
        )
        disclosure = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "ann_date": "20260401", "end_date": "20260331", "pre_date": "20260508", "actual_date": ""},
                {"ts_code": "000002.SZ", "ann_date": "20260401", "end_date": "20260331", "pre_date": "20260505", "actual_date": ""},
                {"ts_code": "000003.SZ", "ann_date": "20260401", "end_date": "20260331", "pre_date": "20260504", "actual_date": ""},
                {"ts_code": "000004.SZ", "ann_date": "20260401", "end_date": "20260331", "pre_date": "20260504", "actual_date": ""},
            ]
        )
        forecast = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "ann_date": "20260420", "end_date": "20260331", "type": "预增", "p_change_min": 35, "p_change_max": 60},
                {"ts_code": "000003.SZ", "ann_date": "20260421", "end_date": "20260331", "type": "预减", "p_change_min": -30, "p_change_max": -10},
            ]
        )
        express = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "ann_date": "20260418", "end_date": "20260331", "revenue": 100, "n_income": 20, "diluted_roe": 9, "yoy_net_profit": 42},
                {"ts_code": "000002.SZ", "ann_date": "20260418", "end_date": "20260331", "revenue": 100, "n_income": 2, "diluted_roe": 2, "yoy_net_profit": -4},
            ]
        )
        fina_indicator = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "ann_date": "20260418", "end_date": "20260331", "roe": 10, "dt_netprofit_yoy": 36, "netprofit_yoy": 42, "or_yoy": 18, "gross_margin": 32, "ocfps": 0.7},
                {"ts_code": "000002.SZ", "ann_date": "20260418", "end_date": "20260331", "roe": 2, "dt_netprofit_yoy": -8, "netprofit_yoy": -4, "or_yoy": 1, "gross_margin": 9, "ocfps": -0.1},
                {"ts_code": "000004.SZ", "ann_date": "20260418", "end_date": "20260331", "roe": 14, "dt_netprofit_yoy": 55, "netprofit_yoy": 60, "or_yoy": 30, "gross_margin": 55, "ocfps": 1.5},
            ]
        )
        daily_basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20260428", "close": 12.5, "turnover_rate": 2.1, "volume_ratio": 1.3, "pe_ttm": 22, "pb": 2.2, "ps_ttm": 2.0, "total_mv": 1_200_000, "circ_mv": 900_000},
                {"ts_code": "000002.SZ", "trade_date": "20260428", "close": 4.2, "turnover_rate": 1.2, "volume_ratio": 0.9, "pe_ttm": 18, "pb": 0.8, "ps_ttm": 0.5, "total_mv": 500_000, "circ_mv": 420_000},
                {"ts_code": "000003.SZ", "trade_date": "20260428", "close": 8.0, "turnover_rate": 1.5, "volume_ratio": 1.0, "pe_ttm": 20, "pb": 1.1, "ps_ttm": 1.0, "total_mv": 600_000, "circ_mv": 500_000},
                {"ts_code": "000004.SZ", "trade_date": "20260428", "close": 58.0, "turnover_rate": 2.5, "volume_ratio": 1.8, "pe_ttm": 45, "pb": 6.5, "ps_ttm": 8.5, "total_mv": 2_000_000, "circ_mv": 1_800_000},
            ]
        )
        daily = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20260428", "open": 12.1, "high": 12.8, "low": 11.9, "close": 12.5, "pre_close": 12.0, "change": 0.5, "pct_chg": 4.17, "vol": 80000, "amount": 90000},
                {"ts_code": "000002.SZ", "trade_date": "20260428", "open": 4.1, "high": 4.3, "low": 4.0, "close": 4.2, "pre_close": 4.2, "change": 0, "pct_chg": 0, "vol": 60000, "amount": 45000},
                {"ts_code": "000003.SZ", "trade_date": "20260428", "open": 8.2, "high": 8.3, "low": 7.9, "close": 8.0, "pre_close": 8.2, "change": -0.2, "pct_chg": -2.44, "vol": 70000, "amount": 60000},
                {"ts_code": "000004.SZ", "trade_date": "20260428", "open": 57.0, "high": 59.0, "low": 56.0, "close": 58.0, "pre_close": 57.0, "change": 1.0, "pct_chg": 1.75, "vol": 90000, "amount": 120000},
            ]
        )
        history = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20260315", "close": 11.0},
                {"ts_code": "000001.SZ", "trade_date": "20260320", "close": 15.0},
                {"ts_code": "000001.SZ", "trade_date": "20260401", "close": 10.0},
                {"ts_code": "000001.SZ", "trade_date": "20260415", "close": 11.0},
                {"ts_code": "000001.SZ", "trade_date": "20260428", "close": 12.5},
                {"ts_code": "000002.SZ", "trade_date": "20260401", "close": 5.0},
                {"ts_code": "000002.SZ", "trade_date": "20260415", "close": 4.6},
                {"ts_code": "000002.SZ", "trade_date": "20260428", "close": 4.2},
                {"ts_code": "000003.SZ", "trade_date": "20260401", "close": 8.4},
                {"ts_code": "000003.SZ", "trade_date": "20260415", "close": 8.3},
                {"ts_code": "000003.SZ", "trade_date": "20260428", "close": 8.0},
                {"ts_code": "000004.SZ", "trade_date": "20260401", "close": 35.0},
                {"ts_code": "000004.SZ", "trade_date": "20260415", "close": 48.0},
                {"ts_code": "000004.SZ", "trade_date": "20260428", "close": 58.0},
            ]
        )

        result = build_financial_report_candidates(
            config=config,
            stock_basic=stock_basic,
            disclosure=disclosure,
            forecast=forecast,
            express=express,
            fina_indicator=fina_indicator,
            daily_basic=daily_basic,
            daily=daily,
            history=history,
        )

        self.assertEqual(result.iloc[0]["ts_code"], "000001.SZ")
        self.assertEqual(result.iloc[0]["name"], "好业绩")
        self.assertGreater(result.iloc[0]["financial_report_score"], 0)
        self.assertNotIn("000002.SZ", set(result["ts_code"]))
        self.assertNotIn("000003.SZ", set(result["ts_code"]))
        self.assertNotIn("000004.SZ", set(result["ts_code"]))


if __name__ == "__main__":
    unittest.main()
