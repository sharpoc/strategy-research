import unittest

import pandas as pd

from tushare_stock_rater.config import load_config
from tushare_stock_rater.rebound_picker import _exit_outcome, _safe_return, build_rebound_candidates


def _make_history(ts_code: str, days: int, start_price: float, step: float, amount: float) -> pd.DataFrame:
    dates = pd.date_range(end="2026-04-21", periods=days, freq="B")
    closes = [start_price + i * step for i in range(days)]
    return pd.DataFrame(
        {
            "ts_code": ts_code,
            "trade_date": [item.strftime("%Y%m%d") for item in dates],
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "pre_close": closes,
            "pct_chg": [0.2] * days,
            "vol": [600000.0] * days,
            "amount": [amount] * days,
        }
    )


class ReboundPickerTests(unittest.TestCase):
    def test_build_rebound_candidates_prefers_safer_name(self):
        config = load_config()
        stock_basic = pd.DataFrame(
            [
                {"ts_code": "600001.SH", "symbol": "600001", "name": "候选A", "area": "上海", "industry": "有色", "market": "主板", "list_date": "20100101", "list_status": "L"},
                {"ts_code": "600002.SH", "symbol": "600002", "name": "候选B", "area": "上海", "industry": "有色", "market": "主板", "list_date": "20100101", "list_status": "L"},
                {"ts_code": "600003.SH", "symbol": "600003", "name": "同行C", "area": "上海", "industry": "有色", "market": "主板", "list_date": "20100101", "list_status": "L"},
                {"ts_code": "600004.SH", "symbol": "600004", "name": "同行D", "area": "上海", "industry": "有色", "market": "主板", "list_date": "20100101", "list_status": "L"},
                {"ts_code": "600005.SH", "symbol": "600005", "name": "ST风险", "area": "上海", "industry": "医药", "market": "主板", "list_date": "20100101", "list_status": "L"},
            ]
        )
        history = pd.concat(
            [
                _make_history("600001.SH", 65, 9.6, 0.005, 950000.0),
                _make_history("600002.SH", 65, 9.6, 0.005, 960000.0),
                _make_history("600003.SH", 65, 9.6, 0.005, 820000.0),
                _make_history("600004.SH", 65, 9.6, 0.005, 810000.0),
                _make_history("600005.SH", 65, 9.6, 0.005, 990000.0),
            ],
            ignore_index=True,
        )
        snapshot = pd.DataFrame(
            [
                {"ts_code": "600001.SH", "name": "候选A", "pre_close": 10.00, "open": 9.70, "high": 9.76, "low": 9.45, "close": 9.58, "vol": 680000.0, "amount": 920000.0},
                {"ts_code": "600002.SH", "name": "候选B", "pre_close": 10.00, "open": 9.78, "high": 9.80, "low": 9.42, "close": 9.48, "vol": 710000.0, "amount": 930000.0},
                {"ts_code": "600003.SH", "name": "同行C", "pre_close": 10.00, "open": 9.96, "high": 10.02, "low": 9.93, "close": 9.94, "vol": 330000.0, "amount": 350000.0},
                {"ts_code": "600004.SH", "name": "同行D", "pre_close": 10.00, "open": 10.02, "high": 10.04, "low": 9.98, "close": 10.01, "vol": 300000.0, "amount": 320000.0},
                {"ts_code": "600005.SH", "name": "ST风险", "pre_close": 10.00, "open": 9.60, "high": 9.62, "low": 9.20, "close": 9.32, "vol": 500000.0, "amount": 500000.0},
            ]
        )
        daily_basic = pd.DataFrame(
            [
                {"ts_code": "600001.SH", "turnover_rate": 3.5, "volume_ratio": 1.2, "circ_mv": 620000.0},
                {"ts_code": "600002.SH", "turnover_rate": 4.0, "volume_ratio": 1.3, "circ_mv": 650000.0},
                {"ts_code": "600003.SH", "turnover_rate": 1.1, "volume_ratio": 0.8, "circ_mv": 420000.0},
                {"ts_code": "600004.SH", "turnover_rate": 1.0, "volume_ratio": 0.7, "circ_mv": 430000.0},
                {"ts_code": "600005.SH", "turnover_rate": 5.0, "volume_ratio": 1.5, "circ_mv": 400000.0},
            ]
        )
        st_df = pd.DataFrame([{"ts_code": "600005.SH", "name": "ST风险", "trade_date": "20260422", "type": "ST", "type_name": "风险警示板"}])
        share_float_df = pd.DataFrame([{"ts_code": "600002.SH", "float_ratio": 8.5}])

        candidates = build_rebound_candidates(
            stock_basic=stock_basic,
            history=history,
            snapshot=snapshot,
            trade_date="20260422",
            mode="live",
            daily_basic=daily_basic,
            st_df=st_df,
            share_float_df=share_float_df,
            config=config,
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].ts_code, "600001.SH")
        self.assertNotIn("600002.SH", [item.ts_code for item in candidates])
        self.assertNotIn("600005.SH", [item.ts_code for item in candidates])
        self.assertIsNotNone(candidates[0].entry_low)
        self.assertIsNotNone(candidates[0].stop_loss)

    def test_safe_return(self):
        self.assertEqual(_safe_return(10, 10.5), 5.0)
        self.assertEqual(_safe_return(10, 9.5), -5.0)
        self.assertIsNone(_safe_return(None, 10))

    def test_exit_outcome_prefers_intraday_take_profit(self):
        settings = load_config()["rebound_strategy"]
        rule, ret, price = _exit_outcome(10, 10.05, 10.4, 9.9, settings)
        self.assertEqual(rule, "take_profit_intraday")
        self.assertEqual(ret, 2.0)
        self.assertAlmostEqual(price, 10.2)


if __name__ == "__main__":
    unittest.main()
