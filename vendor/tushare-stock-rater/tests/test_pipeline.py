import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tushare_stock_rater.features import build_features
from tushare_stock_rater.models import DataBundle, StockIdentity, TableMeta
from tushare_stock_rater.report import write_report_outputs
from tushare_stock_rater.scoring import score_stock


def mock_bundle(name: str = "样例股份", severe: bool = False) -> DataBundle:
    dates = pd.date_range(end="2026-04-17", periods=300, freq="B")
    trade_dates = [d.strftime("%Y%m%d") for d in dates]
    close = [80 + i * 0.08 for i in range(len(trade_dates))]
    daily = pd.DataFrame(
        {
            "ts_code": "600519.SH",
            "trade_date": trade_dates,
            "open": close,
            "high": [x * 1.01 for x in close],
            "low": [x * 0.99 for x in close],
            "close": close,
            "pre_close": close,
            "pct_chg": [0.1] * len(close),
            "vol": [100000 + i * 10 for i in range(len(close))],
            "amount": [500000] * len(close),
        }
    )
    adj = pd.DataFrame({"ts_code": "600519.SH", "trade_date": trade_dates, "adj_factor": [1.0] * len(trade_dates)})
    daily_basic = pd.DataFrame(
        {
            "ts_code": "600519.SH",
            "trade_date": trade_dates,
            "turnover_rate": [2.5] * len(trade_dates),
            "pe_ttm": [22] * len(trade_dates),
            "pb": [4] * len(trade_dates),
            "ps_ttm": [8] * len(trade_dates),
            "dv_ttm": [2.0] * len(trade_dates),
        }
    )
    stock_basic = pd.DataFrame(
        [
            {
                "ts_code": "600519.SH",
                "symbol": "600519",
                "name": name,
                "industry": "白酒",
                "market": "主板",
                "list_date": "20010827",
                "list_status": "L",
            }
        ]
    )
    market_basic = pd.DataFrame(
        {
            "ts_code": [f"6005{i:02d}.SH" for i in range(10)] + ["600519.SH"],
            "trade_date": ["20260417"] * 11,
            "pe_ttm": [12, 15, 18, 20, 24, 30, 35, 40, 45, 60, 22],
            "pb": [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 4],
            "ps_ttm": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8],
        }
    )
    market_stock_basic = pd.concat(
        [
            stock_basic,
            pd.DataFrame(
                {
                    "ts_code": [f"6005{i:02d}.SH" for i in range(10)],
                    "symbol": [f"6005{i:02d}" for i in range(10)],
                    "name": [f"同业{i}" for i in range(10)],
                    "industry": ["白酒"] * 10,
                    "market": ["主板"] * 10,
                    "list_date": ["20100101"] * 10,
                    "list_status": ["L"] * 10,
                }
            ),
        ],
        ignore_index=True,
    )
    fina = pd.DataFrame(
        {
            "ts_code": ["600519.SH"] * 8,
            "end_date": [f"202{i}1231" for i in range(5, -3, -1)],
            "ann_date": [f"202{i}0331" for i in range(6, -2, -1)],
            "roe": [24] * 8,
            "grossprofit_margin": [88] * 8,
            "netprofit_margin": [45] * 8,
            "debt_to_assets": [25] * 8,
            "tr_yoy": [14] * 8,
            "dt_netprofit_yoy": [18] * 8,
            "q_netprofit_yoy": [16] * 8,
        }
    )
    cashflow = pd.DataFrame({"ts_code": ["600519.SH"], "end_date": ["20251231"], "ann_date": ["20260331"], "n_cashflow_act": [120.0]})
    income = pd.DataFrame({"ts_code": ["600519.SH"], "end_date": ["20251231"], "ann_date": ["20260331"], "n_income_attr_p": [100.0]})
    moneyflow = pd.DataFrame({"ts_code": ["600519.SH"] * 30, "trade_date": trade_dates[-30:], "net_mf_amount": [1000.0] * 30})
    forecast = pd.DataFrame({"ts_code": ["600519.SH"], "ann_date": ["20260401"], "type": ["预增"], "p_change_min": [10], "p_change_max": [20]})
    frames = {
        "daily": daily,
        "adj_factor": adj,
        "daily_basic": daily_basic,
        "stock_basic": market_stock_basic,
        "daily_basic_market": market_basic,
        "fina_indicator": fina,
        "cashflow": cashflow,
        "income": income,
        "moneyflow": moneyflow,
        "forecast": forecast,
        "express": pd.DataFrame(),
        "anns_d": pd.DataFrame(),
        "stk_holdertrade": pd.DataFrame(),
        "share_float": pd.DataFrame(),
        "trade_cal": pd.DataFrame(),
        "index_daily": pd.DataFrame(),
        "index_dailybasic": pd.DataFrame(),
    }
    stock = StockIdentity(
        ts_code="600519.SH",
        symbol="600519",
        name=name,
        industry="白酒",
        market="主板",
        list_date="20010827",
        list_status="L",
    )
    if severe:
        stock.name = "ST样例"
    metas = [TableMeta(endpoint=key, row_count=len(value)) for key, value in frames.items()]
    return DataBundle(stock=stock, as_of_date="20260417", requested_as_of_date="20260417", lookback_days=252, frames=frames, table_meta=metas)


class PipelineTests(unittest.TestCase):
    def test_mock_pipeline_writes_outputs(self):
        bundle = mock_bundle()
        features = build_features(bundle)
        result = score_stock(features)
        self.assertGreater(result.total_score, 60)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_report_outputs(result, bundle, Path(tmp))
            self.assertTrue(paths.report_md.exists())
            self.assertTrue(paths.result_json.exists())
            self.assertTrue(paths.features_csv.exists())
            self.assertTrue(paths.data_meta_json.exists())

    def test_severe_risk_caps_rating(self):
        bundle = mock_bundle(severe=True)
        result = score_stock(build_features(bundle))
        self.assertLessEqual(result.total_score, 59)
        self.assertEqual(result.rating, "D")


if __name__ == "__main__":
    unittest.main()
