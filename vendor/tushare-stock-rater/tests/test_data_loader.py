import unittest

import pandas as pd

from tushare_stock_rater.data_loader import choose_latest_trade_date


class DataLoaderTests(unittest.TestCase):
    def test_choose_latest_trade_date_falls_back_from_weekend(self):
        trade_cal = pd.DataFrame(
            [
                {"cal_date": "20260417", "is_open": "1"},
                {"cal_date": "20260418", "is_open": "0"},
                {"cal_date": "20260419", "is_open": "0"},
            ]
        )
        self.assertEqual(choose_latest_trade_date(trade_cal, "20260419"), "20260417")


if __name__ == "__main__":
    unittest.main()
