import unittest

import pandas as pd

from tushare_stock_rater.resolver import normalize_ts_code, resolve_stock


class ResolverTests(unittest.TestCase):
    def test_normalize_ts_code(self):
        self.assertEqual(normalize_ts_code("600519"), "600519.SH")
        self.assertEqual(normalize_ts_code("000001"), "000001.SZ")
        self.assertEqual(normalize_ts_code("830799"), "830799.BJ")
        self.assertEqual(normalize_ts_code("600519.SH"), "600519.SH")

    def test_resolve_by_name(self):
        stock_basic = pd.DataFrame(
            [
                {
                    "ts_code": "600519.SH",
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "area": "贵州",
                    "industry": "白酒",
                    "market": "主板",
                    "list_date": "20010827",
                    "list_status": "L",
                }
            ]
        )
        stock = resolve_stock("贵州茅台", stock_basic)
        self.assertEqual(stock.ts_code, "600519.SH")


if __name__ == "__main__":
    unittest.main()
