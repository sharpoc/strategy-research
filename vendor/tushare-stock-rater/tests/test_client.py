import os
import unittest
from unittest import mock

from tushare_stock_rater.client import TushareClient


class TushareClientTests(unittest.TestCase):
    def test_configures_default_socket_timeout_from_env(self):
        with mock.patch.dict(os.environ, {"TUSHARE_HTTP_TIMEOUT_SECONDS": "12"}, clear=False):
            with mock.patch("tushare_stock_rater.client.socket.setdefaulttimeout") as set_timeout:
                client = TushareClient(token="valid-token", use_cache=False)

        self.assertEqual(client.http_timeout_seconds, 12.0)
        set_timeout.assert_called_once_with(12.0)


if __name__ == "__main__":
    unittest.main()
