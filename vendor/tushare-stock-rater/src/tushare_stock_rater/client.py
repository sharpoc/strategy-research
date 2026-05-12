from __future__ import annotations

import hashlib
import os
import socket
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .models import TableMeta
from .utils import PROJECT_ROOT, load_project_env


class TushareClientError(RuntimeError):
    pass


class TushareClient:
    def __init__(
        self,
        token: str | None = None,
        cache_dir: Path | str | None = None,
        retries: int = 2,
        sleep_sec: float = 0.2,
        use_cache: bool = True,
        custom_http_url: str | None = None,
    ) -> None:
        load_project_env(PROJECT_ROOT)
        self.token = (token or os.getenv("TUSHARE_TOKEN", "")).strip()
        if not self.token or self.token.startswith("PASTE_"):
            raise TushareClientError("Missing TUSHARE_TOKEN. 请在环境变量或 .env 中配置。")

        self.http_timeout_seconds = _configure_socket_timeout()
        self.cache_dir = Path(cache_dir or PROJECT_ROOT / "data" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.sleep_sec = sleep_sec
        self.use_cache = use_cache
        self.custom_http_url = (custom_http_url or os.getenv("TUSHARE_HTTP_URL", "")).strip()
        self._pro = None

    @property
    def pro(self):
        if self._pro is None:
            try:
                import tushare as ts
            except Exception as exc:
                raise TushareClientError("tushare package is not installed. 请先运行 python3 -m pip install -e .") from exc
            self._pro = ts.pro_api(self.token)
            if self.custom_http_url:
                self._configure_custom_endpoint()
        return self._pro

    def _configure_custom_endpoint(self) -> None:
        parsed = urlparse(self.custom_http_url)
        host = (parsed.hostname or "").strip()
        no_proxy_tokens: list[str] = []
        for value in [os.getenv("NO_PROXY", ""), os.getenv("no_proxy", ""), "127.0.0.1", "localhost", host]:
            for token in str(value).split(","):
                token = token.strip()
                if token and token not in no_proxy_tokens:
                    no_proxy_tokens.append(token)
        merged = ",".join(no_proxy_tokens)
        if merged:
            os.environ["NO_PROXY"] = merged
            os.environ["no_proxy"] = merged
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            os.environ.pop(key, None)
        try:
            self._pro._DataApi__token = self.token
            self._pro._DataApi__http_url = self.custom_http_url
        except Exception:
            pass

    def cache_key(self, endpoint: str, params: dict[str, Any]) -> str:
        clean = {k: "" if v is None else v for k, v in sorted(params.items())}
        payload = repr((endpoint, clean)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]

    def call(self, endpoint: str, label: str | None = None, **params: Any) -> tuple[pd.DataFrame, TableMeta]:
        clean_params = {key: value for key, value in params.items() if value is not None and value != ""}
        cache_name = f"{endpoint}_{self.cache_key(endpoint, clean_params)}.csv"
        cache_path = self.cache_dir / endpoint / cache_name
        table_label = label or endpoint
        should_use_cache = self.use_cache and endpoint not in {"rt_k", "rt_min", "rt_min_daily", "realtime_quote"}

        if should_use_cache and cache_path.exists():
            try:
                df = pd.read_csv(cache_path, dtype=str, keep_default_na=False)
                print(f"[tushare] cache {table_label} rows={len(df)} path={cache_path.name}", flush=True)
                return df, TableMeta(endpoint=table_label, row_count=len(df), cached=True, params=clean_params)
            except Exception:
                pass

        fn = getattr(self.pro, endpoint, None)
        if fn is None:
            return pd.DataFrame(), TableMeta(
                endpoint=table_label,
                row_count=0,
                cached=False,
                error=f"Tushare endpoint not found: {endpoint}",
                params=clean_params,
            )

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started_at = time.monotonic()
            print(
                f"[tushare] call {table_label} endpoint={endpoint} attempt={attempt + 1}/{self.retries + 1} params={clean_params}",
                flush=True,
            )
            try:
                df = fn(**clean_params)
                if df is None:
                    df = pd.DataFrame()
                else:
                    df = df.copy()
                if self.sleep_sec:
                    time.sleep(self.sleep_sec)
                if should_use_cache:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                elapsed = time.monotonic() - started_at
                print(f"[tushare] ok {table_label} rows={len(df)} elapsed={elapsed:.1f}s", flush=True)
                return df, TableMeta(endpoint=table_label, row_count=len(df), cached=False, params=clean_params)
            except Exception as exc:
                last_error = exc
                elapsed = time.monotonic() - started_at
                print(f"[tushare] error {table_label} elapsed={elapsed:.1f}s error={exc}", flush=True)
                if attempt < self.retries:
                    time.sleep(0.8 * (attempt + 1))

        return pd.DataFrame(), TableMeta(
            endpoint=table_label,
            row_count=0,
            cached=False,
            error=str(last_error),
            params=clean_params,
        )

    def smoke_test(self) -> list[TableMeta]:
        metas: list[TableMeta] = []
        trade_cal, meta = self.call(
            "trade_cal",
            label="trade_cal_smoke",
            exchange="",
            start_date="20240101",
            end_date="20240110",
        )
        metas.append(meta)
        stock_basic, meta = self.call(
            "stock_basic",
            label="stock_basic_smoke",
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        metas.append(meta)
        if trade_cal.empty or stock_basic.empty:
            raise TushareClientError("Tushare smoke test failed. trade_cal 或 stock_basic 返回为空。")
        return metas


def _configure_socket_timeout() -> float:
    raw = os.getenv("TUSHARE_HTTP_TIMEOUT_SECONDS", "45").strip()
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 45.0
    if timeout > 0:
        socket.setdefaulttimeout(timeout)
    return timeout
