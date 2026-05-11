from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .client import TushareClient
from .config import load_config
from .models import DataBundle, StockIdentity, TableMeta
from .resolver import resolve_stock


def yyyymmdd(value: pd.Timestamp | datetime | str) -> str:
    if isinstance(value, str):
        return pd.to_datetime(value, format="%Y%m%d", errors="coerce").strftime("%Y%m%d")
    return pd.Timestamp(value).strftime("%Y%m%d")


def date_before(value: str, days: int) -> str:
    return (pd.to_datetime(value, format="%Y%m%d") - pd.Timedelta(days=days)).strftime("%Y%m%d")


def date_after(value: str, days: int) -> str:
    return (pd.to_datetime(value, format="%Y%m%d") + pd.Timedelta(days=days)).strftime("%Y%m%d")


def choose_latest_trade_date(trade_cal_df: pd.DataFrame, requested_date: str) -> str:
    if trade_cal_df is None or trade_cal_df.empty:
        raise ValueError("trade_cal returned no data; cannot determine trading day.")
    work = trade_cal_df.copy()
    if "cal_date" not in work.columns or "is_open" not in work.columns:
        raise ValueError("trade_cal missing cal_date/is_open columns.")
    work["cal_date"] = work["cal_date"].astype(str)
    open_days = work[(work["is_open"].astype(str) == "1") & (work["cal_date"] <= requested_date)].copy()
    if open_days.empty:
        raise ValueError(f"No open trading day found before {requested_date}.")
    return str(open_days.sort_values("cal_date").iloc[-1]["cal_date"])


class StockDataLoader:
    def __init__(
        self,
        client: TushareClient,
        config_path: Path | str | None = None,
        cutoff_hour: int = 20,
    ) -> None:
        self.client = client
        self.config = load_config(config_path)
        self.cutoff_hour = cutoff_hour

    def _default_requested_date(self) -> str:
        now = pd.Timestamp.now(tz="Asia/Shanghai")
        target = now
        if now.hour < self.cutoff_hour:
            target = now - pd.Timedelta(days=1)
        return target.strftime("%Y%m%d")

    def _load_stock_basic_all(self) -> tuple[pd.DataFrame, list[TableMeta]]:
        frames: list[pd.DataFrame] = []
        metas: list[TableMeta] = []
        fields = "ts_code,symbol,name,area,industry,market,list_date,list_status"
        for status in ["L", "P", "D"]:
            df, meta = self.client.call(
                "stock_basic",
                label=f"stock_basic_{status}",
                exchange="",
                list_status=status,
                fields=fields,
            )
            metas.append(meta)
            if not df.empty:
                df = df.copy()
                if "list_status" not in df.columns:
                    df["list_status"] = status
                frames.append(df)
        if not frames:
            return pd.DataFrame(), metas
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"], keep="first"), metas

    def _table_warnings(self, metas: list[TableMeta]) -> list[str]:
        warnings: list[str] = []
        for meta in metas:
            if meta.error:
                warnings.append(f"{meta.endpoint}: {meta.error}")
            elif meta.row_count == 0:
                warnings.append(f"{meta.endpoint}: 返回空表")
        return warnings

    def load(self, query: str, as_of_date: str = "", lookback_days: int = 252) -> DataBundle:
        requested_date = as_of_date.strip() if as_of_date else self._default_requested_date()
        requested_date = pd.to_datetime(requested_date, format="%Y%m%d").strftime("%Y%m%d")

        frames: dict[str, pd.DataFrame] = {}
        metas: list[TableMeta] = []

        stock_basic, stock_metas = self._load_stock_basic_all()
        metas.extend(stock_metas)
        frames["stock_basic"] = stock_basic
        stock = resolve_stock(query, stock_basic)

        cal_start = date_before(requested_date, 90)
        trade_cal, meta = self.client.call(
            "trade_cal",
            exchange="",
            start_date=cal_start,
            end_date=requested_date,
        )
        metas.append(meta)
        frames["trade_cal"] = trade_cal
        actual_as_of = choose_latest_trade_date(trade_cal, requested_date)

        price_start = date_before(actual_as_of, max(420, lookback_days * 2))
        finance_start = date_before(actual_as_of, 365 * 5)
        event_days = int(self.config.get("lookbacks", {}).get("event_days", 180))
        ann_days = int(self.config.get("lookbacks", {}).get("announcement_days", 180))
        unlock_days = int(self.config.get("lookbacks", {}).get("unlock_days", 60))
        event_start = date_before(actual_as_of, event_days)
        ann_start = date_before(actual_as_of, ann_days)
        unlock_end = date_after(actual_as_of, unlock_days)

        calls: list[tuple[str, str, dict[str, Any]]] = [
            ("daily", "daily", {"ts_code": stock.ts_code, "start_date": price_start, "end_date": actual_as_of}),
            ("adj_factor", "adj_factor", {"ts_code": stock.ts_code, "start_date": price_start, "end_date": actual_as_of}),
            ("daily_basic", "daily_basic", {"ts_code": stock.ts_code, "start_date": price_start, "end_date": actual_as_of}),
            ("moneyflow", "moneyflow", {"ts_code": stock.ts_code, "start_date": price_start, "end_date": actual_as_of}),
            ("income", "income", {"ts_code": stock.ts_code, "start_date": finance_start, "end_date": actual_as_of}),
            ("balancesheet", "balancesheet", {"ts_code": stock.ts_code, "start_date": finance_start, "end_date": actual_as_of}),
            ("cashflow", "cashflow", {"ts_code": stock.ts_code, "start_date": finance_start, "end_date": actual_as_of}),
            ("fina_indicator", "fina_indicator", {"ts_code": stock.ts_code, "start_date": finance_start, "end_date": actual_as_of}),
            ("forecast", "forecast", {"ts_code": stock.ts_code, "start_date": event_start, "end_date": actual_as_of}),
            ("express", "express", {"ts_code": stock.ts_code, "start_date": event_start, "end_date": actual_as_of}),
            ("anns_d", "anns_d", {"ts_code": stock.ts_code, "start_date": ann_start, "end_date": actual_as_of}),
            ("stk_holdertrade", "stk_holdertrade", {"ts_code": stock.ts_code, "start_date": event_start, "end_date": actual_as_of}),
            ("share_float", "share_float", {"ts_code": stock.ts_code, "start_date": actual_as_of, "end_date": unlock_end}),
            ("index_daily", "index_daily_000001.SH", {"ts_code": "000001.SH", "start_date": price_start, "end_date": actual_as_of}),
            ("index_dailybasic", "index_dailybasic_000001.SH", {"ts_code": "000001.SH", "start_date": price_start, "end_date": actual_as_of}),
            ("daily_basic", "daily_basic_market", {"trade_date": actual_as_of}),
        ]

        for endpoint, label, params in calls:
            df, meta = self.client.call(endpoint, label=label, **params)
            frames[label] = df
            metas.append(meta)

        return DataBundle(
            stock=stock,
            as_of_date=actual_as_of,
            requested_as_of_date=requested_date,
            lookback_days=lookback_days,
            frames=frames,
            table_meta=metas,
            warnings=self._table_warnings(metas),
        )
