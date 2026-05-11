from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from .models import StockIdentity


def normalize_ts_code(value: str) -> str:
    code = str(value or "").strip().upper()
    if not code:
        return code
    if "." in code:
        return code
    if re.fullmatch(r"\d{6}", code):
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
        if code.startswith(("5", "6", "9")):
            return f"{code}.SH"
        return f"{code}.SZ"
    return code


def _row_to_identity(row: pd.Series) -> StockIdentity:
    data = {str(k): "" if pd.isna(v) else str(v) for k, v in row.items()}
    return StockIdentity(
        ts_code=data.get("ts_code", ""),
        symbol=data.get("symbol", ""),
        name=data.get("name", ""),
        area=data.get("area", ""),
        industry=data.get("industry", ""),
        market=data.get("market", ""),
        list_date=data.get("list_date", ""),
        list_status=data.get("list_status", "L") or "L",
    )


def resolve_stock(query: str, stock_basic_df: pd.DataFrame) -> StockIdentity:
    if stock_basic_df is None or stock_basic_df.empty:
        raise ValueError("stock_basic returned no data; cannot resolve stock.")
    work = stock_basic_df.copy().fillna("")
    for column in ["ts_code", "symbol", "name"]:
        if column not in work.columns:
            work[column] = ""

    normalized = normalize_ts_code(query)
    exact = work[
        (work["ts_code"].astype(str).str.upper() == normalized)
        | (work["symbol"].astype(str).str.upper() == str(query).strip().upper())
        | (work["name"].astype(str) == str(query).strip())
    ]
    if not exact.empty:
        return _row_to_identity(exact.iloc[0])

    partial = work[work["name"].astype(str).str.contains(str(query).strip(), regex=False, na=False)]
    if len(partial) == 1:
        return _row_to_identity(partial.iloc[0])
    if len(partial) > 1:
        candidates = ", ".join(
            f"{row.ts_code}({row.name})"
            for row in partial[["ts_code", "name"]].head(10).itertuples(index=False)
        )
        raise ValueError(f"股票名称匹配到多个候选，请输入更精确的代码：{candidates}")

    raise ValueError(f"未找到股票：{query}")


def collect_codes(values: Iterable[str]) -> list[str]:
    codes: list[str] = []
    for value in values:
        for piece in re.split(r"[\s,，]+", str(value).strip()):
            if piece:
                codes.append(piece)
    return codes
