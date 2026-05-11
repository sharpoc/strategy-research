from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class StockIdentity:
    ts_code: str
    symbol: str
    name: str
    area: str = ""
    industry: str = ""
    market: str = ""
    list_date: str = ""
    list_status: str = "L"


@dataclass
class TableMeta:
    endpoint: str
    row_count: int
    cached: bool = False
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataBundle:
    stock: StockIdentity
    as_of_date: str
    requested_as_of_date: str
    lookback_days: int
    frames: dict[str, pd.DataFrame]
    table_meta: list[TableMeta]
    warnings: list[str] = field(default_factory=list)


@dataclass
class FeatureSet:
    stock: StockIdentity
    as_of_date: str
    values: dict[str, Any]
    evidence: dict[str, list[str]]
    positive_points: list[str]
    negative_points: list[str]
    risk_flags: list[str]
    hard_risks: list[str]
    severe_risks: list[str]
    missing_tables: list[str]
    data_warnings: list[str]


@dataclass
class SectionScore:
    key: str
    name: str
    max_score: float
    score: float
    evidence: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        if self.max_score <= 0:
            return 0.0
        return max(0.0, min(1.0, self.score / self.max_score))


@dataclass
class RatingResult:
    stock: StockIdentity
    as_of_date: str
    total_score: float
    raw_total_score: float
    confidence_score: float
    rating: str
    verdict: str
    section_scores: list[SectionScore]
    positive_points: list[str]
    negative_points: list[str]
    risk_flags: list[str]
    hard_risks: list[str]
    severe_risks: list[str]
    missing_tables: list[str]
    data_warnings: list[str]
    features: dict[str, Any]


@dataclass
class ReportPaths:
    output_dir: Path
    report_md: Path
    result_json: Path
    features_csv: Path
    data_meta_json: Path
