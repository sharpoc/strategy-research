from __future__ import annotations

from pathlib import Path

from .client import TushareClient
from .data_loader import StockDataLoader
from .features import build_features
from .models import DataBundle, FeatureSet, RatingResult, ReportPaths
from .report import write_report_outputs
from .scoring import score_stock


def analyze_stock(
    query: str,
    as_of_date: str = "",
    lookback_days: int = 252,
    out_dir: Path | str = "reports",
    config_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    no_cache: bool = False,
    retries: int = 2,
    sleep_sec: float = 0.2,
    cutoff_hour: int = 20,
) -> tuple[RatingResult, FeatureSet, DataBundle, ReportPaths]:
    client = TushareClient(
        cache_dir=cache_dir,
        retries=retries,
        sleep_sec=sleep_sec,
        use_cache=not no_cache,
    )
    loader = StockDataLoader(client=client, config_path=config_path, cutoff_hour=cutoff_hour)
    bundle = loader.load(query=query, as_of_date=as_of_date, lookback_days=lookback_days)
    features = build_features(bundle)
    result = score_stock(features, str(config_path) if config_path else None)
    paths = write_report_outputs(result, bundle, out_dir)
    return result, features, bundle, paths
