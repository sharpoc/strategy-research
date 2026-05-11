from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import PROJECT_ROOT


DEFAULT_CONFIG: dict[str, Any] = {
    "section_weights": {
        "financial_quality": 25,
        "growth": 15,
        "valuation": 15,
        "technical": 20,
        "moneyflow": 15,
        "events": 10,
    },
    "rating_thresholds": {"A": 80, "B": 70, "C": 60},
    "risk_caps": {
        "hard_risk_max_rating": "C",
        "hard_risk_max_score": 69,
        "severe_risk_max_rating": "D",
        "severe_risk_max_score": 59,
    },
    "lookbacks": {
        "default_price_days": 252,
        "financial_quarters": 8,
        "event_days": 180,
        "announcement_days": 180,
        "unlock_days": 60,
    },
    "hard_risk_thresholds": {
        "large_unlock_ratio_60d": 8,
        "near_unlock_ratio_30d": 3,
        "core_decrease_ratio_180d": 0.8,
        "cashflow_to_profit_min": -0.2,
    },
    "rebound_strategy": {
        "markets": ["主板", "创业板"],
        "history_trade_days": 65,
        "min_list_days": 120,
        "min_price": 4.0,
        "min_drop_pct": -6.8,
        "target_drop_pct": -4.8,
        "max_drop_pct": -3.0,
        "min_excess_drop_pct": -1.5,
        "strong_excess_drop_pct": -3.5,
        "min_avg_amount_20d": 500000.0,
        "strong_avg_amount_20d": 1200000.0,
        "min_latest_amount": 300000.0,
        "min_amount_ratio_20d": 0.45,
        "target_amount_ratio_20d": 0.95,
        "max_amount_ratio_20d": 2.50,
        "min_circ_mv": 300000.0,
        "strong_circ_mv": 900000.0,
        "max_distance_below_ma20_pct": 5.5,
        "max_distance_below_ma60_pct": 10.0,
        "min_ma20_slope_5d_pct": -2.0,
        "min_rebound_from_low_pct": 0.8,
        "strong_rebound_from_low_pct": 1.6,
        "min_close_vs_open_pct": -2.2,
        "limit_down_buffer_pct": 0.6,
        "unlock_lookahead_days": 30,
        "max_unlock_ratio_30d": 5.0,
        "holdertrade_lookback_days": 60,
        "max_holder_decrease_ratio_60d": 0.8,
        "express_lookback_days": 45,
        "express_negative_yoy_threshold": -20.0,
        "tail_buy_start_time": "14:30",
        "entry_discount_pct": 0.2,
        "entry_premium_pct": 0.2,
        "stop_loss_pct": 2.6,
        "backtest_take_profit_pct": 2.0,
        "backtest_intraday_stop_pct": -3.0,
        "backtest_gap_stop_pct": -4.0,
        "realtime_ts_code": "0*.SZ,3*.SZ,6*.SH",
        "realtime_fields": "ts_code,name,pre_close,high,open,low,close,vol,amount,trade_time",
    },
}


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "scoring_weights.yaml"
    if not config_path.exists():
        return config
    try:
        import yaml

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return config
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = dict(config[key])
            merged.update(value)
            config[key] = merged
        else:
            config[key] = value
    return config
