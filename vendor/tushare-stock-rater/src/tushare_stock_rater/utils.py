from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_project_env(project_root: Path | None = None) -> None:
    root = project_root or PROJECT_ROOT
    load_env_file(root / ".env")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def to_number(value: Any, digits: int = 2) -> Any:
    number = to_float(value)
    if number is None:
        return None
    return round(number, digits)


def to_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是"}
    return bool(value)


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_range(value: Any, good: float, weak: float, max_score: float, reverse: bool = False) -> float:
    number = to_float(value)
    if number is None:
        return max_score * 0.45
    if reverse:
        if number <= good:
            return max_score
        if number >= weak:
            return 0.0
        return max_score * (weak - number) / (weak - good)
    if number >= good:
        return max_score
    if number <= weak:
        return 0.0
    return max_score * (number - weak) / (good - weak)


def first_present(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row:
            value = row.get(name)
            if to_float(value) is not None or (isinstance(value, str) and value.strip()):
                return value
    return None


def latest_record(df: pd.DataFrame, date_columns: tuple[str, ...] = ("ann_date", "end_date", "trade_date")) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    work = df.copy()
    sort_cols = [col for col in date_columns if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    row = work.iloc[0]
    return {str(k): (None if pd.isna(v) else v) for k, v in row.items()}


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def compact_join(items: list[str], limit: int = 5) -> str:
    filtered = [item for item in items if item]
    if not filtered:
        return "无"
    if len(filtered) <= limit:
        return "；".join(filtered)
    return "；".join(filtered[:limit]) + f"；另有 {len(filtered) - limit} 项"
