from __future__ import annotations

import json

from backtest_strategy_registry import build_price_strategy_registry


def main() -> None:
    registry = build_price_strategy_registry()
    rows = [
        {
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.strategy_name,
            "history_bars": spec.history_bars,
            "recommended_history_years": spec.recommended_history_years,
            "required_dataset_ids": list(spec.required_dataset_ids),
        }
        for spec in registry.values()
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
