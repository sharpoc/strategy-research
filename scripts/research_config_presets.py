from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_backtest_utils import repo_root_dir


RESEARCH_CONFIG_PRESETS: dict[str, dict[str, str]] = {
    "research": {
        "strategy_config_file": "configs/price_strategy_research_filters.json",
        "description": "价格型策略统一研究过滤 + 龙门双阶 research gate。",
    },
    "research_limitup_nogate": {
        "strategy_config_file": "configs/price_strategy_research_filters_limitup_nogate.json",
        "description": "价格型策略研究过滤，龙门双阶不启用 research gate。",
    },
    "research_limitup_selective": {
        "strategy_config_file": "configs/price_strategy_research_filters_limitup_selective.json",
        "description": "价格型策略研究过滤，龙门双阶启用更严格的 selective gate。",
    },
    "research_all_price": {
        "strategy_config_file": "configs/price_strategy_research_all.json",
        "description": "四条价格型策略的一体化研究配置，含真实资金突破保守版。",
    },
    "real_breakout_baseline": {
        "strategy_config_file": "configs/real_breakout_research_baseline.json",
        "description": "真实资金突破基线研究配置。",
    },
    "real_breakout_tuned": {
        "strategy_config_file": "configs/real_breakout_research_tuned.json",
        "description": "真实资金突破收紧版研究配置。",
    },
    "real_breakout_downtrend": {
        "strategy_config_file": "configs/real_breakout_research_downtrend.json",
        "description": "真实资金突破仅下跌趋势出手的研究配置。",
    },
    "real_breakout_downtrend_selective": {
        "strategy_config_file": "configs/real_breakout_research_downtrend_selective.json",
        "description": "真实资金突破仅下跌趋势出手的 selective 研究配置。",
    },
}

EXIT_CONFIG_PRESETS: dict[str, dict[str, str]] = {
    "holder_increase_research_best": {
        "exit_config_file": "configs/exit_holder_increase_research_best.json",
        "description": "星曜增持：2026-03-17 基于 7 个 replay 快照的最佳研究 exit 配置。",
    },
    "limitup_l1l2_research_best": {
        "exit_config_file": "configs/exit_limitup_l1l2_research_best.json",
        "description": "龙门双阶：2026-03-16 本地 exit 优化最佳研究配置。",
    },
    "price_selective_best": {
        "exit_config_file": "configs/exit_price_selective_best.json",
        "description": "龙门双阶 + 真实资金突破：2026-03-16 selective 最佳 exit 组合。",
    },
    "real_breakout_research_best": {
        "exit_config_file": "configs/exit_real_breakout_research_best.json",
        "description": "真实资金突破：2026-03-16 本地 exit 优化最佳研究配置。",
    },
    "real_breakout_target_first": {
        "exit_config_file": "configs/exit_real_breakout_target_first.json",
        "description": "真实资金突破：盘中止盈/止损冲突时优先按目标价成交。",
    },
}


def available_strategy_config_presets() -> list[str]:
    return sorted(RESEARCH_CONFIG_PRESETS.keys())


def strategy_config_preset_help() -> str:
    return ", ".join(
        f"{name}={meta['description']}" for name, meta in sorted(RESEARCH_CONFIG_PRESETS.items(), key=lambda item: item[0])
    )


def available_exit_config_presets() -> list[str]:
    return sorted(EXIT_CONFIG_PRESETS.keys())


def exit_config_preset_help() -> str:
    return ", ".join(
        f"{name}={meta['description']}" for name, meta in sorted(EXIT_CONFIG_PRESETS.items(), key=lambda item: item[0])
    )


def load_json_file(path_str: str) -> dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a JSON object: {path}")
    return data


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_strategy_overrides_with_preset(
    preset_name: str = "",
    config_file: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    preset_name = (preset_name or "").strip()
    config_file = (config_file or "").strip()
    metadata: dict[str, Any] = {
        "strategy_config_preset": preset_name or None,
        "strategy_config_file": None,
        "preset_strategy_config_file": None,
    }

    merged: dict[str, Any] = {}
    if preset_name:
        if preset_name not in RESEARCH_CONFIG_PRESETS:
            raise SystemExit(
                f"Unknown strategy config preset: {preset_name}. "
                f"Available: {', '.join(available_strategy_config_presets())}"
            )
        preset_meta = RESEARCH_CONFIG_PRESETS[preset_name]
        preset_path = (repo_root_dir() / preset_meta["strategy_config_file"]).resolve()
        merged = load_json_file(str(preset_path))
        metadata["preset_strategy_config_file"] = str(preset_path)

    if config_file:
        file_overrides = load_json_file(config_file)
        merged = deep_merge_dict(merged, file_overrides)
        metadata["strategy_config_file"] = str(Path(config_file).expanduser().resolve())

    return merged, metadata


def load_exit_config_with_preset(
    preset_name: str = "",
    config_file: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    preset_name = (preset_name or "").strip()
    config_file = (config_file or "").strip()
    metadata: dict[str, Any] = {
        "exit_config_preset": preset_name or None,
        "exit_config_file": None,
        "preset_exit_config_file": None,
    }

    merged: dict[str, Any] = {}
    if preset_name:
        if preset_name not in EXIT_CONFIG_PRESETS:
            raise SystemExit(
                f"Unknown exit config preset: {preset_name}. "
                f"Available: {', '.join(available_exit_config_presets())}"
            )
        preset_meta = EXIT_CONFIG_PRESETS[preset_name]
        preset_path = (repo_root_dir() / preset_meta["exit_config_file"]).resolve()
        merged = load_json_file(str(preset_path))
        metadata["preset_exit_config_file"] = str(preset_path)

    if config_file:
        file_overrides = load_json_file(config_file)
        merged = deep_merge_dict(merged, file_overrides)
        metadata["exit_config_file"] = str(Path(config_file).expanduser().resolve())

    return merged, metadata
