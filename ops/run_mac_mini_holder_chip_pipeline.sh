#!/bin/zsh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export HOLDER_CONFIG_FILE_OVERRIDE="${HOLDER_CONFIG_FILE_OVERRIDE:-configs/holder_chip_enhanced_screening.json}"
export HOLDER_EXPORT_PREFIX_OVERRIDE="${HOLDER_EXPORT_PREFIX_OVERRIDE:-holder_chip_enhanced_screen_}"
export APP_INTERNAL_API_STRATEGY_ID_OVERRIDE="${APP_INTERNAL_API_STRATEGY_ID_OVERRIDE:-holder_chip_enhanced_screening}"

exec "$REPO_ROOT/ops/run_mac_mini_holder_pipeline.sh"
