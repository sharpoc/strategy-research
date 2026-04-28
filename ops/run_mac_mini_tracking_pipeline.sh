#!/bin/zsh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env.mac_mini}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

LOG_DIR="${TRACKING_PIPELINE_LOG_DIR:-$REPO_ROOT/ops/logs}"
case "$LOG_DIR" in
  /*) ;;
  *) LOG_DIR="$REPO_ROOT/$LOG_DIR" ;;
esac
mkdir -p "$LOG_DIR"

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/tracking_pipeline_${STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%F %T')] tracking pipeline start"
python3 "$REPO_ROOT/ops/run_mac_mini_tracking_pipeline.py"
echo "[$(date '+%F %T')] tracking pipeline done"
