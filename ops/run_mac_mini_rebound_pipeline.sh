#!/bin/zsh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env.mac_mini}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE"
  echo "Copy .env.mac_mini.example to .env.mac_mini and fill it first."
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
  echo "TUSHARE_TOKEN is required."
  exit 1
fi

LOG_DIR="${REBOUND_PIPELINE_LOG_DIR:-${PIPELINE_LOG_DIR:-$REPO_ROOT/ops/logs}}"
case "$LOG_DIR" in
  /*) ;;
  *) LOG_DIR="$REPO_ROOT/$LOG_DIR" ;;
esac
mkdir -p "$LOG_DIR"

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/rebound_pipeline_${STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRADE_DATE="${TRADE_DATE:-$(TZ=Asia/Shanghai date '+%Y%m%d')}"
RETRY_ENABLED="${REBOUND_RETRY_ENABLED:-true}"
MAX_ATTEMPTS="${REBOUND_MAX_ATTEMPTS:-4}"
RETRY_SLEEP_SEC="${REBOUND_RETRY_SLEEP_SEC:-300}"
RETRY_DEADLINE_HHMM="${REBOUND_RETRY_DEADLINE_HHMM:-1515}"

current_hhmm() {
  TZ=Asia/Shanghai date '+%H%M'
}

is_transient_failure() {
  local log_tail="$1"
  [[ "$log_tail" == *"当前接口达到请求上限"* ]] && return 0
  [[ "$log_tail" == *"请稍后重试"* ]] && return 0
  [[ "$log_tail" == *"ReadTimeout"* ]] && return 0
  [[ "$log_tail" == *"TimeoutError"* ]] && return 0
  [[ "$log_tail" == *"timed out"* ]] && return 0
  [[ "$log_tail" == *"Connection reset"* ]] && return 0
  [[ "$log_tail" == *"RemoteDisconnected"* ]] && return 0
  [[ "$log_tail" == *"HTTP 502"* ]] && return 0
  [[ "$log_tail" == *"HTTP 503"* ]] && return 0
  [[ "$log_tail" == *"HTTP 504"* ]] && return 0
  return 1
}

deadline_reached() {
  local now_hhmm
  now_hhmm="$(current_hhmm)"
  [[ "$now_hhmm" -ge "$RETRY_DEADLINE_HHMM" ]]
}

run_with_retry() {
  local attempt=1
  local exit_code=0
  local tail_text=""
  local run_args=(
    "$REPO_ROOT/ops/run_mac_mini_rebound_pipeline.py"
    "--trade-date" "$TRADE_DATE"
  )

  while true; do
    echo "[$(date '+%F %T')] rebound run attempt ${attempt}/${MAX_ATTEMPTS}"
    set +e
    "$PYTHON_BIN" "${run_args[@]}"
    exit_code="$?"
    set -e
    if [[ "$exit_code" -eq 0 ]]; then
      echo "[$(date '+%F %T')] rebound run success on attempt ${attempt}"
      return 0
    fi

    tail_text="$(tail -n 80 "$LOG_FILE" 2>/dev/null || true)"
    echo "[$(date '+%F %T')] rebound run failed on attempt ${attempt}, exit_code=${exit_code}"

    if [[ "$RETRY_ENABLED" != "true" ]]; then
      return "$exit_code"
    fi
    if ! is_transient_failure "$tail_text"; then
      echo "[$(date '+%F %T')] failure looks non-transient, stop retry"
      return "$exit_code"
    fi
    if [[ "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
      echo "[$(date '+%F %T')] reached max attempts, stop retry"
      return "$exit_code"
    fi
    if deadline_reached; then
      echo "[$(date '+%F %T')] reached retry deadline ${RETRY_DEADLINE_HHMM}, stop retry"
      return "$exit_code"
    fi

    echo "[$(date '+%F %T')] transient failure detected, sleep ${RETRY_SLEEP_SEC}s then retry"
    sleep "$RETRY_SLEEP_SEC"
    attempt="$(( attempt + 1 ))"
  done
}

echo "[$(date '+%F %T')] rebound pipeline start"
echo "repo_root=$REPO_ROOT"
echo "env_file=$ENV_FILE"
echo "trade_date=$TRADE_DATE"

cd "$REPO_ROOT"
run_with_retry

echo "[$(date '+%F %T')] rebound pipeline done"
