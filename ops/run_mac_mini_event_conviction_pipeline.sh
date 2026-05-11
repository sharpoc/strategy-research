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

LOG_DIR="${EVENT_CONVICTION_PIPELINE_LOG_DIR:-${PIPELINE_LOG_DIR:-$REPO_ROOT/ops/logs}}"
case "$LOG_DIR" in
  /*) ;;
  *) LOG_DIR="$REPO_ROOT/$LOG_DIR" ;;
esac
mkdir -p "$LOG_DIR"

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/event_conviction_pipeline_${STAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

MIN_START_HHMM="${EVENT_CONVICTION_MIN_START_HHMM:-2000}"
RETRY_ENABLED="${EVENT_CONVICTION_RETRY_ENABLED:-true}"
MAX_ATTEMPTS="${EVENT_CONVICTION_MAX_ATTEMPTS:-8}"
RETRY_SLEEP_SEC="${EVENT_CONVICTION_RETRY_SLEEP_SEC:-600}"
RETRY_DEADLINE_HHMM="${EVENT_CONVICTION_RETRY_DEADLINE_HHMM:-2355}"

current_hhmm() {
  TZ=Asia/Shanghai date '+%H%M'
}

sleep_until_min_start() {
  local now_hhmm
  now_hhmm="$(current_hhmm)"
  if [[ "$now_hhmm" -ge "$MIN_START_HHMM" ]]; then
    return 0
  fi

  local today min_hour min_minute target_epoch now_epoch wait_sec
  today="$(TZ=Asia/Shanghai date '+%Y-%m-%d')"
  min_hour="${MIN_START_HHMM:0:2}"
  min_minute="${MIN_START_HHMM:2:2}"
  target_epoch="$(TZ=Asia/Shanghai date -j -f '%Y-%m-%d %H:%M:%S' "${today} ${min_hour}:${min_minute}:00" '+%s')"
  now_epoch="$(TZ=Asia/Shanghai date '+%s')"
  wait_sec="$(( target_epoch - now_epoch ))"
  if [[ "$wait_sec" -gt 0 ]]; then
    echo "[$(date '+%F %T')] before ${MIN_START_HHMM}, sleep ${wait_sec}s until data window opens"
    sleep "$wait_sec"
  fi
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
  [[ "$log_tail" == *"Temporary failure"* ]] && return 0
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

  sleep_until_min_start

  while true; do
    echo "[$(date '+%F %T')] event conviction run attempt ${attempt}/${MAX_ATTEMPTS}"
    set +e
    "$PYTHON_BIN" "${RUN_ARGS[@]}"
    exit_code="$?"
    set -e
    if [[ "$exit_code" -eq 0 ]]; then
      echo "[$(date '+%F %T')] event conviction run success on attempt ${attempt}"
      return 0
    fi

    tail_text="$(tail -n 80 "$LOG_FILE" 2>/dev/null || true)"
    echo "[$(date '+%F %T')] event conviction run failed on attempt ${attempt}, exit_code=${exit_code}"

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

echo "[$(date '+%F %T')] event conviction pipeline start"
echo "repo_root=$REPO_ROOT"
echo "env_file=$ENV_FILE"

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRADE_DATE="${TRADE_DATE:-$(TZ=Asia/Shanghai date '+%Y%m%d')}"
CONFIG_FILE="${EVENT_CONVICTION_CONFIG_FILE:-configs/event_conviction_quality_first_v2.json}"
EXPORT_ROOT="${EVENT_CONVICTION_EXPORT_ROOT:-output/jupyter-notebook/event_conviction_exports}"
API_SLEEP_SEC="${EVENT_CONVICTION_API_SLEEP_SEC:-0.05}"
SHOW_TOP="${EVENT_CONVICTION_SHOW_TOP:-10}"

case "$CONFIG_FILE" in
  /*) ;;
  *) CONFIG_FILE="$REPO_ROOT/$CONFIG_FILE" ;;
esac
case "$EXPORT_ROOT" in
  /*) ;;
  *) EXPORT_ROOT="$REPO_ROOT/$EXPORT_ROOT" ;;
esac
mkdir -p "$EXPORT_ROOT"

RUN_ARGS=(
  "scripts/run_tushare_event_conviction_strategy.py"
  "--end-date" "$TRADE_DATE"
  "--config-file" "$CONFIG_FILE"
  "--export-root" "$EXPORT_ROOT"
  "--api-sleep-sec" "$API_SLEEP_SEC"
  "--show-top" "$SHOW_TOP"
)

run_with_retry

if [[ -n "${LAB_EVENT_SYNC_SCRIPT:-}" ]]; then
  SYNC_SCRIPT="$LAB_EVENT_SYNC_SCRIPT"
  case "$SYNC_SCRIPT" in
    /*) ;;
    *) SYNC_SCRIPT="$REPO_ROOT/$SYNC_SCRIPT" ;;
  esac

  if [[ ! -x "$SYNC_SCRIPT" ]]; then
    echo "LAB_EVENT_SYNC_SCRIPT is set but not executable: $SYNC_SCRIPT"
    exit 1
  fi

  echo "[$(date '+%F %T')] event sync hook start: $SYNC_SCRIPT"
  SYNC_ENV_FILE="${LAB_EVENT_SYNC_ENV_FILE:-${LAB_SYNC_ENV_FILE:-}}"
  if [[ -n "$SYNC_ENV_FILE" ]]; then
    case "$SYNC_ENV_FILE" in
      /*) ;;
      *) SYNC_ENV_FILE="$REPO_ROOT/$SYNC_ENV_FILE" ;;
    esac
    if [[ ! -f "$SYNC_ENV_FILE" ]]; then
      echo "LAB_EVENT_SYNC_ENV_FILE is set but missing: $SYNC_ENV_FILE"
      exit 1
    fi
    ENV_FILE="$SYNC_ENV_FILE" TRADE_DATE="$TRADE_DATE" "$SYNC_SCRIPT"
  else
    unset ENV_FILE
    TRADE_DATE="$TRADE_DATE" "$SYNC_SCRIPT"
  fi
  echo "[$(date '+%F %T')] event sync hook done"
else
  echo "LAB_EVENT_SYNC_SCRIPT not set, skip online sync."
fi

echo "[$(date '+%F %T')] event conviction pipeline done"
