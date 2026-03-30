#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_PATH="$REPO_DIR/output/jupyter-notebook/tushare-holder-increase-screening.ipynb"
PYTHON_BIN="/Applications/Xcode.app/Contents/Developer/usr/bin/python3"

export PATH="$HOME/Library/Python/3.9/bin:$PATH"

if [[ ! -f "$NOTEBOOK_PATH" ]]; then
  echo "Notebook not found: $NOTEBOOK_PATH" >&2
  exit 1
fi

if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
  printf "Enter Tushare Token: "
  read -r TUSHARE_TOKEN
  export TUSHARE_TOKEN
fi

exec arch -arm64 "$PYTHON_BIN" -m jupyterlab "$NOTEBOOK_PATH"
