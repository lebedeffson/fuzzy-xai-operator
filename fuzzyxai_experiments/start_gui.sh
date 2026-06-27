#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
CLEAN_LINK=0
if [ ! -e fuzzyxai_experiments ]; then
  ln -s . fuzzyxai_experiments
  CLEAN_LINK=1
fi
cleanup() { if [ "$CLEAN_LINK" = "1" ]; then rm -f fuzzyxai_experiments; fi; }
trap cleanup EXIT
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/..:${PYTHONPATH:-}"
python gui_app.py "${PORT:-8501}"
