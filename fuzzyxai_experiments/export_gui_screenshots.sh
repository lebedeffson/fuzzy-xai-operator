#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
python scripts/build_gui_screenshots.py
printf 'PASS: gui_screenshots\n'
