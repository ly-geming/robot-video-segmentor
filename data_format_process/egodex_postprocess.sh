#!/usr/bin/env bash
set -euo pipefail
# Merge data/<subset>/<id>/ + runs/<subset>/<run_id>/samples/<id>/ -> ../processed_Data/<subset>/<id>/
# Example:  ./egodex_postprocess.sh
#           ./egodex_postprocess.sh --run-id egodex_10x100_qwen35 --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/egodex_postprocess.py" "$@"
