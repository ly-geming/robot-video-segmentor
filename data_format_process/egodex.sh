#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <task_count> <num_per_task> [egodex_root] [output_data_root]"
  echo "Example:"
  echo "  $0 10 100 /data/LFT-W02_data/shuaizhou/human_video_data/EgoDex /data/LFT-W02_data/shuaizhou/human_video_data/video2tasks/data"
  exit 1
fi

TASK_COUNT="$1"
NUM_PER_TASK="$2"
EGODEX_ROOT="${3:-/data/LFT-W02_data/shuaizhou/human_video_data/EgoDex}"
OUTPUT_DATA_ROOT="${4:-/data/LFT-W02_data/shuaizhou/human_video_data/video2tasks/data}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/egodex_preprocess.py" \
  --egodex-root "${EGODEX_ROOT}" \
  --output-data-root "${OUTPUT_DATA_ROOT}" \
  --task "${TASK_COUNT}" \
  --num "${NUM_PER_TASK}"
