#!/usr/bin/env bash
set -euo pipefail

DEFAULT_COMMAND=(
  python -m examples.arc_agi_demo.scripts.run_demo
  --config /app/docker/configs/atlas.docker.yaml
  --task-file examples/arc_agi_demo/data/arc_training_0ca9ddb6.json
  --stream-progress
)

if [ "$#" -gt 0 ]; then
  exec "$@"
else
  exec "${DEFAULT_COMMAND[@]}"
fi
