#!/usr/bin/env bash
set -euo pipefail

DEFAULT_COMMAND=(python -m examples.quickstart)
export ATLAS_QUICKSTART_CONFIG="${ATLAS_QUICKSTART_CONFIG:-docker/configs/atlas.docker.yaml}"

if [ "$#" -gt 0 ]; then
  exec "$@"
else
  exec "${DEFAULT_COMMAND[@]}"
fi
