#!/usr/bin/env bash
set -euo pipefail

DEFAULT_COMMAND=(python -m examples.quickstart)

if [ "$#" -gt 0 ]; then
  exec "$@"
else
  exec "${DEFAULT_COMMAND[@]}"
fi
