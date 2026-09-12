#!/usr/bin/env bash
# Thin wrapper around run_all.py, which is the real entry point and works on
# every platform. Kept because a shell one-liner is what most people reach for.
#
#   ./run_all.sh
#   ./run_all.sh --dataset split-cifar10 --epochs 5 --seeds 42 43 44
#
# Windows users: skip this and run `python run_all.py` directly.
set -euo pipefail
PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if command -v python >/dev/null 2>&1; then PY=python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else echo "no python interpreter found on PATH" >&2; exit 1; fi
fi
cd "$(dirname "$0")"
exec "$PY" run_all.py "$@"
