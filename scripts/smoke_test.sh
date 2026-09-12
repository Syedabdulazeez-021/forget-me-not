#!/usr/bin/env bash
# Thin wrapper around scripts/smoke_test.py (the real, cross-platform version).
# Windows users: run `python scripts/smoke_test.py` directly.
set -euo pipefail
PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if command -v python >/dev/null 2>&1; then PY=python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else echo "no python interpreter found on PATH" >&2; exit 1; fi
fi
cd "$(dirname "$0")/.."
exec "$PY" scripts/smoke_test.py "$@"
