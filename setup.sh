#!/usr/bin/env bash
# Usage: ./setup.sh [extras]
#   extras  comma-separated optional extras to install (default: audio,plot,test)
#           e.g. ./setup.sh audio    or    ./setup.sh ""  for the core install only
# Set PYTHON to choose the interpreter, e.g. PYTHON=python3.12 ./setup.sh
set -euo pipefail

cd "$(dirname "$0")"
EXTRAS="${1-audio,plot,test}"
PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "error: Python 3.10+ is required (using: $PYTHON)" >&2
    exit 1
fi

if [ ! -d .venv ]; then
    echo "Creating virtual environment in .venv ..."
    "$PYTHON" -m venv .venv
else
    echo "Reusing existing virtual environment in .venv"
fi

.venv/bin/python -m pip install -U pip
if [ -n "$EXTRAS" ]; then
    .venv/bin/python -m pip install -e ".[$EXTRAS]"
else
    .venv/bin/python -m pip install -e .
fi

echo
echo "Done. Activate the environment with:"
echo
echo "    source .venv/bin/activate"
