#!/usr/bin/env bash
set -euo pipefail

echo "== Python =="
python -V || true
which python || true

echo "\n== Pip (same interpreter) =="
python -m pip -V || true

echo "\n== Top packages =="
python -m pip list 2>/dev/null | head -n 30 || true

echo "\n== Pytest =="
python -m pytest --version || true

echo "\n== Platform =="
uname -a || true
