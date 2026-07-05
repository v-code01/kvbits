#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "GATE FAIL: $*" >&2; exit 1; }
[ -d .venv ] && . .venv/bin/activate || { python3 -m venv .venv; . .venv/bin/activate; pip install -q -r requirements.txt; }
echo "== 1/4 ruff =="; ruff check src tests tools || fail ruff; echo "   ok"
echo "== 2/4 mypy --strict =="; MYPYPATH=src mypy --strict src tools/run_sweep.py tools/analyze.py || fail mypy; echo "   ok"
echo "== 3/4 pytest =="; python -m pytest -q || fail pytest
echo "== 4/4 pure-ASCII =="; bad=$(LC_ALL=C grep -rlP '[^\x00-\x7F]' src tests tools scripts README.md claims.toml bench_results 2>/dev/null || true); [ -z "$bad" ] || { echo "$bad"; fail ascii; }; echo "   ok"
echo "ALL GATES PASS"
