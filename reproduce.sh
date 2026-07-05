#!/usr/bin/env bash
# Regenerate the KV-precision frontier. Usage: ./reproduce.sh /path/to/model.gguf [bin]
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:?usage: reproduce.sh MODEL.gguf [llama-perplexity path]}"
BIN="${2:-/opt/homebrew/bin/llama-perplexity}"
. .venv/bin/activate
# fixed text sample (first 150 GSM8K questions if present, else supply data/ppl_text.txt)
[ -f data/ppl_text.txt ] || { echo "provide data/ppl_text.txt (a fixed English text)"; exit 1; }
python tools/run_sweep.py --model "$MODEL" --bin "$BIN" --ctxs "512,2048" --out results/kvbits.jsonl
python tools/analyze.py --in results/kvbits.jsonl --out bench_results/frontier.md
echo "regenerated bench_results/frontier.md"
