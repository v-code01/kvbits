# kvbits

The KV cache dominates memory at long context, and llama.cpp can store it at f16, q8_0,
q5_1, or q4_0. This measures the exact quality cost of each precision, separately for the
key cache and the value cache, and finds where the bits should actually go.

Perplexity is measured by llama.cpp's own `llama-perplexity` binary, exact and
deterministic. No model judges anything.

## The finding (the key cache is sensitive, the value cache is not)

Qwen2.5-1.5B-Instruct Q4_K_M, flash attention on, fixed English text. Full numbers in
[`bench_results/frontier.md`](bench_results/frontier.md). The K-by-V perplexity matrix at
context 512 (rows = key cache precision, columns = value cache precision):

| K \ V | f16 | q8_0 | q5_1 | q4_0 |
|---|---|---|---|---|
| **f16**  | 7.15 | 7.15 | 7.14 | 7.14 |
| **q8_0** | 7.16 | 7.14 | 7.14 | 7.15 |
| **q5_1** | 148  | 152  | 151  | 127  |
| **q4_0** | 1519 | 1505 | 1623 | 1573 |

Read it in two directions. Across any row (fix the key cache, drop the value cache to
4-bit) perplexity is flat: the value cache is nearly free to quantize all the way to q4_0.
Down any column (fix the value cache, drop the key cache) perplexity explodes: q5_1 keys
(6-bit) already blow it up to ~150, and q4_0 keys destroy the model entirely (>1500).

So the bits belong in the key cache, not split evenly:

- **q8_0 on both is lossless** (7.14 vs 7.15 f16), a 47% KV-cache reduction.
- **Asymmetric Kq8_0/Vq4_0 is the Pareto win: a 59% KV-cache reduction at under 1%
  perplexity change** (7.15). The value cache pays 4-bit, the key cache keeps 8.
- **The naive symmetric q4_0 is catastrophic** (1573 perplexity, ~220x worse). Anyone who
  quantizes both caches to 4-bit to save memory has silently broken their model.
- **The key sensitivity worsens with context.** At context 2048 the q5_1 key rows are even
  worse (~1000 to 1100 perplexity), so longer context makes the key cache MORE demanding of
  precision, not less.

The mechanism is the one the rate-distortion KV literature (RDKV, KIVI) predicts: key
vectors carry the attention-score geometry and have outlier channels that low-bit
quantization wrecks, while value vectors are averaged by the attention weights and tolerate
coarse quantization. This makes that asymmetry exact and visible on a real engine.

## What would change or falsify this

- **A different model or attention shape.** Qwen2.5-1.5B uses grouped-query attention with
  2 key/value heads. A model with more KV heads, different head_dim, or per-channel key
  quantization could move the key cliff. The direction (key more sensitive than value) is
  robust in the literature, but the exact cliff location is model-specific.
- **Better key quantization.** The cliff here is for plain ggml q5_1/q4_0. A key-aware
  scheme (per-channel scales, rotation, as in the RDKV/TurboQuant line) could push usable
  keys below 8 bits. If it cannot, 8-bit keys are the practical floor for this engine.

## Limitations (named, not hidden)

- One model and size (Qwen2.5-1.5B-Instruct Q4_K_M), one text, contexts 512 and 2048, the
  four ggml cache types. Absolute perplexities are model-specific; the asymmetry and the
  Pareto ranking are the transferable results.
- **Perplexity is the metric.** It is the exact intrinsic quality measure and a model at
  perplexity 1500 is fully broken, so the ranking predicts task accuracy; but task-level
  accuracy (for example GSM8K exact-match) under each cache type is not separately measured
  here and is the natural next step.
- Flash attention is required for a quantized value cache and is on for every run, so it is
  held constant and is not a confound in the comparison.

## How it is measured

1. **Sweep (`tools/run_sweep.py`).** For each (key type, value type, context) it runs
   `llama-perplexity` with that cache precision and `-fa on`, parses the final estimate, and
   records it with the exact KV bytes per token.
2. **Byte accounting (`src/kvbits.py`, `src/ggufmeta.py`).** KV bytes per token come from
   the model geometry read out of the GGUF header (n_layer, n_head_kv, head_dim) times the
   ggml block bit rate of each cache type, for key and value separately.
3. **Frontier (`tools/analyze.py`).** Builds the K-by-V matrix, the quality-vs-bytes Pareto
   frontier, and the recommended allocation per context.

## Build and test

```
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m pytest -q          # 9 tests
./scripts/gate.sh            # ruff + mypy --strict + pytest + ASCII
```

## Reproduce the frontier

Needs `llama-perplexity` (llama.cpp) and a Qwen2.5-1.5B-Instruct GGUF:

```
./reproduce.sh /path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Numbers here are on an Apple M4 Pro. kvbits runs the perplexity binary directly and needs no server.

## License

MIT. See [`LICENSE`](LICENSE).
