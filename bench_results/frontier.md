# kvbits: the KV-cache precision frontier and the key-vs-value asymmetry

Qwen2.5-1.5B-Instruct Q4_K_M, llama.cpp llama-perplexity, flash attention on, fixed English text. Perplexity is exact and deterministic. Lower is better. KV bytes/token are computed exactly from the model geometry and the ggml block bit rates (f16=16, q8_0=8.5, q5_1=6, q4_0=4.5 bits/elem).

## Context 512: K-by-V perplexity matrix (rows = key cache, cols = value cache)

| K \ V | f16 | q8_0 | q5_1 | q4_0 |
|---|---|---|---|---|
| **f16** | 7.15 | 7.15 | 7.14 | 7.14 |
| **q8_0** | 7.16 | 7.14 | 7.14 | 7.15 |
| **q5_1** | 148 | 152 | 151 | 127 |
| **q4_0** | 1519 | 1505 | 1623 | 1573 |

### Pareto frontier (quality vs KV bytes/token), context 512

| config | KV bytes/token | perplexity | vs f16 |
|:-------|---------------:|-----------:|-------:|
| Kq4_0/Vq4_0 | 8064 | 1573 | +21907.4% |
| Kq5_1/Vq4_0 | 9408 | 127 | +1677.2% |
| Kq8_0/Vq4_0 | 11648 | 7.15 | +0.0% |
| Kq8_0/Vq5_1 | 12992 | 7.14 | -0.1% |
| Kf16/Vq4_0 | 18368 | 7.14 | -0.1% |

**Recommended at ctx 512:** `Kq8_0/Vq4_0` at 11648 bytes/token, perplexity 7.15 (within 1% of f16), a 59% KV-cache reduction.

## Context 2048: K-by-V perplexity matrix (rows = key cache, cols = value cache)

| K \ V | f16 | q8_0 | q5_1 | q4_0 |
|---|---|---|---|---|
| **f16** | 6.37 | 6.36 | 6.36 | 6.36 |
| **q8_0** | 6.37 | 6.36 | 6.37 | 6.36 |
| **q5_1** | 1039 | 1043 | 1017 | 1107 |
| **q4_0** | 2279 | 2553 | 2474 | 2168 |

### Pareto frontier (quality vs KV bytes/token), context 2048

| config | KV bytes/token | perplexity | vs f16 |
|:-------|---------------:|-----------:|-------:|
| Kq4_0/Vq4_0 | 8064 | 2168 | +33957.4% |
| Kq5_1/Vq4_0 | 9408 | 1107 | +17281.7% |
| Kq5_1/Vq5_1 | 10752 | 1017 | +15878.2% |
| Kq8_0/Vq4_0 | 11648 | 6.36 | -0.0% |
| Kq8_0/Vq8_0 | 15232 | 6.36 | -0.1% |
| Kf16/Vq4_0 | 18368 | 6.36 | -0.1% |

**Recommended at ctx 2048:** `Kq8_0/Vq4_0` at 11648 bytes/token, perplexity 6.36 (within 1% of f16), a 59% KV-cache reduction.

## What the matrix shows

Read down a column (fix the value cache, vary the key cache) versus across a row (fix the key cache, vary the value cache). If the key cache is the sensitive one, perplexity blows up as you move down the rows to q4_0 but stays flat across the value columns. That asymmetry is the whole point: bits spent on the key cache buy far more quality than bits spent on the value cache, so a symmetric budget is wrong and an asymmetric key-high / value-low allocation is the Pareto win.
