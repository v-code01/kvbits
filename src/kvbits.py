"""kvbits: exact accuracy-vs-KV-cache-precision measurement helpers.

The heavy lifting (perplexity) is done by llama.cpp's own llama-perplexity binary; this
module is the deterministic, testable glue: parse its output, compute exact KV-cache bytes
per token for a given precision, and reduce a set of (config, bytes, perplexity) points to
the Pareto frontier. No model is called from here.
"""
from __future__ import annotations

import re
from typing import Optional

# ggml block bit rates (bits per stored element), from the quant block layouts:
#   f16 = 16; q8_0 = (32*8 + 16)/32 = 8.5; q5_1 = (32*5 + 16 + 16)/32 = 6;
#   q4_0 = (32*4 + 16)/32 = 4.5; q4_1 = (32*4 + 16 + 16)/32 = 5.
BITS_PER_ELEM: dict[str, float] = {
    "f16": 16.0,
    "q8_0": 8.5,
    "q5_1": 6.0,
    "q4_1": 5.0,
    "q4_0": 4.5,
}

_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def parse_ppl(output: str) -> tuple[Optional[float], Optional[float]]:
    """Extract (ppl, stderr) from llama-perplexity output; (None, None) if absent."""
    m = _PPL_RE.search(output)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def kv_bytes_per_token(
    n_layer: int, n_head_kv: int, head_dim: int, kbits: float, vbits: float
) -> float:
    """Exact KV-cache bytes stored per token: the key and value caches each hold
    n_layer * n_head_kv * head_dim elements per token, at their own bit rate.
    """
    elems = n_layer * n_head_kv * head_dim
    return elems * kbits / 8.0 + elems * vbits / 8.0


def pareto_frontier(
    points: list[tuple[str, float, float]]
) -> list[tuple[str, float, float]]:
    """Given (label, bytes, ppl) points, return the Pareto-optimal set for minimizing both
    bytes and ppl, sorted by bytes ascending. A point is on the frontier if no other point
    has bytes <= and ppl <= with at least one strictly less.
    """
    front = []
    for p in points:
        _, pb, pp = p
        dominated = False
        for q in points:
            if q is p:
                continue
            _, qb, qp = q
            if qb <= pb and qp <= pp and (qb < pb or qp < pp):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda x: x[1])
    return front
