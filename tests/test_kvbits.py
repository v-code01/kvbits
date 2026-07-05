import math

from kvbits import (
    BITS_PER_ELEM,
    parse_ppl,
    kv_bytes_per_token,
    pareto_frontier,
)


# --- parse llama-perplexity output -----------------------------------------

def test_parse_ppl_final_estimate():
    out = "chunk 1 ...\n0.01.379 I perplexity: 1.02 seconds per pass\nFinal estimate: PPL = 7.1457 +/- 0.27733\n"
    ppl, err = parse_ppl(out)
    assert abs(ppl - 7.1457) < 1e-9
    assert abs(err - 0.27733) < 1e-9

def test_parse_ppl_catastrophic():
    ppl, err = parse_ppl("Final estimate: PPL = 1572.5827 +/- 89.46656")
    assert abs(ppl - 1572.5827) < 1e-6

def test_parse_ppl_missing_returns_none():
    assert parse_ppl("no estimate here, failed to create context") == (None, None)


# --- exact KV byte accounting ----------------------------------------------

def test_bits_per_elem_known():
    # ggml block bit rates
    assert BITS_PER_ELEM["f16"] == 16.0
    assert BITS_PER_ELEM["q8_0"] == 8.5
    assert BITS_PER_ELEM["q5_1"] == 6.0
    assert BITS_PER_ELEM["q4_0"] == 4.5

def test_kv_bytes_symmetric_f16():
    # Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim 128 -> 7168 elems/token each for K,V
    b = kv_bytes_per_token(n_layer=28, n_head_kv=2, head_dim=128, kbits=16.0, vbits=16.0)
    # (7168 K + 7168 V) elems * 16 bits / 8 = 28672 bytes
    assert b == (28 * 2 * 128) * 2 * 16.0 / 8.0
    assert b == 28672.0

def test_kv_bytes_asymmetric_k8_v4():
    b = kv_bytes_per_token(n_layer=28, n_head_kv=2, head_dim=128, kbits=8.5, vbits=4.5)
    elems = 28 * 2 * 128
    assert b == elems * 8.5 / 8.0 + elems * 4.5 / 8.0
    # asymmetric is cheaper than symmetric f16
    assert b < 28672.0

def test_kv_bytes_scales_with_context_via_caller():
    # per-token bytes * tokens is the caller's job; check monotonicity in bits
    hi = kv_bytes_per_token(28, 2, 128, 16.0, 16.0)
    lo = kv_bytes_per_token(28, 2, 128, 4.5, 4.5)
    assert lo < hi


# --- Pareto frontier (minimize bytes, minimize ppl) ------------------------

def test_pareto_frontier_basic():
    # points: (label, bytes, ppl). Lower bytes and lower ppl both better.
    pts = [
        ("f16", 28672, 7.15),     # cheapest ppl, most bytes
        ("q8", 15232, 7.14),      # fewer bytes, ~same ppl -> dominates f16
        ("k8v4", 11648, 7.16),    # even fewer bytes, slightly worse ppl -> on frontier
        ("q4", 8064, 1572.0),     # cheap but terrible ppl -> on frontier (cheapest bytes)
        ("dominated", 20000, 50.0),  # more bytes AND worse ppl than q8 -> dominated
    ]
    front = pareto_frontier(pts)
    labels = {p[0] for p in front}
    assert "q8" in labels
    assert "k8v4" in labels
    assert "q4" in labels          # cheapest bytes is always on the frontier
    assert "dominated" not in labels
    assert "f16" not in labels     # q8 dominates it (fewer bytes, better ppl)

def test_pareto_frontier_sorted_by_bytes():
    pts = [("a", 100, 5.0), ("b", 50, 6.0), ("c", 200, 4.0)]
    front = pareto_frontier(pts)
    byts = [p[1] for p in front]
    assert byts == sorted(byts)
    assert math.isfinite(front[0][2])
