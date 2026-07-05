#!/usr/bin/env python3
"""Turn the KV-precision sweep into the frontier: the K-by-V perplexity matrix (the
asymmetry made visible), the quality-vs-KV-bytes Pareto frontier, and the recommended
bit allocation. Pure offline replay of measured perplexities.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvbits import pareto_frontier  # noqa: E402

TYPES = ["f16", "q8_0", "q5_1", "q4_0"]


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def fmt_ppl(x: float | None) -> str:
    if x is None:
        return "FAIL"
    return f"{x:.2f}" if x < 100 else f"{x:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/kvbits.jsonl")
    ap.add_argument("--out", default="bench_results/frontier.md")
    ap.add_argument("--tol", type=float, default=0.01, help="ppl tolerance vs f16 for 'lossless'")
    args = ap.parse_args()

    recs = load(args.inp)
    ctxs = sorted({r["ctx"] for r in recs})
    L: list[str] = []
    L.append("# kvbits: the KV-cache precision frontier and the key-vs-value asymmetry\n")
    L.append("Qwen2.5-1.5B-Instruct Q4_K_M, llama.cpp llama-perplexity, flash attention on, "
             "fixed English text. Perplexity is exact and deterministic. Lower is better. "
             "KV bytes/token are computed exactly from the model geometry and the ggml block "
             "bit rates (f16=16, q8_0=8.5, q5_1=6, q4_0=4.5 bits/elem).\n")

    for ctx in ctxs:
        cell = {(r["k"], r["v"]): r for r in recs if r["ctx"] == ctx}
        f16 = cell.get(("f16", "f16"), {}).get("ppl")
        L.append(f"## Context {ctx}: K-by-V perplexity matrix (rows = key cache, cols = value cache)\n")
        L.append("| K \\ V | " + " | ".join(TYPES) + " |")
        L.append("|---|" + "|".join(["---"] * len(TYPES)) + "|")
        for kt in TYPES:
            row = [fmt_ppl(cell.get((kt, vt), {}).get("ppl")) for vt in TYPES]
            L.append(f"| **{kt}** | " + " | ".join(row) + " |")
        L.append("")
        # Pareto frontier over all ok configs at this ctx
        pts = [(f"K{r['k']}/V{r['v']}", r["kv_bytes_per_tok"], r["ppl"])
               for r in recs if r["ctx"] == ctx and r["ok"] and r["ppl"] is not None]
        front = pareto_frontier(pts)
        L.append(f"### Pareto frontier (quality vs KV bytes/token), context {ctx}\n")
        L.append("| config | KV bytes/token | perplexity | vs f16 |")
        L.append("|:-------|---------------:|-----------:|-------:|")
        for label, byts, ppl in front:
            delta = f"{(ppl / f16 - 1) * 100:+.1f}%" if f16 else "n/a"
            L.append(f"| {label} | {byts:.0f} | {fmt_ppl(ppl)} | {delta} |")
        L.append("")
        # recommended: fewest bytes whose ppl is within tol of f16 (lossless zone)
        if f16:
            lossless = [(lab, b, p) for lab, b, p in pts if p <= f16 * (1 + args.tol)]
            if lossless:
                rec = min(lossless, key=lambda x: x[1])
                save = (1 - rec[1] / cell[("f16", "f16")]["kv_bytes_per_tok"]) * 100
                L.append(f"**Recommended at ctx {ctx}:** `{rec[0]}` at {rec[1]:.0f} bytes/token, "
                         f"perplexity {rec[2]:.2f} (within {args.tol*100:.0f}% of f16), "
                         f"a {save:.0f}% KV-cache reduction.\n")

    L.append("## What the matrix shows\n")
    L.append("Read down a column (fix the value cache, vary the key cache) versus across a row "
             "(fix the key cache, vary the value cache). If the key cache is the sensitive one, "
             "perplexity blows up as you move down the rows to q4_0 but stays flat across the "
             "value columns. That asymmetry is the whole point: bits spent on the key cache buy "
             "far more quality than bits spent on the value cache, so a symmetric budget is "
             "wrong and an asymmetric key-high / value-low allocation is the Pareto win.\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out} ({len(recs)} configs, ctxs={ctxs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
