#!/usr/bin/env python3
"""Run the KV-cache precision sweep: for each (K type, V type, context) config, call
llama.cpp's llama-perplexity with that cache precision and record perplexity plus the
exact KV bytes per token. Flash attention is on throughout (quantized V requires it, and
it keeps the cache type the only variable). Writes results/kvbits.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvbits import BITS_PER_ELEM, kv_bytes_per_token, parse_ppl  # noqa: E402
from ggufmeta import kv_geometry                                 # noqa: E402

TYPES = ["f16", "q8_0", "q5_1", "q4_0"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="data/ppl_text.txt")
    ap.add_argument("--bin", default="/opt/homebrew/bin/llama-perplexity")
    ap.add_argument("--ctxs", default="512,2048")
    ap.add_argument("--out", default="results/kvbits.jsonl")
    ap.add_argument("--types", default=",".join(TYPES))
    args = ap.parse_args()

    n_layer, n_head_kv, head_dim = kv_geometry(args.model)
    types = args.types.split(",")
    ctxs = [int(c) for c in args.ctxs.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    total = len(ctxs) * len(types) * len(types)
    done = 0
    with open(args.out, "w", buffering=1) as out:
        for ctx in ctxs:
            for kt in types:
                for vt in types:
                    cmd = [
                        args.bin, "-m", args.model, "-f", args.text,
                        "-c", str(ctx), "-ngl", "99", "-fa", "on",
                        "--cache-type-k", kt, "--cache-type-v", vt,
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    ppl, err = parse_ppl(proc.stdout + proc.stderr)
                    kbytes = kv_bytes_per_token(n_layer, n_head_kv, head_dim,
                                                BITS_PER_ELEM[kt], 0.0)
                    vbytes = kv_bytes_per_token(n_layer, n_head_kv, head_dim,
                                                0.0, BITS_PER_ELEM[vt])
                    rec = {
                        "ctx": ctx, "k": kt, "v": vt, "ppl": ppl, "ppl_err": err,
                        "kv_bytes_per_tok": kbytes + vbytes,
                        "ok": ppl is not None,
                    }
                    out.write(json.dumps(rec) + "\n")
                    done += 1
                    ppls = f"{ppl:.3f}" if ppl is not None else "FAIL"
                    print(f"  [{done}/{total}] ctx={ctx} K={kt:<4} V={vt:<4} "
                          f"ppl={ppls} bytes/tok={kbytes+vbytes:.0f}", flush=True)
    print(f"# SWEEP_DONE configs={done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
