"""Minimal GGUF metadata reader: pull the KV-cache geometry (n_layer, n_head_kv,
head_dim) from a .gguf file so the byte accounting is exact for any model, not hardcoded.
Reads only the metadata header, not the tensor data.
"""
from __future__ import annotations

import struct
from typing import BinaryIO


def _rstr(f: BinaryIO) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def _rval(f: BinaryIO, t: int) -> object:
    if t == 8:  # string
        return _rstr(f)
    if t in (0, 1):
        return struct.unpack("<b" if t == 1 else "<B", f.read(1))[0]
    if t in (2, 3):
        return struct.unpack("<h" if t == 3 else "<H", f.read(2))[0]
    if t in (4, 5):
        return struct.unpack("<i" if t == 5 else "<I", f.read(4))[0]
    if t == 6:
        return struct.unpack("<f", f.read(4))[0]
    if t == 7:
        return struct.unpack("?", f.read(1))[0]
    if t in (10, 11):
        return struct.unpack("<q" if t == 11 else "<Q", f.read(8))[0]
    if t == 12:
        return struct.unpack("<d", f.read(8))[0]
    if t == 9:  # array
        et = struct.unpack("<I", f.read(4))[0]
        ln = struct.unpack("<Q", f.read(8))[0]
        return [_rval(f, et) for _ in range(ln)]
    raise ValueError(f"unknown gguf value type {t}")


def read_meta(path: str, wanted: set[str]) -> dict[str, object]:
    """Return the subset of metadata keys whose name contains any string in `wanted`."""
    out: dict[str, object] = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a gguf file")
        struct.unpack("<I", f.read(4))  # version
        struct.unpack("<Q", f.read(8))  # n_tensors
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            k = _rstr(f)
            t = struct.unpack("<I", f.read(4))[0]
            v = _rval(f, t)
            if any(s in k for s in wanted):
                out[k] = v
    return out


def kv_geometry(path: str) -> tuple[int, int, int]:
    """(n_layer, n_head_kv, head_dim) for the model at `path`."""
    m = read_meta(path, {"block_count", "head_count", "head_count_kv", "embedding_length"})
    n_layer = _pick_int(m, "block_count")
    n_head = _pick_int(m, "attention.head_count", exclude="kv")
    n_head_kv = _pick_int(m, "head_count_kv")
    n_embd = _pick_int(m, "embedding_length")
    head_dim = n_embd // n_head
    return n_layer, n_head_kv, head_dim


def _pick_int(m: dict[str, object], needle: str, exclude: str = "\x00") -> int:
    for k, v in m.items():
        if needle in k and exclude not in k:
            if not isinstance(v, int):
                raise TypeError(f"{k} is not an int: {v!r}")
            return v
    raise KeyError(needle)
