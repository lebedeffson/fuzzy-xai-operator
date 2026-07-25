from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import numpy as np


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def tensor_digest(value: object) -> str:
    """Hash numeric tensor storage without materializing Python scalar lists."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays are not valid evidence tensors")
    contiguous = np.ascontiguousarray(array)
    header = {
        "dtype": contiguous.dtype.str,
        "shape": contiguous.shape,
        "byteorder": contiguous.dtype.byteorder,
    }
    hasher = hashlib.sha256()
    hasher.update(_canonical_json(header))
    hasher.update(memoryview(contiguous).cast("B"))
    return hasher.hexdigest()


def merkle_root(named_digests: Iterable[tuple[str, str]]) -> str:
    leaves = []
    for name, digest in sorted(named_digests):
        name_bytes = name.encode()
        try:
            digest_bytes = bytes.fromhex(digest)
        except ValueError as error:
            raise ValueError(f"Merkle digest is not hexadecimal for {name}") from error
        leaf = hashlib.sha256()
        leaf.update(b"fuzzyxai-merkle-leaf-v1\0")
        leaf.update(len(name_bytes).to_bytes(4, "big"))
        leaf.update(name_bytes)
        leaf.update(len(digest_bytes).to_bytes(4, "big"))
        leaf.update(digest_bytes)
        leaves.append(leaf.digest())
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()
