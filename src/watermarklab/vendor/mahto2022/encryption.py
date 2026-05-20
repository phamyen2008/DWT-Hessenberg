from __future__ import annotations

import hashlib
import numpy as np


def _seed_from_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def encrypt_image(img: np.ndarray, key: str = "mahto2022-demo-key") -> tuple[np.ndarray, dict]:
    """Keyed permutation + XOR diffusion encryption for grayscale watermark.

    This is a transparent replacement for the cited SIE scheme, whose full code is
    not given in the paper. It is reversible and strongly key-sensitive.
    """
    arr = np.asarray(img, dtype=np.uint8)
    shape = arr.shape
    flat = arr.ravel()
    rng = np.random.default_rng(_seed_from_key(key))
    perm = rng.permutation(flat.size)
    xor_stream = rng.integers(0, 256, flat.size, dtype=np.uint8)
    encrypted_flat = np.bitwise_xor(flat[perm], xor_stream)
    return encrypted_flat.reshape(shape), {"shape": shape, "key_hash": hashlib.sha256(key.encode()).hexdigest()}


def decrypt_image(enc: np.ndarray, key: str = "mahto2022-demo-key") -> np.ndarray:
    arr = np.asarray(enc, dtype=np.uint8)
    flat = arr.ravel()
    rng = np.random.default_rng(_seed_from_key(key))
    perm = rng.permutation(flat.size)
    xor_stream = rng.integers(0, 256, flat.size, dtype=np.uint8)
    permuted_plain = np.bitwise_xor(flat, xor_stream)
    plain = np.empty_like(permuted_plain)
    plain[perm] = permuted_plain
    return plain.reshape(arr.shape)
