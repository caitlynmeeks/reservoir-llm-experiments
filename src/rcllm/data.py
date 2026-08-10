"""text8 corpus loading + char-level encoding for experiment 02.

text8: 100M chars of cleaned Wikipedia, alphabet = 'a'-'z' + space (27
symbols). Standard splits: first 90M train / 5M val / 5M test, metric is
bits per character (bpc). Downloads on first use (~30MB zip) to
data/ — runs on the Mac; this container has no network access to the host.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

import numpy as np

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
VOCAB = " abcdefghijklmnopqrstuvwxyz"
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)  # 27


def load_text8(cache_dir: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (train, val, test) as uint8 id arrays (90M / 5M / 5M)."""
    os.makedirs(cache_dir, exist_ok=True)
    npy = os.path.join(cache_dir, "text8_ids.npy")
    if not os.path.exists(npy):
        zpath = os.path.join(cache_dir, "text8.zip")
        if not os.path.exists(zpath):
            print(f"downloading {TEXT8_URL} ...")
            urllib.request.urlretrieve(TEXT8_URL, zpath)
        with zipfile.ZipFile(zpath) as zf:
            text = zf.read("text8").decode("ascii")
        ids = np.frombuffer(text.encode("ascii"), dtype=np.uint8).copy()
        # remap ascii -> 0..26
        lut = np.zeros(128, dtype=np.uint8)
        for c, i in CHAR2ID.items():
            lut[ord(c)] = i
        ids = lut[ids]
        np.save(npy, ids)
    ids = np.load(npy)
    return ids[:90_000_000], ids[90_000_000:95_000_000], ids[95_000_000:]


def one_hot(ids: np.ndarray, dtype=np.float32) -> np.ndarray:
    """(...,) int -> (..., V) one-hot."""
    out = np.zeros(ids.shape + (V,), dtype=dtype)
    np.put_along_axis(out, ids[..., None].astype(np.int64), 1.0, axis=-1)
    return out


def as_parallel_segments(ids: np.ndarray, n_segments: int, seg_len: int | None = None):
    """Split a long stream into S equal segments for batched reservoir runs.
    Returns (S, T) id array. Each segment gets its own washout downstream."""
    if seg_len is None:
        seg_len = len(ids) // n_segments
    usable = n_segments * seg_len
    return ids[:usable].reshape(n_segments, seg_len)
