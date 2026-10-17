"""Lossless probability container; decoded float64 bits must match originals.

Normalized float32 seeds usually recover float64 research probabilities.
Any row that does not recover bit-for-bit is stored as exact float64. This is
storage compression only; no scores, models, predictions or experiments change.
"""
import hashlib
from pathlib import Path
import numpy as np


def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pack(probability):
    original=np.asarray(probability)
    if original.dtype!=np.float64 or original.ndim!=2 or not np.isfinite(original).all() or original.min()<0:
        raise ValueError('Finite nonnegative float64 matrix required')
    seed=original.astype(np.float32)
    normalizer=seed.astype(np.float64).sum(1)
    for _ in range(5):
        reconstructed=seed.astype(np.float64)/normalizer[:,None]
        failed=np.any(reconstructed.view(np.uint64)!=original.view(np.uint64),axis=1)
        if not failed.any():break
        candidate=(original[failed]*normalizer[failed,None]).astype(np.float32)
        seed[failed]=candidate;normalizer[failed]=candidate.astype(np.float64).sum(1)
    reconstructed=seed.astype(np.float64)/normalizer[:,None]
    failed=np.flatnonzero(np.any(reconstructed.view(np.uint64)!=original.view(np.uint64),axis=1))
    payload=dict(probability=seed,probability_normalizer=normalizer,probability_exact_rows=failed,
        probability_exact_values=original[failed],probability_decoded_sha256=np.array(digest(original)))
    assert digest(decode(payload))==digest(original)
    return payload


def decode(container):
    if 'probability_normalizer' not in container:return np.asarray(container['probability'])
    probability=container['probability'].astype(np.float64)/container['probability_normalizer'][:,None]
    probability[container['probability_exact_rows']]=container['probability_exact_values']
    if digest(probability)!=str(container['probability_decoded_sha256']):raise RuntimeError('Lossless probability checksum mismatch')
    return probability


def load(path):
    path=Path(path)
    if not path.exists():path=path.with_name(path.stem+'.packed.npz')
    return np.load(path)
