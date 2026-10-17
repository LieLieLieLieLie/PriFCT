"""Fixed per-record feature map used by the resubmission experiments.

No fitted source statistics, cohort extrema, labels, or test records enter it.
Clipping after this map preserves replace-one feature-record sensitivity.
"""
import numpy as np


def fixed_record_map(x, log_transform=True, radius=1.0):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite input records must be resolved locally")
    if log_transform:
        x = np.sign(x) * np.log1p(np.abs(x))
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    # Unit normalisation is per record, not a statistic fitted on the cohort.
    return (radius * x / np.maximum(norm, 1e-15)).astype(np.float64)
