"""Known public paired relation maps for isolated v6 research.

Ordered landmarks and regularized kernel CCA are shared information resources,
not the proposed mechanism innovation. Fit accepts public covariates only.
Every source/target transform is row-local and projected to the unit ball
before the single class joint-sum/count query. No protected-cohort centering.
"""
from dataclasses import dataclass
import sys
import numpy as np
try:
    import torch
except Exception:
    for module_name in [name for name in sys.modules if name == 'torch' or name.startswith('torch.')]:
        sys.modules.pop(module_name, None)
    torch = None
from scipy.linalg import eigh, svd


def unit_clip(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.)


def squared_distance(x, y, device='auto', batch_size=2048):
    if torch is None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        parts = []
        for first in range(0, len(x), batch_size):
            a = x[first:first + batch_size]
            parts.append(np.maximum((a * a).sum(1)[:, None] + (y * y).sum(1)[None, :] -
                                    2 * a @ y.T, 0.0))
        return np.concatenate(parts) if parts else np.empty((0, len(y)))
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    b = torch.as_tensor(np.asarray(y), dtype=torch.float64, device=device)
    parts = []
    for first in range(0, len(x), batch_size):
        a = torch.as_tensor(np.asarray(x[first:first+batch_size]), dtype=torch.float64, device=device)
        parts.append(torch.clamp((a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] -
                                2*a@b.T, min=0).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0, len(y)))


@dataclass(frozen=True)
class PublicStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    logarithmic: bool

    @classmethod
    def fit(cls, public, logarithmic=False):
        x = np.asarray(public, dtype=np.float64)
        if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
            raise ValueError('At least three finite public covariate rows required')
        z = np.sign(x)*np.log1p(np.abs(x)) if logarithmic else x
        return cls(z.mean(0), np.maximum(z.std(0), 1e-8), logarithmic)

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Finite rows in the fitted public feature space required')
        z = np.sign(x)*np.log1p(np.abs(x)) if self.logarithmic else x
        return unit_clip((z-self.mean)/self.scale)


@dataclass(frozen=True)
class PairedRelationMap:
    source_standardizer: PublicStandardizer
    target_standardizer: PublicStandardizer
    source_landmarks: np.ndarray
    target_landmarks: np.ndarray
    source_bandwidth: float
    target_bandwidth: float
    source_center: np.ndarray
    target_center: np.ndarray
    source_basis: np.ndarray
    target_basis: np.ndarray
    landmark_rows: np.ndarray
    mode: str
    canonical_correlations: np.ndarray
    dimension: int

    def _relation(self, x, source, device='auto'):
        standard = self.source_standardizer if source else self.target_standardizer
        landmarks = self.source_landmarks if source else self.target_landmarks
        bandwidth = self.source_bandwidth if source else self.target_bandwidth
        z = standard.transform(x)
        # Ordered coordinates refer to the same physical public pair IDs.
        return np.exp(-squared_distance(z, landmarks, device)/bandwidth)/np.sqrt(len(landmarks))

    def source(self, x, device='auto'):
        return unit_clip((self._relation(x, True, device)-self.source_center)@self.source_basis)

    def target(self, x, device='auto'):
        return unit_clip((self._relation(x, False, device)-self.target_center)@self.target_basis)

    def calibration(self, paired_source, paired_target, device='auto'):
        if len(paired_source) != len(paired_target) or len(paired_source) < 2:
            raise ValueError('Independent paired public calibration rows required')
        a, b = self.source(paired_source, device), self.target(paired_target, device)
        difference = a-b
        return {'paired_mean_squared_difference': float(np.mean(np.sum(difference**2,axis=1))),
                'paired_coordinate_mse': np.mean(difference**2,axis=0),
                'paired_norm_quantiles': np.quantile(np.linalg.norm(difference,axis=1), [.5,.9,.95,1.]),
                'n_public_calibration': len(a),
                'scope': 'public cross-view residual diagnostic; not a semantic or DP confidence radius'}


def fit_public_relation(public_source, public_target, *, mode='cca', landmarks=64,
                        rank=16, ridge=.01, seed=74013, logarithmic=False, device='auto'):
    """Freeze known paired-landmark maps without labels or private arrays.

    CCA is a comparator/common representation choice, not a new contribution.
    Ordered mode projects both relation vectors by a single public PCA basis.
    Source and target landmarks are selected by identical public row IDs.
    """
    if len(public_source) != len(public_target):
        raise ValueError('Public source/target rows must be real paired observations')
    if mode not in ['cca', 'ordered'] or ridge <= 0 or rank < 1 or landmarks < 2:
        raise ValueError('Valid public mapping configuration required')
    sm, tm = PublicStandardizer.fit(public_source, logarithmic), PublicStandardizer.fit(public_target, logarithmic)
    xs, xt = sm.transform(public_source), tm.transform(public_target)
    ids = np.random.default_rng(seed).permutation(len(xs))[:min(landmarks,len(xs))]
    ls, lt = xs[ids], xt[ids]
    def bandwidth(landmarks):
        distance = squared_distance(landmarks,landmarks,device)
        return max(float(np.median(distance[np.triu_indices(len(landmarks),1)])),1e-4)
    ws, wt = bandwidth(ls), bandwidth(lt)
    a = np.exp(-squared_distance(xs,ls,device)/ws)/np.sqrt(len(ls))
    b = np.exp(-squared_distance(xt,lt,device)/wt)/np.sqrt(len(lt))
    ms, mt = a.mean(0), b.mean(0);a, b = a-ms, b-mt
    width = min(rank,len(ids),len(xs)-1)
    if mode == 'cca':
        def inverse_sqrt(c):
            values,vectors=eigh(c)
            return (vectors/np.sqrt(np.maximum(values,1e-12)))@vectors.T
        n = len(xs)-1
        isource = inverse_sqrt(a.T@a/n+ridge*np.eye(len(ids)))
        itarget = inverse_sqrt(b.T@b/n+ridge*np.eye(len(ids)))
        u,correlations,v = svd(isource@(a.T@b/n)@itarget,full_matrices=False)
        bs,bt = isource@u[:,:width],itarget@v[:width].T
        correlations = correlations[:width]
    else:
        _,basis=eigh((a.T@a+b.T@b)/(2*max(1,len(xs)-1)))
        bs=bt=basis[:,::-1][:,:width]
        correlations=np.array([])
    return PairedRelationMap(sm,tm,ls,lt,ws,wt,ms,mt,bs,bt,ids,mode,correlations,width)
