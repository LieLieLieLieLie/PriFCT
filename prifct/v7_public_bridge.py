"""Known public paired kernel regression bridge, not a novel mechanism.

Uses independent paired covariates only. Fits are frozen before private data;
transformations are row-local with a final unit bound. Raw modalities remain
physically distinct. Public bridge quality does not establish class semantics.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import svd
from prifct.v6_public_relation import squared_distance,unit_clip


@dataclass(frozen=True)
class PublicLogScale:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or len(x)<3 or not np.isfinite(x).all():raise ValueError('Finite public matrix required')
        z=np.sign(x)*np.log1p(abs(x))
        return cls(z.mean(0),np.maximum(z.std(0),1e-6))

    def transform(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():raise ValueError('Finite same-view records required')
        return (np.sign(x)*np.log1p(abs(x))-self.mean)/self.std


@dataclass(frozen=True)
class PublicRegressionBridge:
    source_scaler: PublicLogScale
    target_scaler: PublicLogScale
    source_basis: np.ndarray
    source_scale: float
    target_landmarks: np.ndarray
    target_bandwidth: float
    target_coefficients: np.ndarray

    def source(self,x):
        return unit_clip(self.source_scaler.transform(x)@self.source_basis/self.source_scale)

    def target(self,x):
        raw=self.target_scaler.transform(x)
        kernel=np.exp(-squared_distance(raw,self.target_landmarks)/self.target_bandwidth)
        return unit_clip(kernel@self.target_coefficients)


def fit_public_bridge(public_source,public_target,*,rank=32,landmarks=256,ridge=1.,bandwidth_factor=.5,seed=76101):
    if len(public_source)!=len(public_target):raise ValueError('True paired public records required')
    if ridge<=0 or bandwidth_factor<=0:raise ValueError('Positive public parameters required')
    sm=PublicLogScale.fit(public_source);tm=PublicLogScale.fit(public_target)
    xs=sm.transform(public_source);xt=tm.transform(public_target)
    _,_,basis=svd(xs,full_matrices=False)
    basis=basis[:min(rank,len(basis))].T
    norm=np.linalg.norm(xs@basis,axis=1)
    scale=max(float(np.quantile(norm,.95)),1e-6)
    truth=unit_clip(xs@basis/scale)
    indices=np.random.default_rng(seed).permutation(len(xs))[:min(landmarks,len(xs))]
    dictionary=xt[indices]
    distance=squared_distance(dictionary,dictionary)
    bandwidth=max(float(np.median(distance[np.triu_indices(len(indices),1)])),1e-6)*bandwidth_factor
    kernel=np.exp(-squared_distance(xt,dictionary)/bandwidth)
    coefficients=np.linalg.solve(kernel.T@kernel+ridge*np.eye(len(indices)),kernel.T@truth)
    return PublicRegressionBridge(sm,tm,basis,scale,dictionary,bandwidth,coefficients)
