"""Independent public, unlabelled paired calibration.

The bridge is frozen before any private record is processed. CCA is a known
representation method, not a novelty claim. This changes the information
contract relative to the zero-proxy experiments.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh,svd
from revision.protocol import clip


@dataclass(frozen=True)
class PublicBridge:
    source_mean: np.ndarray
    source_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    source_basis: np.ndarray
    target_basis: np.ndarray
    calibration_error: float
    residual_variance: np.ndarray

    def source(self,x):
        z=((np.asarray(x)-self.source_mean)/self.source_scale)@self.source_basis
        return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-15)

    def target(self,x):
        z=((np.asarray(x)-self.target_mean)/self.target_scale)@self.target_basis
        return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-15)


def fit_bridge(public_source,public_target,validation_source,validation_target,
               dimension=16,ridge=.1):
    if len(public_source)!=len(public_target):raise ValueError('Public pairs required')
    xs,xt=np.asarray(public_source,float),np.asarray(public_target,float)
    ms,mt=xs.mean(0),xt.mean(0)
    ss,st=np.maximum(xs.std(0),1e-6),np.maximum(xt.std(0),1e-6)
    a,b=(xs-ms)/ss,(xt-mt)/st;n=len(a)
    cs=a.T@a/(n-1)+ridge*np.eye(a.shape[1]);ct=b.T@b/(n-1)+ridge*np.eye(b.shape[1])
    cross=a.T@b/(n-1)
    def invsqrt(c):
        w,v=eigh(c);return (v/np.sqrt(np.maximum(w,1e-10)))@v.T
    ws,wt=invsqrt(cs),invsqrt(ct)
    u,d,vh=svd(ws@cross@wt,full_matrices=False)
    r=min(dimension,len(d))
    # Common signs are intrinsic to the same SVD, no private-label matching.
    s_basis=ws@u[:,:r];t_basis=wt@vh.T[:,:r]
    bridge=PublicBridge(ms,ss,mt,st,s_basis,t_basis,0.,np.zeros(r))
    z1,z2=bridge.source(validation_source),bridge.target(validation_target)
    residual=z1-z2
    return PublicBridge(ms,ss,mt,st,s_basis,t_basis,
                        float(np.mean(np.sum(residual**2,axis=1))),np.mean(residual**2,axis=0))
