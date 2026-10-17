"""Public soft-cell functions and Gaussian histogram consistency decoding."""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from prifct.v6_public_relation import squared_distance

def simplex(x):
    x=np.asarray(x,float);ordered=-np.sort(-x,axis=1)
    cumulative=np.cumsum(ordered,axis=1)-1.
    active=ordered-cumulative/np.arange(1,x.shape[1]+1)>0
    count=active.sum(1)
    threshold=cumulative[np.arange(len(x)),count-1]/count
    return np.maximum(x-threshold[:,None],0.)

@dataclass(frozen=True)
class CellSignature:
    centers:np.ndarray
    bandwidth:float
    def transform(self,point):
        return softmax(-squared_distance(point,self.centers)/self.bandwidth,axis=1)
    @classmethod
    def public_fit(cls,point):
        fit=KMeans(n_clusters=32,n_init=5,random_state=88111).fit(point)
        distance=squared_distance(point,fit.cluster_centers_)
        width=max(float(np.median(distance.min(1)))*4,1e-3)
        return cls(fit.cluster_centers_,width)

def histogram(raw,sigma,consistency=True,smoothing=.5):
    raw=np.asarray(raw,float)
    feature=raw[:,:-1];count=raw[:,-1]
    if consistency:
        # Project each noisy feature/count pair onto nonnegative histograms
        # with count equal to its row sum. Both coordinates have equal noise.
        low=-np.max(abs(feature),axis=1)-abs(count)-1.
        high=np.max(abs(feature),axis=1)+abs(count)+1.
        for _ in range(70):
            middle=(low+high)/2
            residual=middle+np.maximum(feature+middle[:,None],0.).sum(1)-count
            high=np.where(residual>0,middle,high)
            low=np.where(residual<=0,middle,low)
        h=np.maximum(feature+((low+high)/2)[:,None],0.)
    else:h=np.maximum(feature,0.)
    q=np.maximum(h.sum(1),1e-12);q/=q.sum()
    h=h+float(smoothing)*sigma*q[:,None]
    cellmass=h.sum(0)
    teacher=np.divide(h,cellmass[None,:],out=np.broadcast_to(q[:,None],h.shape).copy(),where=cellmass[None,:]>1e-15).T
    assert np.isfinite(teacher).all() and np.allclose(teacher.sum(1),1.)
    return teacher,q,h
