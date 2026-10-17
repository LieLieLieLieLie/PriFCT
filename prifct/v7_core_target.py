"""Retained target geometry, without unused legacy graph/anchor computation.

Same KMeans, GPU distance, width and soft responsibilities as TargetProblem.
No target labels, evaluator, pairwise N-by-N matrix or source input accepted.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from prifct.method import gpu_sqdist


@dataclass
class CoreTarget:
    x:np.ndarray
    xtest:np.ndarray
    centers:np.ndarray
    labels:np.ndarray
    responsibilities:np.ndarray
    test_responsibilities:np.ndarray
    k:int
    width:float
    target_mass:np.ndarray

    @classmethod
    def build(cls,adaptation,test,classes,seed,components=None):
        x=np.asarray(adaptation,float);xt=np.asarray(test,float);j=2*classes if components is None else int(components)
        if x.ndim!=2 or xt.ndim!=2 or x.shape[1]!=xt.shape[1] or not np.isfinite(x).all() or not np.isfinite(xt).all():
            raise ValueError('Finite prepared target functions required')
        if j<1 or j>len(x):raise ValueError('Valid fixed target cardinality required')
        km=KMeans(n_clusters=j,n_init=5,random_state=seed).fit(x)
        centers=km.cluster_centers_;labels=km.labels_;count=np.bincount(labels,minlength=j)
        if np.any(count==0):raise ValueError('Required target components must be nonempty')
        distance=gpu_sqdist(x,centers);test_distance=gpu_sqdist(xt,centers)
        width=max(float(np.median(distance.min(1)))*4,1e-3)
        return cls(x,xt,centers,labels,softmax(-distance/width,axis=1),softmax(-test_distance/width,axis=1),
            classes,width,count.astype(float)/count.sum())
