"""Bounded labelled feature aggregation without an N-by-feature materialization.

This changes storage, not the native DP-HP/NTK feature or joint Gaussian query.
Chunk sums can differ by floating summation roundoff; the equivalence test
quantifies this, rather than claiming bit-identical accumulation.
"""
import numpy as np
from core.privacy import compute_sigma
from revision.joint_release import JointMessage


def release(source,y,classes,feature,epsilon,*,seed,delta=1e-5,device='cuda',batch_size=128):
    source=np.asarray(source,float);y=np.asarray(y)
    if source.ndim!=2 or not len(source) or not np.isfinite(source).all() or np.max(np.linalg.norm(source,axis=1))>1+1e-9:
        raise ValueError('Fixed public bounded source functions required')
    if y.shape!=(len(source),) or np.any((y<0)|(y>=classes)) or not np.all(y==y.astype(int)):
        raise ValueError('Fixed public class catalogue required')
    if batch_size<1:raise ValueError('Positive chunk size required')
    dimension=feature.transform(np.zeros((1,source.shape[1])),device).shape[1]
    query=np.zeros((classes,dimension+1));n=len(source)
    for start in range(0,n,batch_size):
        functions=feature.transform(source[start:start+batch_size],device)
        # Same final clipping as joint_query, including normalized e-NTK rows.
        functions/=np.maximum(np.linalg.norm(functions,axis=1,keepdims=True),1.)
        labels=y[start:start+batch_size]
        for c in range(classes):
            query[c,:-1]+=functions[labels==c].sum(0)/n
            query[c,-1]+=np.sum(labels==c)/n
    sigma=float(compute_sigma(2/n,epsilon,delta))
    noise=np.random.default_rng(seed).normal(size=query.shape)*sigma
    return JointMessage(query+noise,sigma,epsilon,n,delta)
