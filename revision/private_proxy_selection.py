"""Exploratory paired-public exponential prototype selection.

Exponential selection of public prototypes is established (e.g. DPPL).
The paired target-side proxy bypasses a common-coordinate requirement.
No novelty claim follows merely from this construction.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.spatial.distance import cdist
from revision.uncertain_moments import target_graph

@dataclass(frozen=True)
class PublicSensorMap:
    mean: np.ndarray
    scale: np.ndarray
    def transform(self,x):
        code=(np.asarray(x,float)-self.mean)/self.scale
        return code/np.maximum(np.linalg.norm(code,axis=1,keepdims=True),1e-15)

def fit_public_sensor_map(public):
    return PublicSensorMap(public.mean(0),np.maximum(public.std(0),1e-6))

def score_query(x,y,proxy,k):
    x=np.asarray(x,float);proxy=np.asarray(proxy,float)
    if np.max(np.linalg.norm(x,axis=1))>1+1e-10 or np.max(np.linalg.norm(proxy,axis=1))>1+1e-10:
        raise ValueError('Unit-ball fixed public maps required')
    if len(y)!=len(x) or np.any((y<0)|(y>=k)):raise ValueError('Fixed public label universe required')
    # No private denominator or private class-count release.
    score=np.zeros((k,len(proxy)))
    for c in range(k):score[c]=x[y==c].sum(0)@proxy.T
    return score

def select_proxies(x,y,proxy,k,epsilon,seed,draws=1):
    """Joint K-class exponential mechanism, epsilon-pure labelled replace-one.

    For tuple (a_1,...,a_K), utility=sum_i <x_i,proxy[a_yi]>.
    Replacing the entire (x,y) changes this utility by at most 2, regardless
    of K or empty classes. Factorised probabilities exp(eps*utility/4)
    therefore give ONE epsilon-pure joint release, not K separate budgets.
    R independent releases with eps/R compose to epsilon. Public index IDs
    are the only outgoing object; scores/posteriors are never published.
    """
    if draws<1:raise ValueError('Positive draws required')
    score=score_query(x,y,proxy,k)
    if np.isinf(epsilon):selected=np.repeat(score.argmax(1)[:,None],draws,axis=1)
    else:
        probability=softmax(score*(epsilon/draws)/4,axis=1)
        rng=np.random.default_rng(seed)
        selected=np.array([rng.choice(len(proxy),size=draws,p=p) for p in probability])
    return selected.astype(np.int32),{'epsilon':float(epsilon),'delta':0.,'draws':draws,
        'payload_bytes':k*draws*4,'mechanism':'joint exponential tuple; public unit codes; full-labelled-record replace-one',
        'private_posterior_released':False}

def predict_selected(xtr,xte,public_target,indices,variant='graph'):
    prototypes=public_target[indices].mean(1)
    if variant=='nearest':return -cdist(xte,prototypes)**2
    if variant!='graph':raise ValueError(variant)
    teacher=softmax(-cdist(xtr,prototypes)**2/.1,axis=1)
    graph=target_graph(xtr);q=teacher.copy()
    for _ in range(50):q=.8*teacher+.2*graph.transition@q
    return graph.interpolate(xte,q)
