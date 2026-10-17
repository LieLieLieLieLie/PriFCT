"""Public-only prototype catalogue denoising; no cross-view record pairs.

The catalogue is an empirical prior, not true source class prototypes.
Posterior weights are a heuristic under noisy counts; privacy follows solely
from postprocessing the valid joint query. No exact Bayesian claim is made.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.cluster import KMeans
from revision.protocol import Message,clip

def catalogue(public_codes,k,seed=88419):
    n=min(len(public_codes),max(2*k,16))
    return KMeans(n_clusters=n,n_init=5,random_state=seed).fit(public_codes).cluster_centers_

@dataclass
class CatalogueQuery:
    base:object
    codebook:np.ndarray
    mixing:float=.75
    width:float=.05
    @property
    def raw(self):return self.base.raw
    @property
    def sigma_query(self):return self.base.sigma_query
    @property
    def epsilon(self):return self.base.epsilon
    @property
    def delta(self):return self.base.delta
    @property
    def n_public(self):return self.base.n_public
    @property
    def count_floor(self):return self.base.count_floor
    def decode(self):
        original,counts=self.base.decode()
        distance=cdist(original.prototypes,self.codebook)**2
        weights=softmax(-distance/self.width,axis=1)
        center=weights@self.codebook
        prototypes=clip((1-self.mixing)*original.prototypes+self.mixing*center)
        return Message(prototypes,counts,original.sigma,self.epsilon),counts

def effective_radii(query,radii):
    # Unit-ball squared distances are 4-Lipschitz in the input prototype.
    # Softmax is nonexpansive from l_infinity to l_1 (Jacobian dual norm <=1).
    # Taking a convex combination of unit-ball codepoints is l_1->l_2 Lipschitz.
    return np.minimum(2.,(1-query.mixing)*radii+query.mixing*np.minimum(2.,4*radii/query.width))
