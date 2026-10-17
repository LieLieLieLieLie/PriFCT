"""Public paired regression of bounded source functions rather than points.

The kernel conditional-mean construction is known. This module supplies an
explicit resource contract for testing its coupling to private transport.
No public pairs share identities with the private or evaluated cohorts.
"""
from dataclasses import dataclass
import numpy as np
from prifct.v6_public_relation import squared_distance,unit_clip


@dataclass(frozen=True)
class FunctionalBridge:
    point_bridge: object
    signature: object
    bandwidth: float
    coefficients: np.ndarray

    def source(self,raw):
        return self.signature.transform(self.point_bridge.source(raw))

    def target(self,raw):
        standardized=self.point_bridge.target_scaler.transform(raw)
        kernel=np.exp(-squared_distance(standardized,self.point_bridge.target_landmarks)/self.bandwidth)
        return unit_clip(kernel@self.coefficients)


def fit_functional_bridge(point_bridge,signature,source_public,target_public,*,ridge,bandwidth_factor):
    if len(source_public)!=len(target_public) or ridge<=0 or bandwidth_factor<=0:
        raise ValueError('Independent true public pairs and positive parameters required')
    truth=signature.transform(point_bridge.source(source_public))
    raw=point_bridge.target_scaler.transform(target_public)
    bandwidth=point_bridge.target_bandwidth*bandwidth_factor
    kernel=np.exp(-squared_distance(raw,point_bridge.target_landmarks)/bandwidth)
    coefficients=np.linalg.solve(kernel.T@kernel+ridge*np.eye(kernel.shape[1]),kernel.T@truth)
    return FunctionalBridge(point_bridge,signature,bandwidth,coefficients)


class IdentitySignature:
    def transform(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or not np.isfinite(x).all() or np.max(np.linalg.norm(x,axis=1),initial=0)>1+1e-8:
            raise ValueError('Finite bounded precomputed functions required')
        return x
