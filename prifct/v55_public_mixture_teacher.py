"""Private class-centroid posterior with a public total-covariance reference.

The mixture covariance identity motivates subtracting the released between-
class term from a public total covariance. PSD projection and positive ridge
are post-processing. Shared within-class covariance and population compatibility
are utility assumptions. No protected second moments or source rows are read.
"""
import numpy as np

def fit(x,raw,n_source,ridge,strength):
    x=np.asarray(x,float);raw=np.asarray(raw,float);n,d=x.shape;k=len(raw)
    if strength not in [0.,.5,1.] or ridge<=0:raise ValueError('Registered settings required')
    frequencies=np.maximum(raw[:,-1],1./n_source);q=frequencies/frequencies.sum()
    centers=raw[:,:-1]/frequencies[:,None]
    centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1.)
    mean=x.mean(0);centered=x-mean;means=centers-mean
    total=centered.T@centered/n;between=means.T@(q[:,None]*means)
    covariance=(total-strength*between);covariance=(covariance+covariance.T)/2
    values,basis=np.linalg.eigh(covariance)
    inverse=basis@np.diag(1/(np.maximum(values,0.)+ridge))@basis.T
    weight=np.r_[inverse@means.T,(-.5*np.sum((means@inverse)*means,axis=1)+np.log(q))[None,:]]
    weight-=weight.mean(1,keepdims=True)
    return weight,mean,dict(public_total_covariance=True,protected_second_moments_accessed=False,
        between_class_term_from_released_packet=True,minimum_unprojected_eigenvalue=float(values.min()),negative_eigenvalues=int((values<0).sum()),
        projected_minimum_curvature=ridge,source_class_means_unit_clipped=True)
