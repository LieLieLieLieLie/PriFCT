"""Simultaneous DP-noise confidence sets for clean prototype decisions.

This certifies agreement with the NONPRIVATE same-map nearest-prototype
classifier. It cannot certify true labels or cross-view semantic correctness.
It uses only one released sum/count vector, and needs no new private query.
"""
import numpy as np
from scipy.stats import chi2,norm
from scipy.spatial.distance import cdist

def prototype_radii(joint,alpha=.05):
    if not 0<alpha<1:raise ValueError('alpha must be in (0,1)')
    raw=np.asarray(joint.raw,float);k,d=raw.shape;r=d-1
    message,_=joint.decode()
    if joint.sigma_query==0:
        return message.prototypes,np.zeros(k),{'feature_radius':0.,'count_radius':0.,'alpha':alpha}
    # K vector-norm and K two-sided scalar events, each alpha/(2K).
    feature=joint.sigma_query*np.sqrt(chi2.ppf(1-alpha/(2*k),r))
    count=joint.sigma_query*norm.isf(alpha/(4*k))
    mass=raw[:,-1];floor=joint.count_floor/joint.n_public
    denominator=np.maximum(mass,floor)
    # The same public floor is used by the actual decoder and its clean
    # counterpart. max(m,floor) is 1-Lipschitz, including empty classes.
    # The clean denominator >=max(floor,noisy_mass-count_radius).
    mean_norm=np.linalg.norm(raw[:,:-1],axis=1)/denominator
    radii=(feature+mean_norm*count)/np.maximum(floor,mass-count)
    radii=np.minimum(radii,2.) # Both clipped prototype and clean mean in unit ball.
    return message.prototypes,radii,{'feature_radius':float(feature),'count_radius':float(count),'alpha':alpha,
        'scope':'simultaneous clean same-map prototype inclusion; not semantic correctness'}

def certify_predictions(x,joint,alpha=.05):
    prototypes,radii,details=prototype_radii(joint,alpha)
    distance=cdist(x,prototypes);winner=distance.argmin(1)
    upper=distance[np.arange(len(x)),winner]+radii[winner]
    lower=np.maximum(distance-radii[None,:],0)
    lower[np.arange(len(x)),winner]=np.inf
    accepted=upper<lower.min(1)-1e-12
    return winner,accepted,{'radii':radii,'details':details,
        'meaning':'accepted decisions agree with clean nearest-prototype simultaneously with confidence >=1-alpha; no true-label guarantee'}
