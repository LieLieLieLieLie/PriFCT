"""Full labelled-record replace-one DP of joint class sums and counts.

Known class universe and public total cohort size are required. Individual
features, labels and class counts are not declared public. Class membership
presence/absence (add/remove adjacency) is not claimed. All maps are frozen
using independent public proxies before this query.
"""
from dataclasses import dataclass
import numpy as np
from core.privacy import compute_sigma
from revision.protocol import Message,clip,align,assignment
from scipy.stats import beta


@dataclass(frozen=True)
class JointMessage:
    raw: np.ndarray
    sigma_query: float
    epsilon: float
    n_public: int
    delta: float=1e-5
    count_floor: float=1.

    def decode(self):
        # The lower denominator bound is a public numerical convention.
        noisy_counts=self.raw[:,-1]*self.n_public
        denominator=np.maximum(noisy_counts,self.count_floor)
        means=self.raw[:,:-1]*self.n_public/denominator[:,None]
        prototypes=clip(means)
        # Local linearisation used only as descriptive scale, NOT DP calibration.
        sigma=np.full((len(means),1),self.n_public*self.sigma_query)/denominator[:,None]
        return Message(prototypes,noisy_counts,sigma,self.epsilon),noisy_counts


def joint_query(x,y,k,n_public):
    if n_public<=0 or len(x)!=n_public:raise ValueError('Fixed public N required')
    y=np.asarray(y)
    if y.shape!=(n_public,) or np.any((y<0)|(y>=k)) or not np.all(y==y.astype(int)):
        raise ValueError('Labels must belong to the fixed public class universe')
    x=clip(np.asarray(x,float));q=np.zeros((k,x.shape[1]+1))
    for c in range(k):
        q[c,:-1]=x[y==c].sum(0)/n_public
        q[c,-1]=np.sum(y==c)/n_public
    return q


def release_joint(x,y,k,epsilon,seed=None,delta=1e-5,count_floor=1.):
    n=len(x);q=joint_query(x,y,k,n)
    # Same label: count cancels, feature difference <=2.
    # Different labels: two blocks have squared norm <=1+1+1+1=4.
    sigma=float(compute_sigma(2./n,epsilon,delta))
    if seed is None:
        # Deployment randomness is not a public reproducibility seed. Seeded
        # releases below are for public-benchmark simulations only; publishing
        # their seed reveals their noise and is NOT a private deployment.
        import random
        secure=random.SystemRandom()
        noise=np.array([secure.gauss(0.,sigma) for _ in range(q.size)]).reshape(q.shape)
    else:noise=np.random.default_rng(seed).normal(size=q.shape)*sigma
    raw=q+noise
    return JointMessage(raw,sigma,epsilon,n,delta,count_floor)


def screen_joint(target,joint,rho,s,mapping,gap,seed,multipliers=(1.,),counts=(100,),mass=.9,tau=.05):
    rng=np.random.default_rng(seed)
    noise=rng.normal(size=(max(counts),*joint.raw.shape))*joint.sigma_query
    records=[]
    for mul in multipliers:
        same=[];dev=[]
        for z in noise:
            pert=JointMessage(joint.raw+mul*z,joint.sigma_query,joint.epsilon,joint.n_public,joint.delta,joint.count_floor)
            message,_=pert.decode()
            sp,mp,_,_=align(target,message,rho,mass=mass,tau=tau,declared_coordinates=True)
            same.append(bool(np.array_equal(mp,mapping)));dev.append(float(np.max(abs(sp-s))))
        for b in counts:
            ns=int(np.sum(same[:b]));lo=float(beta.ppf(.05,ns,b-ns+1)) if ns else 0.
            eta=float(np.quantile(dev[:b],.95));rad=gap/(2*len(mapping))
            records.append({'multiplier':mul,'B':b,'successes':ns,'stability':float(np.mean(same[:b])),
                  'stability_lower95':lo,'gap':gap,'radius':rad,'eta95':eta,
                  'gap_accepted':bool(gap>1e-12 and eta<rad),
                  'accepted':bool(gap>1e-12 and lo>=.9),'status':'stable' if gap>1e-12 and lo>=.9 else 'abstain'})
    return records
