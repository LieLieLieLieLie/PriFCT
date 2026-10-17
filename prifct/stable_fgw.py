"""Entropic FGW control with checked fixed marginals and monotone line search.

Optimises the standard nonconvex entropy-regularised FGW objective. Only a
stationary solution is sought; no global optimality claim or LP fallback.
"""
import numpy as np
import hashlib
from prifct.method import fixed_transport
_CACHE={}

def solve(cost,ct,cs,tau=.01,alpha=.5,max_iter=100,tol=1e-6):
    key=hashlib.sha256(b''.join(np.ascontiguousarray(a,dtype=np.float64).tobytes() for a in [cost,ct,cs])+repr((tau,alpha,max_iter,tol,cost.shape,ct.shape,cs.shape)).encode()).hexdigest()
    if key in _CACHE:
        t,state=_CACHE[key]
        return t.copy(),state.copy()
    n,k=cost.shape;plan=np.ones((n,k))/(n*k)
    def value(t):
        # Fixed-margin squared-distance constants do not affect optimisation.
        return float((1-alpha)*np.sum(cost*t)-2*alpha*np.sum((ct@t@cs.T)*t)
                     +tau*np.sum(t*np.log(np.maximum(t,1e-300))))
    history=[value(plan)]
    for iteration in range(max_iter):
        gradient=(1-alpha)*cost-4*alpha*(ct@plan@cs.T)
        candidate,_=fixed_transport(gradient,1.,tau)
        direction=candidate[:,:k]-plan
        residual=float(np.linalg.norm(direction))
        if residual<tol:break
        derivative=float(np.sum((gradient+tau*(np.log(np.maximum(plan,1e-300))+1))*direction))
        step=1.;current=history[-1]
        for _ in range(40):
            trial=plan+step*direction;next_value=value(trial)
            if next_value<=current+1e-4*step*min(derivative,0)+1e-12:break
            step*=.5
        if next_value>current+1e-10:raise RuntimeError('FGW line search failed')
        plan=trial;history.append(next_value)
    if not np.isfinite(plan).all():raise RuntimeError('FGW nonfinite result')
    assert np.max(abs(plan.sum(1)-1/n))<1e-6 and np.max(abs(plan.sum(0)-1/k))<1e-6
    state={'iterations':iteration+1,'stationarity_residual':residual,'objective':history,'converged':bool(residual<tol),'max_iter':max_iter,'tol':tol}
    if len(_CACHE)>128:_CACHE.clear()
    _CACHE[key]=(plan.copy(),state.copy())
    return plan,state
