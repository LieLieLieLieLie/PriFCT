"""Convex private posterior reconstruction from a single existing DP packet.

The log partition uses independent unlabelled public source functions. This
is DP post-processing, not training on protected rows or their covariance.
Public/source marginal compatibility remains an explicit utility assumption.
"""
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize

def fit(source_public,raw,n_public,ridge):
    x=np.asarray(source_public,float);raw=np.asarray(raw,float)
    n,d=x.shape;k=len(raw)
    if ridge<=0 or raw.shape!=(k,d+1):raise ValueError('Positive ridge and matching released joint packet required')
    mean=x.mean(0);z=np.c_[x-mean,np.ones(n)]
    q=np.maximum(raw[:,-1]*n_public,1.);q/=q.sum()
    moment=raw[:,:-1].T-mean[:,None]*raw[:,-1][None,:]
    sufficient=np.r_[moment,q[None,:]]
    sufficient-=sufficient.mean(1,keepdims=True)
    # The 1/K scale matches the initial multinomial Hessian, rather than
    # treating squared loss and multinomial loss as identically scaled.
    penalty=np.full((d+1,1),ridge/k)
    def objective(vector):
        weight=vector.reshape(d+1,k)
        centered=weight-weight.mean(1,keepdims=True)
        logits=z@centered;p=softmax(logits,axis=1)
        value=np.mean(logsumexp(logits,axis=1))-np.sum(sufficient*centered)+.5*np.sum(penalty*weight*weight)
        gradient=z.T@p/n-sufficient;gradient-=gradient.mean(1,keepdims=True)
        gradient+=penalty*weight
        return value,gradient.ravel()
    solved=minimize(objective,np.zeros((d+1)*k),jac=True,method='L-BFGS-B',
                    options=dict(ftol=1e-14,gtol=1e-9,maxiter=2000,maxls=60))
    value,gradient=objective(solved.x);error=float(np.max(abs(gradient)))
    if not np.isfinite(solved.x).all() or error>2e-7:raise RuntimeError('Maxent gradient residual '+str(error))
    weight=solved.x.reshape(d+1,k);weight-=weight.mean(1,keepdims=True)
    return weight,mean,dict(gradient_infinity=error,iterations=int(solved.nit),objective=float(value),
        regularization=ridge/k,private_source_rows_accessed=0,private_source_covariance_accessed=False)

def features(x,mean):
    return np.c_[np.asarray(x,float)-mean,np.ones(len(x))]
