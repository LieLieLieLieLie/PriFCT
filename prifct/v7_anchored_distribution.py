"""Anchored distribution transport with a checked block objective.

All inputs are bounded RECEIVER functions and a single private message.
This is a development mechanism, not a validated superiority claim. Exact
block minimizers decrease the stated nonconvex joint objective; convergence
to a global optimum and semantic correctness are not asserted.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from prifct.v7_conditional_moments import component_means


@dataclass(frozen=True)
class ProjectedSignature:
    original: object
    basis: np.ndarray

    @classmethod
    def public_fit(cls,original,public,rank=32):
        functions=original.transform(public)
        _,_,vectors=svd(functions,full_matrices=False)
        basis=vectors[:min(rank,len(vectors))].T
        np.testing.assert_allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-12)
        return cls(original,basis)

    def transform(self,x):
        # Orthogonal projection is a contraction, with no private centering.
        return self.original.transform(x)@self.basis


def squared_cost(components,means):
    return np.maximum(.5*(np.sum(components**2,1)[:,None]+np.sum(means**2,1)[None,:]-2*components@means.T),0.)


def transport(cost,a,q,temperature,prior_strength):
    """Row-constrained KL-prior entropic transport, exact convex dual.

min Pi: <Pi,C> +tau KL(Pi||a q) +beta KL(Pi.T1||q), Pi1=a.
The dual uses finite-beta column potentials; beta=0 is the no-global-
coupling ablation, whose unique minimizer is the row posterior.
"""
    a=np.asarray(a,float);q=np.asarray(q,float)
    if np.any(a<=0) or np.any(q<=0) or temperature<=0 or prior_strength<0:
        raise ValueError('Positive margins/temperature and nonnegative prior required')
    a=a/a.sum();q=q/q.sum();logq=np.log(q)
    if prior_strength==0:
        conditional=softmax(logq[None,:]-cost/temperature,axis=1)
        return a[:,None]*conditional,dict(dual_gradient_infinity=0.,duality_gap=0.,iterations=0)
    beta=prior_strength;tau=temperature
    def objective(v):
        logit=logq[None,:]-(cost+v[None,:])/tau
        posterior=softmax(logit,axis=1);mass=(a[:,None]*posterior).sum(0)
        value=beta*np.sum(q*np.expm1(v/beta))+tau*np.sum(a*logsumexp(logit,axis=1))
        gradient=q*np.exp(v/beta)-mass
        return value,gradient
    fit=minimize(objective,np.zeros(len(q)),jac=True,method='L-BFGS-B',
        options={'ftol':1e-14,'gtol':1e-10,'maxiter':3000,'maxls':60})
    conditional=softmax(logq[None,:]-(cost+fit.x[None,:])/tau,axis=1)
    plan=a[:,None]*conditional
    gradient=float(np.max(abs(objective(fit.x)[1])))
    primal=transport_objective(plan,cost,a,q,tau,beta)
    # All row-marginal Pi are feasible, and every v gives a feasible dual bound.
    gap=float(primal+objective(fit.x)[0])
    if not np.isfinite(plan).all() or gradient>2e-7 or gap>1e-7 or gap< -1e-8:
        raise RuntimeError('Entropic block solver accuracy failed '+str((gradient,gap,fit.message)))
    return plan,dict(dual_gradient_infinity=gradient,duality_gap=max(gap,0.),iterations=int(fit.nit))


def kl(a,b):
    return float(np.sum(a*(np.log(np.maximum(a,1e-300))-np.log(b))-a+b))


def transport_objective(plan,cost,a,q,temperature,prior_strength):
    return float(np.sum(plan*cost))+temperature*kl(plan,a[:,None]*q[None,:])+prior_strength*kl(plan.sum(0),q)


def fit(problem,message,*,temperature=.02,prior_strength=.1,anchor_strength=.1,steps=10,uncertainty_weight=True,freeze_means=False,hard_condition=False):
    if anchor_strength<=0 or steps<1:raise ValueError('Positive anchoring and iteration count required')
    train=np.asarray(problem.x,float);centers=component_means(train,problem.labels,len(problem.centers))
    decoded,noisy_counts=message.decode();anchor=decoded.prototypes
    q=np.maximum(noisy_counts,1.);q/=q.sum();a=problem.target_mass
    dimension=anchor.shape[1]
    uncertainty=dimension*(message.n_public*message.sigma_query/np.maximum(noisy_counts,1.))**2
    # This is an explicit postprocessing regularizer, not an accuracy certificate.
    reliability=1/(1+uncertainty/.01) if uncertainty_weight else np.ones(problem.k)
    weight=anchor_strength*q*reliability
    means=anchor.copy();history=[];solver=[]
    def total(plan,state):
        return transport_objective(plan,squared_cost(centers,state),a,q,temperature,prior_strength)+.5*np.sum(weight*np.sum((state-anchor)**2,1))
    previous=np.inf
    for _ in range(steps):
        plan,details=transport(squared_cost(centers,means),a,q,temperature,prior_strength);solver.append(details)
        before=total(plan,means)
        update_plan=a[:,None]*np.eye(problem.k)[plan.argmax(1)] if hard_condition else plan
        update=anchor.copy() if freeze_means else (update_plan.T@centers+weight[:,None]*anchor)/(update_plan.sum(0)+weight)[:,None]
        after=total(plan,update)
        if not hard_condition and (after>before+1e-8 or before>previous+1e-7):raise RuntimeError('Alternating objective failed monotonicity')
        history.append([before,after]);change=float(np.max(np.linalg.norm(update-means,axis=1)));means=update;previous=after
        if change<1e-7:break
    plan,details=transport(squared_cost(centers,means),a,q,temperature,prior_strength);solver.append(details)
    teacher=np.eye(problem.k)[plan.argmax(1)] if hard_condition else plan/a[:,None]
    return dict(means=means,anchor=anchor,class_prior=q,plan=plan,teacher=teacher,objective_history=np.array(history),
        solver_diagnostics=solver,uncertainty_scale=uncertainty,reliability_weight=reliability,
        caveat='Exact unique SOFT block updates; joint problem nonconvex, no global/semantic guarantee. Hard-condition ablation uses rowargmax before moment update/final teacher and has no monotonic guarantee.')


def predict(problem,message,*,transport_mix=.75,**options):
    if not 0<=transport_mix<=1:raise ValueError('Convex prediction mixture required')
    model=fit(problem,message,**options)
    direct=softmax(-squared_cost(problem.xtest,model['means'])/options['temperature']+np.log(model['class_prior'])[None,:],axis=1)
    probability=(1-transport_mix)*direct+transport_mix*(problem.test_responsibilities@model['teacher'])
    probability=np.maximum(probability,1e-12);probability/=probability.sum(1,keepdims=True)
    return probability,model
