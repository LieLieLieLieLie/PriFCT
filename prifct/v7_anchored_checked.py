"""Same mathematical blocks, Newton refinement after L-BFGS roundoff stop.

The original pilot's tiny duality gaps nevertheless triggered its independent
gradient threshold. Preserve that pilot and recompute ALL configurations with
this stricter-gradient refinement rather than averaging surviving cases.
"""
import ast,inspect
import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp
from prifct import v7_anchored_distribution as original


def transport(cost,a,q,temperature,prior_strength):
    a=np.asarray(a,float);q=np.asarray(q,float)
    if np.any(a<=0) or np.any(q<=0) or temperature<=0 or prior_strength<0:raise ValueError('Positive public margins required')
    a=a/a.sum();q=q/q.sum();logq=np.log(q);tau=temperature;beta=prior_strength
    if beta==0:
        conditional=softmax(logq[None,:]-cost/tau,axis=1)
        return a[:,None]*conditional,dict(dual_gradient_infinity=0.,duality_gap=0.,iterations=0)
    def state(v,hessian=False):
        posterior=softmax(logq[None,:]-(cost+v[None,:])/tau,axis=1)
        prior=q*np.exp(v/beta);mass=(a[:,None]*posterior).sum(0)
        value=beta*np.sum(q*np.expm1(v/beta))+tau*np.sum(a*logsumexp(logq[None,:]-(cost+v[None,:])/tau,axis=1))
        gradient=prior-mass
        if not hessian:return value,gradient
        second=np.diag(prior/beta+mass/tau)-(posterior.T*a)@posterior/tau
        return value,gradient,second,posterior
    fit=minimize(state,np.zeros(len(q)),jac=True,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':3000,'maxls':60})
    v=fit.x;refinements=0
    for _ in range(40):
        value,gradient,hessian,posterior=state(v,True)
        if np.max(abs(gradient))<=1e-10:break
        direction=solve(hessian,-gradient,assume_a='pos')
        accepted=False;scale=1.
        for _ in range(50):
            candidate=v+scale*direction
            if np.max(candidate/beta)>700:scale*=.5;continue
            new_value,new_gradient=state(candidate)
            if new_value<=value+1e-14 and np.max(abs(new_gradient))<np.max(abs(gradient)):
                v=candidate;accepted=True;refinements+=1;break
            scale*=.5
        if not accepted:break
    value,gradient,_,posterior=state(v,True);plan=a[:,None]*posterior
    primal=original.transport_objective(plan,cost,a,q,tau,beta);gap=float(primal+value)
    error=float(np.max(abs(gradient)))
    if not np.isfinite(plan).all() or error>2e-9 or gap>1e-10 or gap< -1e-8:
        raise RuntimeError('Checked dual solver failed '+str((error,gap)))
    return plan,dict(dual_gradient_infinity=error,duality_gap=max(gap,0.),iterations=int(fit.nit)+refinements,
        newton_refinements=refinements)


# Copy ONLY the two unchanged block/prediction function bodies into a separate
# namespace. This leaves original frozen module bytes and globals untouched.
namespace=dict(original.__dict__);namespace['transport']=transport
tree=ast.parse(inspect.getsource(original))
nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in {'fit','predict'}]
assert len(nodes)==2
exec(compile(ast.Module(body=nodes,type_ignores=[]),original.__file__,'exec'),namespace)
fit=namespace['fit'];predict=namespace['predict']
