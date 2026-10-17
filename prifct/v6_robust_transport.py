"""Class-shared prototype adversary, isolated from frozen v5.

For each class all transport edges share one unknown prototype displacement.
The uncertainty set is a Euclidean ball around the released signature-space
prototype. Entropy gives a strictly convex plan objective for positive tau.
CVXPY/Clarabel is a CPU conic optimizer; feature matrices can be built on CUDA.
Returned floating-point diagnostics are NOT directed-arithmetic certificates.
"""
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from scipy.special import xlogy


def squared_cost(target, prototypes):
    difference = prototypes[None,:,:]-target[:,None,:]
    return np.sum(difference**2,axis=2)/4, difference


def robust_objective(plan, target, prototypes, radii, tau=.01, reject_cost=.5):
    """Exact formula evaluated in floating point, including constant terms."""
    cost,difference=squared_cost(np.asarray(target),np.asarray(prototypes))
    k=len(prototypes);class_plan=np.asarray(plan)[:,:k]
    shared=np.einsum('jk,jkd->kd',class_plan,difference)
    b=class_plan.sum(0)
    objective=float(np.sum(class_plan*cost)+np.sum(b*np.asarray(radii)**2)/4+
                    np.sum(np.asarray(radii)*np.linalg.norm(shared,axis=1))/2+
                    tau*np.sum(xlogy(plan,plan)))
    if plan.shape[1]==k+1:objective+=reject_cost*float(plan[:,-1].sum())
    return objective


@dataclass(frozen=True)
class RobustTransportResult:
    plan: np.ndarray
    conditional_plan: np.ndarray
    teacher: np.ndarray
    row_mass: np.ndarray
    augmented_class_mass: np.ndarray
    radii: np.ndarray
    adversary: np.ndarray
    dual_row: np.ndarray
    dual_column: np.ndarray
    cost: np.ndarray
    difference: np.ndarray
    reject_only_rows: np.ndarray
    diagnostics: dict


def robust_transport(target, prototypes, radii, *, mass=.95,tau=.01,
                      target_mass=None,class_mass=None,reject_cost=.5,
                      max_iterations=500,tolerance=1e-8):
    target,prototypes,radii=map(lambda x:np.asarray(x,dtype=np.float64),(target,prototypes,radii))
    if target.ndim!=2 or prototypes.ndim!=2 or target.shape[1]!=prototypes.shape[1]:
        raise ValueError('Common signature-coordinate target/prototype matrices required')
    n,k=len(target),len(prototypes)
    if radii.shape!=(k,) or np.any(radii<0) or not all(np.isfinite(x).all() for x in [target,prototypes,radii]):
        raise ValueError('Finite nonnegative class uncertainty radii required')
    if not np.isfinite([mass,tau,reject_cost,tolerance]).all() or not 0<mass<=1 or tau<=0 or tolerance<=0:
        raise ValueError('Finite positive transported mass, entropy and tolerance required')
    if n<1 or k<1 or max_iterations<1:raise ValueError('Nonempty valid transport problem required')
    if np.max(np.linalg.norm(target,axis=1))>1+1e-10 or np.max(np.linalg.norm(prototypes,axis=1))>1+1e-10:
        raise ValueError('Fixed public unit-ball signature map required')
    a=np.full(n,1/n) if target_mass is None else np.asarray(target_mass,dtype=np.float64)
    q=np.full(k,1/k) if class_mass is None else np.asarray(class_mass,dtype=np.float64)
    if a.shape!=(n,) or q.shape!=(k,) or not np.isfinite(a).all() or not np.isfinite(q).all() or np.any(a<=0) or np.any(q<=0):
        raise ValueError('Strictly positive fixed row and class margins required')
    a=a/a.sum();q=q/q.sum();b=q if mass==1 else np.r_[mass*q,1-mass]
    cost,difference=squared_cost(target,prototypes)
    variable=cp.Variable((n,len(b)),nonneg=True)
    constraints=[cp.sum(variable,axis=1)==a,cp.sum(variable,axis=0)==b]
    objective=cp.sum(cp.multiply(cost,variable[:,:k]))+np.sum(b[:k]*radii**2)/4
    # Explicit epigraph exposes valid interior SOC duals when a column's
    # shared moment is zero. A normalized near-zero primal moment is unstable.
    epigraph=cp.Variable(k,nonneg=True);soc={}
    for c in range(k):
        if radii[c]>0:
            soc[c]=cp.SOC(epigraph[c],difference[:,c,:].T@variable[:,c])
            constraints.append(soc[c]);objective+=radii[c]/2*epigraph[c]
        else:constraints.append(epigraph[c]==0)
    if mass<1:objective+=reject_cost*cp.sum(variable[:,-1])
    objective-=tau*cp.sum(cp.entr(variable))
    problem=cp.Problem(cp.Minimize(objective),constraints)
    assert problem.is_dcp()
    problem.solve(solver='CLARABEL',max_iter=max_iterations,tol_gap_abs=tolerance,
                    tol_gap_rel=tolerance,tol_feas=tolerance,verbose=False)
    if problem.status not in ['optimal','optimal_inaccurate'] or variable.value is None:
        raise RuntimeError('Robust conic solver failure: '+str(problem.status))
    plan=np.asarray(variable.value)
    if np.any(plan<0) or not np.isfinite(plan).all():
        raise RuntimeError('Invalid nonnegative robust plan')
    residual=max(float(np.max(abs(plan.sum(1)-a))),float(np.max(abs(plan.sum(0)-b))))
    if residual>1e-6:raise RuntimeError('Robust solver marginal residual exceeds development tolerance')
    adversary=np.zeros_like(prototypes)
    for c,cone in soc.items():
        # SOC dual convention: scalar*epigraph + vector*shared moment is
        # subtracted from the Lagrangian; u/4 = -vector.
        proposed=-4*np.asarray(cone.dual_value[1]).reshape(-1)
        adversary[c]=proposed*min(1.,2*radii[c]/max(np.linalg.norm(proposed),1e-30))
    # A feasible adversary supplies a valid mathematical linearized-cost dual.
    # Float evaluation below is only a diagnostic, without outward rounding.
    linear=cost+radii[None,:]**2/4+np.einsum('kd,jkd->jk',adversary,difference)/4
    if mass<1:linear=np.c_[linear,np.full(n,reject_cost)]
    lam=np.asarray(constraints[0].dual_value)
    mu=np.asarray(constraints[1].dual_value).copy()
    # r^2 b /4 was a constant in the conic primal. Incorporating it as a
    # linear column cost requires the matching equality-potential shift.
    mu[:k]-=radii**2/4
    with np.errstate(over='ignore'):
        lower=-float(a@lam+b@mu)-tau*float(np.exp(-(linear+lam[:,None]+mu[None,:])/tau-1).sum())
    primal=robust_objective(plan,target,prototypes,radii,tau,reject_cost)
    gap=primal-lower
    if np.isfinite(gap) and gap < -1e-5:
        raise RuntimeError('Robust primal/dual diagnostics inconsistent')
    conditional=plan/a[:,None]
    teacher=conditional[:,:k].copy();class_row_sum=teacher.sum(1)
    reject_only=class_row_sum<=1e-15
    teacher/=np.maximum(class_row_sum[:,None],1e-15)
    # This explicit fallback is reported, never an all-zero pseudo-probability.
    # The pilot must preserve the mask and cannot call such a row certified.
    teacher[reject_only]=1/k
    diagnostics={'solver':'CLARABEL','execution_device':'CPU conic solver',
      'status':problem.status,'iterations':int(problem.solver_stats.num_iters),
      'solver_time_seconds':float(problem.solver_stats.solve_time),
      'primal_objective_float':primal,'dual_lower_float':lower,'gap_float':gap,
      'max_marginal_residual':residual,'tau':tau,'mass':mass,'reject_cost':reject_cost,
      'scope':'floating solver diagnostics, not directed-arithmetic optimality or semantic certificate',
      'reject_only_rows':np.flatnonzero(reject_only).tolist(),
      'reject_only_teacher_fallback':'uniform class probabilities, explicitly flagged',
      'uncertainty_set':'one Euclidean prototype displacement shared by every edge in its class column',
      'rho':1.,'signature':'bounded ordered public landmark/kernel relation coordinates'}
    stored_cost=cost if mass==1 else np.c_[cost,np.full(n,reject_cost)]
    return RobustTransportResult(plan,conditional,teacher,a,b,radii,adversary,lam,mu,
                                  stored_cost,difference,reject_only,diagnostics)
