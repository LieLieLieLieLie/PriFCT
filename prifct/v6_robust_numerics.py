"""Directed objective-gap audit for class-shared prototype uncertainty.

The exact objective uses the stored floating point target/prototype/radius
vectors as real numbers and rationally normalized public margins. Upstream
feature calculations and the Gaussian sampler are outside this audit.
It does not borrow the linear-cost transport lower bound or certify semantics.
"""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
import numpy as np
from prifct.transport_numerics import (_Interval, _fraction, _float_upper,
    _probability_margins)


def _norm(values, precision):
    if all(v.lo==v.hi==0 for v in values):return _Interval.rational(0,precision)
    total=_Interval.rational(0,precision)
    for value in values:
        product=value*value
        lo=Decimal(0) if value.lo<=0<=value.hi else product.lo
        total=total+_Interval(max(Decimal(0),lo),product.hi,precision)
    with localcontext() as ctx:
        ctx.prec=precision
        lo=max(Decimal(0),total.lo).sqrt().next_minus()
        hi=max(Decimal(0),total.hi).sqrt().next_plus()
    return _Interval(max(Decimal(0),lo),hi,precision)


def _failed(reason,n):
    return {'status':'numerical-unverified','reason':str(reason),
        'normalized_plan_l1_radius':2.,'returned_plan_l1_radius':float('inf'),
        'row_score_radius':np.ones(n),'objective_gap_upper':None,'must_abstain':True,
        'reference':'exact class-shared robust optimizer; no numerical guarantee available'}


def audit_robust_transport(target,prototypes,radii,plan,adversary,dual_row,dual_column,
                          *,mass=.95,tau=.01,target_mass=None,class_mass=None,
                          reject_cost=.5,conditional_plan=None,precision=70,max_cells=20000):
    """A valid gap using ANY inward-feasible class adversary and finite dual.

    Row/column duals use CVXPY equality-multiplier convention, with the column
    shift for the r^2/4 term already applied by the solver. Infeasible primal
    margins use an existential feasible repair plus objective continuity.
    Returned plans and predictions are never silently replaced.
    """
    target=np.asarray(target,float);prototypes=np.asarray(prototypes,float)
    n=len(target) if target.ndim else 0
    try:
        radii=np.asarray(radii,float);plan=np.asarray(plan,float);u=np.asarray(adversary,float)
        lam=np.asarray(dual_row,float);mu=np.asarray(dual_column,float)
        if target.ndim!=2 or prototypes.ndim!=2 or not n or not len(prototypes) or target.shape[1]!=prototypes.shape[1]:raise ValueError('Nonempty shared-coordinate matrices required')
        k,d=prototypes.shape
        if radii.shape!=(k,) or np.any(radii<0) or u.shape!=(k,d):raise ValueError('Valid class radii and adversary shape required')
        if not isinstance(precision,int) or precision<40:raise ValueError('At least forty directed digits required')
        if not np.isfinite(mass) or not 0<mass<=1 or not np.isfinite(tau) or tau<=0 or not np.isfinite(reject_cost):raise ValueError('Finite mass/tau/reject cost required')
        arrays=[target,prototypes,radii,plan,u,lam,mu]
        if not all(np.isfinite(x).all() for x in arrays) or np.any(plan<0):raise ValueError('Finite inputs and nonnegative plan required')
        h=k+(mass<1)
        if plan.shape not in [(n,k),(n,k+1)]:raise ValueError('Plan shape mismatch')
        if mass<1 and plan.shape[1]!=k+1:raise ValueError('Reject column required')
        if mass==1 and plan.shape[1]==k+1 and np.any(plan[:,-1]):raise ValueError('Zero-margin reject column has positive mass')
        if lam.shape!=(n,) or mu.shape!=(h,):raise ValueError('Finite equality duals on active support required')
        cells=n*h
        if cells>max_cells:raise ValueError('Audit cell budget exceeded')
        a=_probability_margins(n,target_mass);q=_probability_margins(k,class_mass)
        m,t,rc=map(_fraction,[mass,tau,reject_cost]);b=[m*v for v in q]
        if mass<1:b.append(1-m)
        raw=[[_fraction(v) for v in row[:h]] for row in plan]
        total=sum(map(sum,raw))
        if total<=0:raise ValueError('Positive returned mass required')
        p=[[v/total for v in row] for row in raw]
        rows=list(map(sum,p));cols=[sum(p[i][j] for i in range(n)) for j in range(h)]
        if any(v<=0 for v in rows):raise ValueError('Positive returned row masses required')
        er=sum(abs(rows[i]-a[i]) for i in range(n));ec=sum(abs(cols[j]-b[j]) for j in range(h))
        repair=min(Fraction(2),2*(er+ec))
        interval=lambda v:_Interval.rational(v,precision)
        r=list(map(_fraction,radii));z=[[ _fraction(v) for v in row] for row in target]
        source=[[ _fraction(v) for v in row] for row in prototypes]
        delta=[[[source[c][l]-z[i][l] for l in range(d)] for c in range(k)] for i in range(n)]
        cost=[[sum(v*v for v in delta[i][c])/4+r[c]*r[c]/4 for c in range(k)] for i in range(n)]
        if mass<1:
            for row in cost:row.append(rc)
        # Prove ball feasibility exactly; use an explicitly inward Decimal
        # factor when the float solver vector lies just outside the ball.
        uf=[list(map(_fraction,row)) for row in u];repaired_adversaries=[]
        for c in range(k):
            norm2=sum(v*v for v in uf[c]);bound=4*r[c]*r[c]
            if norm2>bound:
                if not r[c]:uf[c]=[Fraction(0)]*d
                else:
                    upper=_norm([interval(v) for v in uf[c]],precision).hi
                    ratio=(interval(2*r[c])/_Interval(upper,upper,precision)).lo
                    uf[c]=[v*Fraction(ratio) for v in uf[c]]
                    if sum(v*v for v in uf[c])>bound:raise ArithmeticError('Inward adversary projection not verified')
                repaired_adversaries.append(c)
        f=[-_fraction(v) for v in lam];g=[-_fraction(v) for v in mu]
        ti=interval(t);primal=interval(0);exponential=interval(0)
        for i in range(n):
            for c in range(h):
                ci=interval(cost[i][c]);pi=interval(p[i][c])
                if p[i][c]:primal=primal+ci*pi+ti*pi*pi.log()
                linear=cost[i][c]
                if c<k:linear+=sum(uf[c][l]*delta[i][c][l] for l in range(d))/4
                exponential=exponential+((interval(f[i]+g[c]-linear)/ti)-1).exp()
        for c in range(k):
            moment=[interval(sum(p[i][c]*delta[i][c][l] for i in range(n))) for l in range(d)]
            primal=primal+interval(r[c]/2)*_norm(moment,precision)
        lower=interval(sum(a[i]*f[i] for i in range(n))+sum(b[c]*g[c] for c in range(h)))-ti*exponential
        if repair and cells>1:
            ei=interval(repair);entropy=min(interval(cells).log().hi,(ei*(interval(cells)/ei).log()+ei).hi)
        else:entropy=Decimal(0)
        # General stored-vector bound: each class moment coefficient has norm
        # at most max_j||delta_jk||. No unverified unit-ball assertion needed.
        lip=Decimal(0)
        for i in range(n):
            for c in range(k):
                value=interval(r[c]/2)*_norm([interval(v) for v in delta[i][c]],precision)
                lip=max(lip,value.hi)
        cost_norm=max(abs(v) for row in cost for v in row)
        repaired=primal+interval(cost_norm*repair)+_Interval(0,lip,precision)*interval(repair)+ti*_Interval(0,entropy,precision)
        with localcontext() as ctx:
            ctx.prec=precision;ctx.rounding=ROUND_CEILING
            gap=repaired.hi-lower.lo
        if not gap.is_finite() or gap<0:raise ArithmeticError('Nonnegative directed gap not established')
        optimum_radius=(interval(2)*_Interval(0,gap,precision)/ti).sqrt_upper()
        with localcontext() as ctx:
            ctx.prec=precision;ctx.rounding=ROUND_CEILING
            radius=min(Decimal(2),interval(repair).hi+optimum_radius)
        radius_float=min(2.,_float_upper(radius));representation=np.zeros(n)
        if conditional_plan is not None:
            observed=np.asarray(conditional_plan,float)
            if observed.shape!=plan.shape or not np.isfinite(observed).all():raise ValueError('Valid conditional scores required')
            for i in range(n):
                errors=[abs(_fraction(observed[i,c])-raw[i][c]/sum(raw[i])) for c in range(h)]
                if plan.shape[1]>h:errors.append(abs(_fraction(observed[i,-1])))
                representation[i]=_float_upper(max(errors))
        score=np.array([min(1.,_float_upper((_fraction(radius_float)+abs(rows[i]-a[i]))/(2*a[i]))) for i in range(n)])
        score=np.nextafter(score+representation,np.inf)
        return {'status':'verified-directed-robust-gap','must_abstain':bool(np.all(score>=.5)),
            'objective_gap_upper':_float_upper(gap),'objective_gap_upper_decimal':str(gap),
            'primal_repair_upper_decimal':str(repaired.hi),'dual_lower_decimal':str(lower.lo),
            'normalized_plan_l1_radius':radius_float,'returned_plan_l1_radius':_float_upper(abs(total-1)+_fraction(radius_float)),
            'row_score_radius':score,'conditional_representation_error':representation,
            'repair_l1_radius':_float_upper(repair),'normalized_row_l1_residual':_float_upper(er),
            'normalized_column_l1_residual':_float_upper(ec),'inward_projected_adversary_classes':repaired_adversaries,
            'precision':precision,'active_cells':cells,
            'reference':'unique exact class-shared robust optimizer of stored real-valued target/prototype/radius vectors and rational public margins',
            'scope':'numerical optimizer proximity only; not semantic accuracy or upstream machine arithmetic',
            'proof':'shared norm primal + feasible adversary entropic dual + rational margin repair + convex entropy/Pinsker'}
    except (ValueError,ArithmeticError,OverflowError) as error:
        return _failed(error,n)
