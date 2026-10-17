"""Conservative numerical audit of fixed-margin entropic transport.

The audit certifies distance from a returned floating plan/conditional scores
to the EXACT optimizer of the stated, fixed public margins. Feasibility alone
is never used as an optimality certificate. Rational mass accounting and
outward Decimal objective/dual envelopes give a conservative upper gap.

No plan is silently changed. An exact feasible repair is used existentially:
its distance is bounded by the transport rounding lemma, and entropy
continuity bounds its objective without constructing huge rational matrices.
Unsupported/overflowing audits return radius 2 and must cause abstention.
"""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import numpy as np


class _Interval:
    def __init__(self, lo, hi, precision):
        self.lo, self.hi, self.precision = Decimal(lo), Decimal(hi), precision
        if self.lo > self.hi:
            raise ArithmeticError('Inverted decimal enclosure')

    @classmethod
    def rational(cls, value, precision):
        value = Fraction(value)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = precision, ROUND_FLOOR
            lo = Decimal(value.numerator) / Decimal(value.denominator)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = precision, ROUND_CEILING
            hi = Decimal(value.numerator) / Decimal(value.denominator)
        return cls(lo, hi, precision)

    def _coerce(self, other):
        return other if isinstance(other, _Interval) else self.rational(other, self.precision)

    def __add__(self, other):
        other = self._coerce(other)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = self.precision, ROUND_FLOOR
            lo = self.lo + other.lo
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = self.precision, ROUND_CEILING
            hi = self.hi + other.hi
        return _Interval(lo, hi, self.precision)

    def __sub__(self, other):
        other = self._coerce(other)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = self.precision, ROUND_FLOOR
            lo = self.lo - other.hi
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = self.precision, ROUND_CEILING
            hi = self.hi - other.lo
        return _Interval(lo, hi, self.precision)

    def __mul__(self, other):
        other = self._coerce(other)
        pairs = [(a,b) for a in [self.lo,self.hi] for b in [other.lo,other.hi]]
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = self.precision, ROUND_FLOOR
            lo = min(a*b for a,b in pairs)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = self.precision, ROUND_CEILING
            hi = max(a*b for a,b in pairs)
        return _Interval(lo, hi, self.precision)

    def __truediv__(self, other):
        other = self._coerce(other)
        if other.lo <= 0:
            raise ArithmeticError('Only positive denominator intervals supported')
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = self.precision, ROUND_FLOOR
            lo = Decimal(1) / other.hi
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = self.precision, ROUND_CEILING
            hi = Decimal(1) / other.lo
        return self * _Interval(lo, hi, self.precision)

    def log(self):
        if self.lo <= 0:
            raise ArithmeticError('Logarithm requires strictly positive lower endpoint')
        # Decimal ln/exp/sqrt are correctly round-to-nearest, regardless of
        # requested rounding mode. Adjacent representables give enclosures.
        with localcontext() as ctx:
            ctx.prec = self.precision
            lo = self.lo.ln().next_minus()
            hi = self.hi.ln().next_plus()
        return _Interval(lo, hi, self.precision)

    def exp(self):
        with localcontext() as ctx:
            ctx.prec = self.precision
            lo = self.lo.exp().next_minus()
            hi = self.hi.exp().next_plus()
        return _Interval(max(Decimal(0),lo), hi, self.precision)

    def sqrt_upper(self):
        if self.hi < 0:
            raise ArithmeticError('Negative squared radius')
        with localcontext() as ctx:
            ctx.prec = self.precision
            return self.hi.sqrt().next_plus()


def _fraction(value):
    return Fraction.from_float(float(value))


def _float_upper(value):
    # Exact Fraction/Decimal -> closest float followed by outward nextafter.
    result = float(value)
    return float(np.nextafter(result, np.inf)) if np.isfinite(result) else result


def _probability_margins(count, values):
    if values is None:
        return [Fraction(1,count)]*count
    a = np.asarray(values,float)
    if a.shape != (count,) or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('Positive finite predeclared public margin weights required')
    exact = [_fraction(v) for v in a]
    total = sum(exact)
    return [v/total for v in exact]


def _unverified(reason, n):
    return {'status':'numerical-unverified', 'reason':str(reason),
            'normalized_plan_l1_radius':2., 'returned_plan_l1_radius':float('inf'),
            'row_score_radius':np.ones(n), 'objective_gap_upper':None,
            'reference':'exact fixed-margin entropic optimizer; no numerical guarantee available',
            'must_abstain':True}


def audit_fixed_transport(cost, plan, mass=.95, tau=.01,
                          target_mass=None, class_mass=None,
                          conditional_plan=None, dual_potentials=None,
                          precision=70, max_cells=20000):
    """Numerical radii for class-cost matrix and reject-augmented returned plan.

    Exact public margins are normalized rational values of supplied weights;
    defaults are exactly 1/J, m/K, and 1-m. Mass/tau are the exact real values
    represented by their supplied floats. Positive mass=1 drops the zero
    reject column. Any computed conditional scores can be supplied, and their
    deviation from exact row-normalization is included in row radii.

    dual_potentials=(f,g) may be any finite vectors for the ACTIVE support and
    entropy convention tau sum P log P. Absent potentials are obtained from
    a row+column least-squares decomposition of C+tau(1+log P); they need not
    be optimal to give a VALID lower bound. Zero entries use only a proposal
    log floor; the actual primal entropy always uses exact 0log0=0.
    """
    cost, plan = np.asarray(cost,float), np.asarray(plan,float)
    n = len(cost) if cost.ndim else 0
    try:
        if cost.ndim!=2 or not n or not cost.shape[1]:
            raise ValueError('Nonempty class cost matrix required')
        k = cost.shape[1]
        if not np.isfinite(cost).all() or not np.isfinite(plan).all() or np.any(plan<0):
            raise ValueError('Finite cost and nonnegative finite plan required')
        if not np.isfinite(tau) or tau<=0 or not np.isfinite(mass) or not 0<mass<=1:
            raise ValueError('tau>0 and mass in (0,1] required')
        if precision<40 or not isinstance(precision,int):
            raise ValueError('At least 40-digit directed precision required')
        if plan.shape not in [(n,k),(n,k+1)]:
            raise ValueError('Class or reject-augmented plan shape required')
        a = _probability_margins(n,target_mass)
        q = _probability_margins(k,class_mass)
        exact_mass, exact_tau = _fraction(mass), _fraction(tau)
        b = [exact_mass*v for v in q]
        if mass<1:
            if plan.shape[1]!=k+1:
                raise ValueError('Reject column required when mass<1')
            b.append(1-exact_mass)
            augmented = np.c_[cost,np.full(n,.5)]
            active_plan = plan
        else:
            if plan.shape[1]==k+1 and np.any(plan[:,-1]!=0):
                raise ValueError('Zero-margin reject column contains nonzero mass')
            augmented, active_plan = cost, plan[:,:k]
        h = augmented.shape[1]
        cells = n*h
        if cells>max_cells:
            raise ValueError('Audit cell cap exceeded; no numerical claim made')
        raw = [[_fraction(v) for v in row] for row in active_plan]
        total = sum(sum(row) for row in raw)
        if total<=0:
            raise ValueError('Positive total plan mass required')
        p = [[v/total for v in row] for row in raw]
        rows = [sum(row) for row in p]
        cols = [sum(p[i][j] for i in range(n)) for j in range(h)]
        if any(v<=0 for v in rows):
            raise ValueError('Positive returned row masses required')
        residual_row = sum(abs(rows[i]-a[i]) for i in range(n))
        residual_col = sum(abs(cols[j]-b[j]) for j in range(h))
        # Transport rounding constructs Q feasible on the SAME support with
        # ||Q-p||1 <= 2(||p1-a||1+||p^T1-b||1). Both are probabilities.
        repair_radius = min(Fraction(2),2*(residual_row+residual_col))
        if dual_potentials is None:
            proposal = augmented + tau*(1+np.log(np.maximum(active_plan,1e-300)))
            f = proposal.mean(axis=1)
            g = proposal.mean(axis=0)-proposal.mean()
        else:
            f,g = [np.asarray(v,float) for v in dual_potentials]
        if f.shape!=(n,) or g.shape!=(h,) or not np.isfinite(f).all() or not np.isfinite(g).all():
            raise ValueError('Finite dual vectors on the active support required')
        f,g = list(map(_fraction,f)),list(map(_fraction,g))
        interval = lambda v:_Interval.rational(v,precision)
        tau_i = interval(exact_tau)
        primal, dual_exponential = interval(0),interval(0)
        for i in range(n):
            for j in range(h):
                c = interval(_fraction(augmented[i,j]))
                if p[i][j]:
                    pi = interval(p[i][j])
                    primal = primal + c*pi + tau_i*pi*pi.log()
                exponent = (interval(f[i]+g[j])-c)/tau_i-1
                dual_exponential = dual_exponential + exponent.exp()
        dual = interval(sum(a[i]*f[i] for i in range(n))+sum(b[j]*g[j] for j in range(h)))-tau_i*dual_exponential
        # Entropy continuity for two M-cell probabilities separated by e:
        # |sum q logq - sum p logp| <= min(log M,e log(M/e)+e).
        if repair_radius and cells>1:
            e = interval(repair_radius)
            coarse = e*(interval(cells)/e).log()+e
            entropy_upper = min(interval(cells).log().hi,coarse.hi)
        else:
            entropy_upper = Decimal(0)
        cost_norm = max(abs(_fraction(v)) for v in augmented.ravel())
        continuity = interval(cost_norm*repair_radius)+tau_i*_Interval(0,entropy_upper,precision)
        primal_repair = primal + continuity
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = precision, ROUND_CEILING
            gap_upper = primal_repair.hi-dual.lo
        # A negative upper gap contradicts a valid primal/dual enclosure. It
        # must NEVER be truncated to zero and passed off as a certificate.
        if not gap_upper.is_finite() or gap_upper<0:
            raise ArithmeticError('No trustworthy nonnegative objective-gap upper bound')
        radius_interval = interval(2)*_Interval(0,gap_upper,precision)/tau_i
        eta = radius_interval.sqrt_upper()
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = precision, ROUND_CEILING
            normalized_radius_decimal = interval(repair_radius).hi+eta
        normalized_radius = min(2.,_float_upper(normalized_radius_decimal))
        score_error = np.zeros(n)
        if conditional_plan is not None:
            observed = np.asarray(conditional_plan,float)
            if observed.shape!=plan.shape or not np.isfinite(observed).all():
                raise ValueError('Finite conditional scores matching returned plan required')
            for i in range(n):
                diffs = [abs(_fraction(observed[i,j])-raw[i][j]/sum(raw[i])) for j in range(h)]
                if observed.shape[1]>h:
                    diffs.append(abs(_fraction(observed[i,-1])))
                score_error[i] = _float_upper(max(diffs))
        numerical_row = np.array([min(1.,_float_upper((_fraction(normalized_radius)+abs(rows[i]-a[i]))/(2*a[i]))) for i in range(n)])
        numerical_row = np.nextafter(numerical_row+score_error,np.inf)
        return {'status':'verified-directed-gap', 'must_abstain':bool(np.all(numerical_row>=.5)),
                'normalized_plan_l1_radius':normalized_radius,
                'returned_plan_l1_radius':_float_upper(abs(total-1)+_fraction(normalized_radius)),
                'row_score_radius':numerical_row,
                'conditional_representation_error':score_error,
                'objective_gap_upper':_float_upper(gap_upper),
                'objective_gap_upper_decimal':str(gap_upper),
                'repair_l1_radius':_float_upper(repair_radius),
                'normalized_row_l1_residual':_float_upper(residual_row),
                'normalized_column_l1_residual':_float_upper(residual_col),
                'primal_repair_upper_decimal':str(primal_repair.hi),
                'dual_lower_decimal':str(dual.lo),
                'precision':precision, 'active_cells':cells,
                'reference':'exact fixed-margin entropic optimizer on rationally normalized public margins',
                'margin_convention':'exact float-valued mass/tau; public weights rationally normalized; total mass exactly one',
                'proof':'exact-mass normalization + feasible repair lemma + entropy continuity + outward primal/dual + Pinsker'}
    except (ValueError,ArithmeticError,OverflowError) as error:
        return _unverified(error,n)


def combined_score_radius(noise_plan_l1_radius, exact_row_masses,
                          private_numerical, clean_numerical=None):
    """Score radius versus exact clean optimizer, or supplied computed clean.

    noise_plan_l1_radius already compares TWO EXACT fixed-margin optimizers.
    private numerical error is always added. clean numerical error is added
    only when the reference is the computed clean implementation. A failed
    numerical audit returns infinity and must cause full abstention.
    """
    a = np.asarray(exact_row_masses,float)
    if np.any(a<=0) or not np.isfinite(a).all() or not np.isfinite(noise_plan_l1_radius) or noise_plan_l1_radius<0:
        return np.full(len(a),np.inf)
    # Same exact rational normalization convention as the audited optimizer.
    rational_a = _probability_margins(len(a),exact_row_masses)
    for audit in [private_numerical,clean_numerical]:
        if audit is not None and audit.get('status')!='verified-directed-gap':
            return np.full(len(a),np.inf)
        if audit is not None:
            numerical = np.asarray(audit.get('row_score_radius'),float)
            if numerical.shape!=a.shape or not np.isfinite(numerical).all() or np.any(numerical<0):
                return np.full(len(a),np.inf)
    noise_radius = np.array([min(1.,_float_upper(_fraction(noise_plan_l1_radius)/(2*v))) for v in rational_a])
    result = np.nextafter(noise_radius + private_numerical['row_score_radius'],np.inf)
    if clean_numerical is not None:
        result = np.nextafter(result + clean_numerical['row_score_radius'],np.inf)
    return result
