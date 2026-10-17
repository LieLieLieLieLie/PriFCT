"""Exploratory target learning from one noisy joint sum/count message.

No source records or semantic target labels are accepted by these functions.
This is a research candidate, not a validated novelty or accuracy claim.
Constrained DP inference and graph SSL are established ideas; their use here
must be isolated against same-message controls.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.special import softmax, logsumexp, wrightomega
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize


def cone_simplex_projection(query, tol=1e-12):
    """Exact Euclidean projection onto ||sum_k||<=mass_k, sum mass_k=1.

    A single count-coordinate Lagrange multiplier separates K second-order
    cone projections. Public total mass is one; no clean class counts enter.
    """
    raw=np.asarray(query,float);s=raw[:,:-1];t=raw[:,-1]
    radius=np.linalg.norm(s,axis=1)
    def project(lam):
        u=t-lam
        mass=np.where(radius<=u,u,np.where(radius<=-u,0.,(radius+u)/2))
        factor=np.minimum(1.,mass/np.maximum(radius,1e-30))
        return s*factor[:,None],mass
    lo,hi=-1.,1.
    while project(lo)[1].sum()<1:lo*=2
    while project(hi)[1].sum()>1:hi*=2
    for _ in range(100):
        mid=(lo+hi)/2;features,mass=project(mid)
        if abs(mass.sum()-1)<tol:break
        if mass.sum()>1:lo=mid
        else:hi=mid
    out=np.c_[features,mass]
    assert abs(mass.sum()-1)<1e-9
    assert np.all(np.linalg.norm(features,axis=1)<=mass+1e-10)
    return out


@dataclass
class TargetGraph:
    x: np.ndarray
    neighbours: object
    transition: object
    laplacian: object
    bandwidth: float

    def interpolate(self,xte,prob):
        distance,index=self.neighbours.kneighbors(xte,n_neighbors=10)
        logw=-distance**2/self.bandwidth
        weight=softmax(logw,axis=1)
        return (weight[:,:,None]*prob[index]).sum(1)


def target_graph(x, bandwidth=.1):
    x=np.asarray(x,float)
    model=NearestNeighbors(n_neighbors=11).fit(x)
    distance,index=model.kneighbors(x)
    distance,index=distance[:,1:],index[:,1:]
    values=softmax(-distance**2/bandwidth,axis=1)
    w=sparse.csr_matrix((values.ravel(),(np.repeat(np.arange(len(x)),10),index.ravel())),shape=(len(x),len(x)))
    sym=(w+w.T)*.5
    lap=sparse.diags(np.asarray(sym.sum(1)).ravel())-sym
    return TargetGraph(x,model,w,lap,bandwidth)


def uncertainty_teacher(x, query, sigma_query, public_source, public_target, n_source,
                        projected=True, use_uncertainty=True):
    """Diagonal predictive scores; linearised quotient variance is heuristic.

    For rare/noisy counts this is NOT an exact Gaussian mean posterior or
    semantic confidence guarantee. Public neighbour spread supplies a data
    scale independent of private source statistics.
    """
    q=cone_simplex_projection(query) if projected else np.asarray(query,float)
    mass=np.maximum(q[:,-1],1/n_source)
    means=q[:,:-1]/mass[:,None]
    means*=np.minimum(1.,1/np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-30))
    if not use_uncertainty:
        return softmax(-cdist(x,means)**2/.1,axis=1),{'mean':means,'mass':mass,'variance':None}
    residual=np.mean((public_source-public_target)**2,axis=0)
    nnb=NearestNeighbors(n_neighbors=min(11,len(public_target))).fit(public_target)
    _,index=nnb.kneighbors(public_target)
    local=np.mean((public_target[:,None,:]-public_target[index[:,1:]])**2,axis=(0,1))*.5
    # Count and sum coordinate noise are independent in the released query.
    # Delta-method variance includes the noisy denominator contribution.
    variance=(residual+local)[None,:]+(sigma_query/mass[:,None])**2*(1+means**2)
    variance=np.maximum(variance,1e-5)
    logp=-.5*np.sum((x[:,None,:]-means[None,:,:])**2/variance[None,:,:]
                   +np.log(variance)[None,:,:],axis=2)
    return softmax(logp,axis=1),{'mean':means,'mass':mass,'variance':variance,
                 'public_residual':residual,'public_local_variance':local}


def simplex_entropy_prox(v,logprior,scale):
    """Euclidean prox of scale*KL(p,prior) on each row simplex.

    p + scale*log(p) = v-lambda+scale*log(prior)-scale.
    Wright omega avoids exponent overflow; safeguarded Newton solves lambda.
    """
    if scale<=0:raise ValueError('Positive entropy weight required')
    c=v+scale*logprior-scale
    maximum=c.max(1,keepdims=True);k=v.shape[1]
    lo=maximum-1.;hi=maximum-1/k+scale*np.log(k)
    lam=(lo+hi)*.5
    for _ in range(50):
        p=scale*wrightomega((c-lam)/scale-np.log(scale))
        error=p.sum(1,keepdims=True)-1
        if np.max(np.abs(error))<2e-12:break
        lo=np.where(error>0,lam,lo);hi=np.where(error<=0,lam,hi)
        proposal=lam+error/np.maximum((p/(p+scale)).sum(1,keepdims=True),1e-30)
        lam=np.where((proposal>lo)&(proposal<hi),proposal,(lo+hi)*.5)
    return p/p.sum(1,keepdims=True)


def solve_moments(x,query,sigma_query,public_residual,n_source,prior,graph,
                  graph_weight=1.,moment_weight=1.,count_weight=1.,anchor_weight=1.,
                  tolerance=2e-6,max_iter=4000):
    """Convex moment/graph/KL inference by Euclidean entropy-proximal FISTA.

    Same-population joint moments are only a modelling assumption. Noise
    scaling includes Gaussian variance, public cross-view residual and a
    conservative bounded-variable sampling allowance. No hard class-count
    equality or clean counts is imposed.
    """
    x=np.asarray(x,float);n,r=x.shape
    a=np.c_[x,np.ones(n)]
    q=np.asarray(query,float)
    # Variance for a coordinate of an average of bounded [-1,1] variables;
    # public residual is an approximation to view discrepancy, not a theorem.
    variance=np.r_[sigma_query**2+public_residual/max(n_source,1)+(1/n_source+1/n)/r,
                    sigma_query**2+.25*(1/n_source+1/n)]
    weights=np.r_[np.full(r,moment_weight),count_weight]/variance
    prior=np.maximum(prior,1e-12);prior/=prior.sum(1,keepdims=True)
    lp=np.log(prior);p=prior.copy()
    # n-scaled smooth Hessian upper bound, used for a safe proximal step.
    aw=a*np.sqrt(weights)[None,:]
    lipschitz=np.linalg.norm(aw,2)**2/n
    maxdegree=np.max(graph.laplacian.diagonal())
    step=1/max(1.,lipschitz+2*graph_weight*maxdegree)
    def value(prob):
        residual=prob.T@a/n-q
        kl=np.sum(prob*(np.log(np.maximum(prob,1e-300))-lp))/n
        return float(.5*np.sum(residual**2*weights)+graph_weight*.5*np.sum(prob*(graph.laplacian@prob))/n+anchor_weight*kl)
    history=[value(p)];converged=False;extrapolated=p.copy();acceleration=1.
    for iteration in range(max_iter):
        residual=extrapolated.T@a/n-q
        grad=a@(residual*weights).T+graph_weight*(graph.laplacian@extrapolated)
        pnew=simplex_entropy_prox(extrapolated-step*grad,lp,step*anchor_weight)
        if value(pnew)>value(p)+1e-12:
            acceleration=1.;extrapolated=p.copy()
            grad=a@((p.T@a/n-q)*weights).T+graph_weight*(graph.laplacian@p)
            pnew=simplex_entropy_prox(p-step*grad,lp,step*anchor_weight)
        if iteration%25==0 or iteration==max_iter-1:
            current=value(pnew);history.append(current)
            if current>history[-2]+1e-8:raise RuntimeError('Moment objective increased')
            # Stationarity after removing row-simplex Lagrange multipliers.
            resnew=pnew.T@a/n-q
            g=a@(resnew*weights).T+graph_weight*(graph.laplacian@pnew)
            g+=anchor_weight*(np.log(np.maximum(pnew,1e-300))-lp+1)
            gap=float(np.sum(pnew*g)-np.min(g,axis=1).sum())/n
            if gap<tolerance:
                p=pnew;converged=True;break
        next_acceleration=(1+np.sqrt(1+4*acceleration**2))/2
        extrapolated=pnew+(acceleration-1)/next_acceleration*(pnew-p)
        acceleration=next_acceleration;p=pnew
    # The conservative global step can be very slow when count curvature is
    # large. L-BFGS in full-support logits accelerates the SAME objective;
    # convergence is still checked by its convex simplex stationarity gap.
    if not converged:
        def objective_logits(z):
            logp=z.reshape(p.shape);logp-=logsumexp(logp,axis=1,keepdims=True)
            prob=np.exp(logp);residual=prob.T@a/n-q
            g=a@(residual*weights).T+graph_weight*(graph.laplacian@prob)
            g+=anchor_weight*(logp-lp+1)
            gradient=prob*(g-np.sum(prob*g,axis=1,keepdims=True))/n
            return value(prob),gradient.ravel()
        refined=minimize(objective_logits,np.log(np.maximum(p,1e-200)).ravel(),
            jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-13,'gtol':1e-10,'maxls':50})
        p=softmax(refined.x.reshape(p.shape),axis=1)
        g=a@((p.T@a/n-q)*weights).T+graph_weight*(graph.laplacian@p)
        g+=anchor_weight*(np.log(np.maximum(p,1e-300))-lp+1)
        gap=float(np.sum(p*g)-np.min(g,axis=1).sum())/n
        converged=gap<tolerance
        history.append(value(p))
        iteration+=refined.nit
    return p,{'iterations':iteration+1,'converged':converged,'simplex_gap':gap,
             'objective':history[-1],'objective_history':history,'step':step,
             'moment_residual_norm':float(np.linalg.norm(p.T@a/n-q)),
             'variance':variance,'graph_weight':graph_weight,'moment_weight':moment_weight,
             'count_weight':count_weight,'anchor_weight':anchor_weight}


def candidate_predict(xtr,xte,joint,public_source,public_target,variant='full'):
    graph=target_graph(xtr)
    teacher,meta=uncertainty_teacher(xtr,joint.raw,joint.sigma_query,
        public_source,public_target,joint.n_public,
        projected=variant!='no_projection',use_uncertainty=variant!='no_uncertainty')
    if variant=='teacher':return graph.interpolate(xte,teacher),meta
    query=cone_simplex_projection(joint.raw) if variant!='no_projection' else joint.raw
    probabilities,diagnostics=solve_moments(xtr,query,joint.sigma_query,
        np.mean((public_source-public_target)**2,axis=0),joint.n_public,teacher,graph,
        graph_weight=0. if variant=='no_graph' else 1.,
        moment_weight=0. if variant=='no_moments' else 1.,
        count_weight=0. if variant in ['no_moments','no_counts'] else 1.)
    return graph.interpolate(xte,probabilities),{**meta,**diagnostics}
