"""Private prototype transport: stable convex plan, rectangular reject option.

No evaluator labels or source records enter TargetProblem. Sparse target
anchors are explicit training information; a zero-anchor mode is supported.
"""
from dataclasses import dataclass
import sys
import numpy as np
try:
    import torch
except Exception:
    for module_name in [name for name in sys.modules if name == 'torch' or name.startswith('torch.')]:
        sys.modules.pop(module_name, None)
    torch = None
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from scipy.stats import beta
import ot
from core.preprocessing import fixed_record_map
from revision.joint_release import release_joint,JointMessage
from revision.prototype_certificate import prototype_radii
from revision.protocol import signatures

DEVICE='cuda' if torch is not None and torch.cuda.is_available() else 'cpu'

def gpu_sqdist(x,y):
    if torch is None:
        x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
        return np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T,0.)
    a=torch.as_tensor(x,device=DEVICE,dtype=torch.float64)
    b=torch.as_tensor(y,device=DEVICE,dtype=torch.float64)
    return torch.clamp((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,min=0).cpu().numpy()

def kernel_prob(x,xa,ya,k,width,ridge=.01):
    if not len(xa):return np.full((len(x),k),1/k)
    gram=np.exp(-gpu_sqdist(xa,xa)/width)
    coef=np.linalg.solve(gram+ridge*np.eye(len(xa)),np.eye(k)[ya])
    score=np.exp(-gpu_sqdist(x,xa)/width)@coef
    return softmax(score*4,axis=1)

def fixed_transport(cost,mass=.9,tau=.05,target_mass=None,class_mass=None):
    """Balanced augmentation of partial mass, with exact positive row margins."""
    n,k=cost.shape;a=np.full(n,1/n) if target_mass is None else np.asarray(target_mass,float)
    q=np.full(k,1/k) if class_mass is None else np.asarray(class_mass,float)
    a=a/a.sum();q=q/q.sum()
    full=mass==1.
    b=q if full else np.r_[mass*q,1-mass]
    augmented=cost if full else np.c_[cost,np.full(n,.5)]
    plan=ot.sinkhorn(a,b,augmented,tau,method='sinkhorn_log',numItermax=3000,stopThr=1e-10,warn=False)
    if np.max(abs(plan.sum(1)-a))>1e-7 or np.max(abs(plan.sum(0)-b))>1e-7:
        # Eliminate the row dual potentials and optimise the SAME smooth
        # convex entropic dual; this is not an unregularised transport fallback.
        def objective(v):
            potential=np.r_[v,0.];z=(potential[None,:]-augmented)/tau
            probability=softmax(z,axis=1)
            value=tau*np.sum(a*logsumexp(z,axis=1))-b@potential
            grad=(a[:,None]*probability).sum(0)-b
            return value,grad[:-1]
        fit=minimize(objective,np.zeros(len(b)-1),jac=True,method='L-BFGS-B',
                     options={'ftol':1e-15,'gtol':1e-11,'maxiter':10000,'maxls':100})
        plan=a[:,None]*softmax((np.r_[fit.x,0.][None,:]-augmented)/tau,axis=1)
    assert np.isfinite(plan).all() and np.max(abs(plan.sum(1)-a))<1e-6
    assert np.max(abs(plan.sum(0)-b))<1e-6
    if full:plan=np.c_[plan,np.zeros(n)]
    return plan,plan/a[:,None]

@dataclass
class TargetProblem:
    x:np.ndarray
    xtest:np.ndarray
    centers:np.ndarray
    labels:np.ndarray
    responsibilities:np.ndarray
    test_responsibilities:np.ndarray
    anchor_indices:np.ndarray
    anchor_labels:np.ndarray
    k:int
    comparable:bool
    transition:object
    test_transition:object
    anchor_prob:np.ndarray
    test_anchor_prob:np.ndarray
    center_anchor_prob:np.ndarray
    width:float
    target_mass:np.ndarray
    class_mass:np.ndarray

    @classmethod
    def build(cls,raw,rawtest,k,indices,labels,seed,comparable=False,clusters=1,bandwidth=1.,prepared=False):
        if prepared:
            x=np.asarray(raw,float);xt=np.asarray(rawtest,float)
        elif comparable:
            x=fixed_record_map(raw,True);xt=fixed_record_map(rawtest,True)
        else:
            scaler=StandardScaler().fit(np.sign(raw)*np.log1p(np.abs(raw)))
            x=scaler.transform(np.sign(raw)*np.log1p(np.abs(raw)))
            xt=scaler.transform(np.sign(rawtest)*np.log1p(np.abs(rawtest)))
            # Target-only map; row bounded after local training standardisation.
            x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            xt=xt/np.maximum(np.linalg.norm(xt,axis=1,keepdims=True),1e-12)
        km=KMeans(n_clusters=k*clusters,n_init=5,random_state=seed).fit(x)
        centers=km.cluster_centers_;cl=km.labels_
        d=gpu_sqdist(x,centers);dt=gpu_sqdist(xt,centers)
        width=max(float(np.median(d.min(1)))*4,1e-3)*bandwidth
        resp=softmax(-d/width,axis=1);testresp=softmax(-dt/width,axis=1)
        neighbours=NearestNeighbors(n_neighbors=min(16,len(x))).fit(x)
        distance,index=neighbours.kneighbors(x)
        distance,index=distance[:,1:],index[:,1:]
        local=max(float(np.median(distance[:,-1]**2)),1e-4)*bandwidth
        weights=softmax(-distance**2/local,axis=1)
        transition=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(len(x)),index.shape[1]),index.ravel())),shape=(len(x),len(x)))
        distance,index=neighbours.kneighbors(xt,n_neighbors=min(15,len(x)))
        weights=softmax(-distance**2/local,axis=1)
        testtransition=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(len(xt)),index.shape[1]),index.ravel())),shape=(len(xt),len(x)))
        # Same anchor kernel supplied to every message-level transport comparator.
        # All label-aware parameters will be frozen using development-only validation.
        awidth=max(float(np.median(gpu_sqdist(x[indices],x[indices]))) if len(indices)>1 else .5,1e-3)*bandwidth
        ap=kernel_prob(x,x[indices],labels,k,awidth)
        at=kernel_prob(xt,x[indices],labels,k,awidth)
        ac=kernel_prob(centers,x[indices],labels,k,awidth)
        target_mass=np.maximum(np.bincount(cl,minlength=len(centers)),1).astype(float);target_mass/=target_mass.sum()
        class_mass=np.maximum(ap.mean(0),.01);class_mass/=class_mass.sum()
        return cls(x,xt,centers,cl,resp,testresp,indices,labels,k,comparable,transition,testtransition,ap,at,ac,width,target_mass,class_mass)

    def cost(self,joint,relational_weight=.25,semantic_weight=1.,ordered=False,rho=1.,signature_length=16):
        message,_=joint.decode();prototypes=message.prototypes
        if self.comparable:
            direct=gpu_sqdist(self.centers,prototypes)/4
            s=signatures(prototypes,signature_length);t=signatures(self.centers,signature_length)
            relational=np.mean((t[:,None,:]-s[None,:,:])**2,axis=2)
            geometric=rho*direct+(1-rho)*relational
        else:
            s=signatures(prototypes,signature_length)
            t=signatures(self.centers,signature_length)
            geometric=np.mean((t[:,None,:]-s[None,:,:])**2,axis=2)
            if ordered and len(self.anchor_indices):
                # Labelled target anchors establish class semantics, not record pairs.
                tc=np.stack([self.x[self.anchor_indices[self.anchor_labels==c]].mean(0) for c in range(self.k)])
                ds=cdist(prototypes,prototypes)
                dt=cdist(self.centers,tc)
                ss=max(float(np.median(ds[np.triu_indices(self.k,1)])),.01)
                st=max(float(np.median(cdist(tc,tc)[np.triu_indices(self.k,1)])),.01)
                rs=np.minimum(ds/ss,4)/4;rt=np.minimum(dt/st,4)/4
                geometric=.5*geometric+.5*np.mean((rt[:,None,:]-rs[None,:,:])**2,axis=2)
        semantic=-np.log(np.maximum(self.center_anchor_prob,1e-8))/8
        if not len(self.anchor_indices):semantic[:]=0
        return relational_weight*geometric+semantic_weight*semantic

    def propagate(self,initial,steps=100,clamp=True):
        p=initial.copy()
        for _ in range(steps):
            p=.95*(self.transition@p)+.05*initial
            if clamp and len(self.anchor_indices):p[self.anchor_indices]=np.eye(self.k)[self.anchor_labels]
        return p

    def predict(self,joint,variant='full',relational_weight=1.,semantic_weight=.25,mass=.9,tau=.01,blend=.7,ordered=False,graph_mix=1.,adaptive_mass=False,rho=1.,prototype_mix=0.,prototype_temperature=.1,count_prior=0.,signature_length=16):
        cost=self.cost(joint,relational_weight,semantic_weight,ordered,rho,signature_length)
        plan,s=fixed_transport(cost,mass,tau,self.target_mass if adaptive_mass else None,self.class_mass if adaptive_mass else None)
        teacher=s[:,:self.k].copy();teacher/=np.maximum(teacher.sum(1,keepdims=True),1e-15)
        if variant=='hard':teacher=np.eye(self.k)[teacher.argmax(1)]
        initial=self.responsibilities@teacher
        initial=blend*initial+(1-blend)*self.anchor_prob
        if prototype_mix and self.comparable:
            message,counts=joint.decode()
            prior=np.maximum(counts,1.);prior/=prior.sum()
            direct=softmax(-gpu_sqdist(self.x,message.prototypes)/prototype_temperature+count_prior*np.log(prior)[None,:],axis=1)
            initial=(1-prototype_mix)*initial+prototype_mix*direct
        if variant=='no_graph':
            pred=blend*(self.test_responsibilities@teacher)+(1-blend)*self.test_anchor_prob
        else:
            p=self.propagate(initial)
            pred=graph_mix*(self.test_transition@p)+(1-graph_mix)*self.test_anchor_prob
        pred=np.maximum(pred,1e-12);pred/=pred.sum(1,keepdims=True)
        return {'probability':pred,'plan':plan,'conditional_plan':s,'cost':cost,'teacher':teacher}

    def baseline(self,joint,method,parameters):
        if method=='Target-RBF':return self.test_anchor_prob
        if method=='Target-SVM':
            if not len(self.anchor_indices):return np.full((len(self.xtest),self.k),1/self.k)
            model=SVC(C=10,gamma='scale',probability=False,decision_function_shape='ovr',random_state=0).fit(self.x[self.anchor_indices],self.anchor_labels)
            scores=model.decision_function(self.xtest)
            if scores.ndim==1:scores=np.c_[-scores,scores]
            # Tiny-anchor Platt cross-validation can reverse decisions; use
            # decision-preserving scores and label them uncalibrated explicitly.
            return softmax(scores,axis=1)
        if method=='Target-LP':return self.test_transition@self.propagate(self.anchor_prob)
        if method=='Prototype-Only':
            if self.comparable:
                msg,_=joint.decode();return softmax(-gpu_sqdist(self.xtest,msg.prototypes)/self.width,axis=1)
            return self.test_anchor_prob # no coordinate comparison is valid here
        if method=='Rel-FullOT':return self.predict(joint,mass=1.,**{k:v for k,v in parameters.items() if k!='mass'})['probability']
        if method=='Rel-UOT':
            cost=self.cost(joint,parameters['relational_weight'],parameters['semantic_weight'],parameters.get('ordered',False))
            a=np.full(len(cost),1/len(cost));b=np.full(self.k,1/self.k)
            plan=ot.unbalanced.sinkhorn_unbalanced(a,b,cost,parameters['tau'],.1,numItermax=3000)
            teacher=plan/np.maximum(plan.sum(1,keepdims=True),1e-15)
            initial=parameters['blend']*(self.responsibilities@teacher)+(1-parameters['blend'])*self.anchor_prob
            return parameters.get('graph_mix',1.)*(self.test_transition@self.propagate(initial))+(1-parameters.get('graph_mix',1.))*self.test_anchor_prob
        if method=='EGW-Private':
            # A message-level FGW control, not a reproduction of the JMLR EGW
            # variational estimator, which has a different input/solver contract.
            msg,_=joint.decode();ct=cdist(self.centers,self.centers)/2;cs=cdist(msg.prototypes,msg.prototypes)/2
            a=np.full(len(ct),1/len(ct));b=np.full(self.k,1/self.k)
            cost=-np.log(np.maximum(self.center_anchor_prob,1e-8))/8
            if not len(self.anchor_indices):cost[:]=0
            plan=ot.gromov.entropic_fused_gromov_wasserstein(cost,ct,cs,a,b,epsilon=parameters['tau'],alpha=.5,max_iter=100,tol=1e-8)
            if not np.isfinite(plan).all() or np.min(plan.sum(1))<=1e-15:
                raise RuntimeError('EGW control did not produce a finite plan')
            teacher=plan/plan.sum(1,keepdims=True)
            initial=parameters['blend']*(self.responsibilities@teacher)+(1-parameters['blend'])*self.anchor_prob
            return parameters.get('graph_mix',1.)*(self.test_transition@self.propagate(initial))+(1-parameters.get('graph_mix',1.))*self.test_anchor_prob
        raise ValueError(method)

def certify_plan(problem,joint,output,tau,relational_weight,alpha=.05,ordered=False,rho=1.):
    if hasattr(joint,'base'):
        from prifct.denoise import effective_radii
        _,radii,detail=prototype_radii(joint.base,alpha)
        radii=effective_radii(joint,radii)
    else:_,radii,detail=prototype_radii(joint,alpha)
    # Source distance /2 is 1/2-Lipschitz in each endpoint. Quantiles are
    # nonexpansive under paired max error. Squared-signature MSE on [0,1]
    # changes by at most 2*max prototype radius.
    dc=((2-rho)*np.max(radii) if problem.comparable else 2*np.max(radii))*relational_weight
    if ordered and not problem.comparable:
        # Median distance is nonexpansive and floored at .01. The ratio has
        # bounded numerator 2, so its worst-case Lipschitz constant is 20200.
        # Clipping /4 and squared [0,1] cost produce a valid, conservative
        # ordered-descriptor error bound. Do not reuse the unordered bound.
        dc=relational_weight*min(1.,2*np.max(radii)*20200)
    # Augmented plan has total mass 1 and fixed rows a_i=1/Ct. Strong entropy
    # convexity gives ||T-T0||_1 <= ||G-G0||_inf/tau.
    plan_radius=min(2.,dc/tau)
    # Adaptive margins are estimated from target training only and remain
    # fixed in the private/clean comparison.
    row_mass=output['plan'].sum(1)
    # Each row difference sums to zero. Its largest absolute coordinate is
    # at most half its l1 norm, which is at most the global l1 deviation.
    score_radius=np.minimum(1.,plan_radius/(2*row_mass))
    s=output['conditional_plan'];winner=s.argmax(1)
    sortedscore=np.sort(s,axis=1);gap=sortedscore[:,-1]-sortedscore[:,-2]
    accepted=(gap>2*score_radius)&(winner<problem.k)
    return {'prototype_radii':radii,'cost_radius':dc,'plan_l1_radius':plan_radius,'row_score_radius':score_radius,
        'accepted_rows':accepted,'gap':gap,'alpha':alpha,'meaning':'agreement with clean augmented transport row decision; not semantic correctness'}

def empirical_screen(problem,joint,parameters,seed,B=100,multiplier=1.):
    # Kept as an explicitly local heuristic comparison, never a semantic certificate.
    original=problem.predict(joint,**parameters)['conditional_plan'].argmax(1)
    rng=np.random.default_rng(seed);same=[]
    for _ in range(B):
        pert=JointMessage(joint.raw+rng.normal(size=joint.raw.shape)*joint.sigma_query*multiplier,
                          joint.sigma_query,joint.epsilon,joint.n_public)
        s=problem.predict(pert,**parameters)['conditional_plan'].argmax(1)
        same.append(s==original)
    successes=np.asarray(same).sum(0)
    lower=np.where(successes>0,beta.ppf(.05,successes,B-successes+1),0)
    return {'accepted':lower>=.9,'local_stability':successes/B,'local_lower95':lower,'B':B,'multiplier':multiplier}
