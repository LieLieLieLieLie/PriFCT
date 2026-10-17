"""Exploratory private class-function transfer through public paired kernels.

Nyström maps, kernel mean release and kernel distillation are established.
This candidate's public coupled operator and noise propagation must be
compared to simpler same-resource alternatives before any novelty claim.
"""
import numpy as np
import torch
from scipy.spatial.distance import pdist,cdist
from scipy.special import softmax
from revision.private_proxy_selection import fit_public_sensor_map
from core.privacy import compute_sigma
from revision.uncertain_moments import target_graph

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
DTYPE=torch.float64

def matrix(x):return torch.as_tensor(x,dtype=DTYPE,device=DEVICE)
def solve(a,b):return torch.linalg.solve(matrix(a),matrix(b)).cpu().numpy()

class KernelMap:
    def __init__(self,public,dimension=64,bandwidth_factor=1.):
        self.standard=fit_public_sensor_map(public)
        self.anchors=self.standard.transform(public)
        self.bandwidth=max(float(np.median(pdist(self.anchors)**2))*bandwidth_factor,1e-6)
        k=np.exp(-cdist(self.anchors,self.anchors)**2/self.bandwidth)
        eigenvalue,eigenvector=torch.linalg.eigh(matrix(k))
        ix=torch.arange(len(public)-1,max(-1,len(public)-dimension-1),-1,device=DEVICE)
        self.basis=(eigenvector[:,ix]/torch.sqrt(torch.clamp(eigenvalue[ix],min=1e-8))).cpu().numpy()
        self.public=self.transform(public)

    def transform(self,x):
        features=np.exp(-cdist(self.standard.transform(x),self.anchors)**2/self.bandwidth)@self.basis
        # Nyström feature norm <=1 theoretically; clip for numerical tails.
        return features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1.)

def class_function_query(features,y,k):
    if np.max(np.linalg.norm(features,axis=1))>1+1e-10:raise ValueError('Bounded public features required')
    query=np.zeros((features.shape[1],k))
    for c in range(k):query[:,c]=features[y==c].sum(0)/len(y)
    return query

def release_function(features,y,k,epsilon,seed,delta=1e-5):
    query=class_function_query(features,y,k)
    sigma=float(compute_sigma(2/len(y),epsilon,delta))
    raw=query+np.random.default_rng(seed).normal(size=query.shape)*sigma
    return raw,sigma


class KernelBridge:
    """Known regularised kernel CCA, optional public observability weights."""
    def __init__(self,ps,pt,cs,ct,dimension=16,rank=64,ridge=.01,weighted=False):
        self.sm,self.tm=KernelMap(ps,rank),KernelMap(pt,rank)
        a,b=self.sm.public,self.tm.public;n=len(a)
        self.ms,self.mt=a.mean(0),b.mean(0);a,b=a-self.ms,b-self.mt
        def invsqrt(c):
            w,v=torch.linalg.eigh(matrix(c))
            return ((v/torch.sqrt(torch.clamp(w,min=1e-12)))@v.T).cpu().numpy()
        ws=invsqrt(a.T@a/(n-1)+ridge*np.eye(a.shape[1]))
        wt=invsqrt(b.T@b/(n-1)+ridge*np.eye(b.shape[1]))
        u,s,v=torch.linalg.svd(matrix(ws@(a.T@b/(n-1))@wt),full_matrices=False)
        self.bs=ws@u[:,:dimension].cpu().numpy();self.bt=wt@v[:dimension].T.cpu().numpy()
        self.weights=np.ones(dimension);self.canonical=s[:dimension].cpu().numpy()
        if weighted:
            source=(self.sm.transform(cs)-self.ms)@self.bs
            target=(self.tm.transform(ct)-self.mt)@self.bt
            disagreement=np.mean((source-target)**2,axis=0)
            energy=np.mean(source**2+target**2,axis=0)*.5
            self.weights=np.sqrt(np.maximum(0,1-disagreement/np.maximum(2*energy,1e-10)))
        self.dimension=dimension

    def source(self,x):
        code=((self.sm.transform(x)-self.ms)@self.bs)*self.weights
        return code/np.maximum(np.linalg.norm(code,axis=1,keepdims=True),1e-15)

    def target(self,x):
        code=((self.tm.transform(x)-self.mt)@self.bt)*self.weights
        return code/np.maximum(np.linalg.norm(code,axis=1,keepdims=True),1e-15)

def function_predict(xs,xt,xtr,xte,query,ridge=.001,alignment=1.,graph_weight=0.,variant='coupled'):
    """One linear public operator maps the released class moments to scores.

    Source Hessian is PUBLIC, not private sufficient statistics. This assumes
    public source covariates approximate the source feature distribution.
    Coupling and target graph are public/target-unlabelled postprocessing.
    """
    n=len(xs);ds,dt=xs.shape[1],xt.shape[1]
    cs=xs.T@xs/n;ct=xt.T@xt/n;cross=xs.T@xt/n
    if variant=='two_stage':
        a=solve(cs+ridge*np.eye(ds),query)
        operator=solve(ct+ridge*np.eye(dt),cross.T)
        b=operator@a
    elif variant=='direct':
        a=solve(cs+ridge*np.eye(ds),query)
        # public paired score interpolation with target ridge
        b=solve(ct+ridge*np.eye(dt),xt.T@(xs@a)/n)
    elif variant=='coupled':
        hss=(1+alignment)*cs+ridge*np.eye(ds)
        htt=alignment*ct+ridge*np.eye(dt)
        if graph_weight:
            # Frozen target train graph: no test points used in fitting.
            graph=target_graph(xtr)
            htt+=graph_weight*xtr.T@(graph.laplacian@xtr)/len(xtr)
        h=np.block([[hss,-alignment*cross],[-alignment*cross.T,htt]])
        both=solve(h,np.r_[query,np.zeros((dt,query.shape[1]))])
        a,b=both[:ds],both[ds:]
    else:raise ValueError(variant)
    return xte@b,{'source_coefficients':a,'target_coefficients':b,'ridge':ridge,'alignment':alignment,
        'graph_weight':graph_weight,'variant':variant,'device':DEVICE,
        'scope':'public-Hessian conditional covariate model; no private covariance release'}
