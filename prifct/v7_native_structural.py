"""Pinned recent heterogeneous transport objectives, explicit task adapters.

Source rows and labels are available: these are NONPRIVATE context controls,
not equal-interface DP baselines. Native objectives are preserved; the common
inductive barycentric SVM head and public-pair KPG guidance are disclosed.
"""
import ast,hashlib,inspect
from pathlib import Path
import numpy as np
import torch
import ot
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parent/'author_baselines_v7'
AGW=ROOT/'AGW-AISTATS24/af0139f682531609d8eb3d6bd7e63fe38629ac23/src'
KPG=ROOT/'KPG-RL/51ffa1480a97842198d8d1e0be40293063cab529/HDA'


def load_body(path,skip=(),injected=None):
    source=path.read_text(encoding='utf8');tree=ast.parse(source)
    tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.module in skip)]
    namespace={} if injected is None else dict(injected)
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace


def author_provenance():
    files=[AGW/name for name in ['agw_scootr.py','optim.py','utils.py']]+[KPG/'utils.py',KPG/'keypointguide_POT/sinkhorn.py']
    installed=Path(inspect.getsourcefile(ot.gromov.unbalanced_co_optimal_transport))
    files.append(installed)
    return dict(pot_version=ot.__version__,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def barycentric_classifier(plan,target,source_y,*,classes,C=1.,seed=76601):
    plan=np.asarray(plan,float)
    if plan.shape!=(len(source_y),len(target)) or not np.isfinite(plan).all() or np.any(plan< -1e-12):
        raise ValueError('Finite nonnegative native sample coupling required')
    mass=plan.sum(1)
    if np.any(mass<=1e-16):raise ValueError('Native coupling contains zero source row mass; retain as failed fit')
    transferred=plan@target/mass[:,None]
    classifier=SVC(C=C,gamma='scale',probability=True,random_state=seed)
    classifier.fit(transferred,source_y,sample_weight=mass/mass.mean())
    if not np.array_equal(classifier.classes_,np.arange(classes)):raise ValueError('Complete fixed class catalog required')
    return classifier,dict(minimum_row_mass=float(mass.min()),maximum_row_mass=float(mass.max()),transported_mass=float(mass.sum()))


def agw(source,target,*,alpha=.5):
    utils=load_body(AGW/'utils.py')
    optim=load_body(AGW/'optim.py',skip=('utils',),injected=utils)
    namespace=load_body(AGW/'agw_scootr.py',skip=('utils','optim'),injected={**utils,**optim})
    from scipy.spatial.distance import cdist
    d1=cdist(source,source,'sqeuclidean');d2=cdist(target,target,'sqeuclidean')
    d1/=max(float(d1.max()),1e-12);d2/=max(float(d2.max()),1e-12)
    plan,feature_plan,value,log=namespace['agw_scootr'](source,target,d1,d2,alpha=alpha,niter=10,
        algo='cg',algo2='emd',scaleCost=True,supervised=False,verbose=False,log=True)
    return plan,dict(feature_plan=feature_plan,objective=float(value),objective_history=log['cost'],
        sample_row_error=float(np.max(abs(plan.sum(1)-1/len(source)))),
        sample_column_error=float(np.max(abs(plan.sum(0)-1/len(target)))))


def ucoot(source,target,*,marginal_reg=1.,entropy=.01):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for native UCOOT development')
    x=torch.as_tensor(source,device='cuda',dtype=torch.float64)
    y=torch.as_tensor(target,device='cuda',dtype=torch.float64)
    plan,feature,log=ot.gromov.unbalanced_co_optimal_transport(x,y,reg_marginals=marginal_reg,epsilon=entropy,
        divergence='kl',unbalanced_solver='sinkhorn',max_iter=50,max_iter_ot=500,tol=1e-7,tol_ot=1e-7,log=True)
    def plain(value):return value.detach().cpu().numpy() if torch.is_tensor(value) else value
    return plain(plan),dict(feature_plan=plain(feature),log={key:plain(value) if torch.is_tensor(value) else
        [plain(item) for item in value] if isinstance(value,list) else value for key,value in log.items()})


def kpg(source,target,public_source,public_target,*,reg=.005,relation_temperature=.1,keypoint_mass=.05):
    if len(public_source)!=len(public_target) or not 0<keypoint_mass<1:raise ValueError('True independent keypoint pairs required')
    utils=load_body(KPG/'utils.py');solver=load_body(KPG/'keypointguide_POT/sinkhorn.py')
    ns,nt,nkey=len(source),len(target),len(public_source)
    xs=np.vstack([public_source,source]);xt=np.vstack([public_target,target]);keys=np.arange(nkey)
    cs=utils['cost_matrix'](xs,xs);ct=utils['cost_matrix'](xt,xt)
    cs/=max(float(cs.max()),1e-12);ct/=max(float(ct.max()),1e-12)
    # Native JS relation objective, batched without altering the formula.
    r1=utils['softmax_matrix'](-cs[:,keys]/relation_temperature)
    r2=utils['softmax_matrix'](-ct[:,keys]/relation_temperature)
    cost=np.vstack([utils['JS_matrix'](r1[start:start+64],r2) for start in range(0,len(r1),64)])
    cost/=max(float(cost.max()),1e-12)
    mask=np.ones_like(cost);mask[keys,:]=0;mask[:,keys]=0;mask[keys,keys]=1
    # Native objective accepts arbitrary margins. Paired anchor margins MUST
    # agree when cohort lengths differ; uniform margins would be infeasible.
    p=np.r_[np.full(nkey,keypoint_mass/nkey),np.full(ns,(1-keypoint_mass)/ns)]
    q=np.r_[np.full(nkey,keypoint_mass/nkey),np.full(nt,(1-keypoint_mass)/nt)]
    args=[torch.as_tensor(v,device='cuda',dtype=torch.float64) for v in [p,q,cost,mask]]
    plan=solver['sinkhorn_log_domain_torch'](*args,reg=reg,niter=10000,thresh=1e-7).detach().cpu().numpy()
    error=max(float(np.max(abs(plan.sum(1)-p))),float(np.max(abs(plan.sum(0)-q))))
    if error>1e-6:raise RuntimeError('Native masked solver marginal error '+str(error))
    return plan[nkey:,nkey:],dict(keypoint_count=nkey,keypoint_mass=keypoint_mass,marginal_error=error,
        task_adapter='Public UNLABELLED true pairs instead of native labelled target/class keypoints; source-only SVM labels',
        forbidden_mass=float(plan[mask==0].sum()))
