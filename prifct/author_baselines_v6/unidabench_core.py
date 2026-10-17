"""Audited author objective runner; extra-resource, NONPRIVATE contextual methods.

Only selected unchanged AST definitions are compiled from the pinned author
files. Unrelated trainer/backbone dependencies are not imported. A common
feature MLP is an explicit architecture adaptation, not native raw-series FNO.
No target labels, validation objects or metric callback can enter fit().
"""
import ast
from collections import defaultdict
import copy
from copy import deepcopy
import hashlib
import itertools
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import ot
import skimage.filters as sfil
import torch
from torch import nn
from torch.nn import BCELoss
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from tqdm import tqdm

HERE=Path(__file__).resolve().parent
COMMIT='da3e5cfe02952d38f081e30a130d4660bc1e64fb'
UPSTREAM=HERE/'upstream'/COMMIT


class TrainingModeManager:
    """Equivalent train/eval context for the three imported author methods."""
    def __init__(self, modules, train=False):
        self.modules=list(modules);self.train=train
    def __enter__(self):
        self.previous=[m.training for m in self.modules]
        for m in self.modules:m.train(self.train)
        return self
    def __exit__(self,*args):
        for m,previous in zip(self.modules,self.previous):m.train(previous)


class Accumulator:
    """Only the author's centroid accumulator API; concatenation, not averaging."""
    def __init__(self,names):self.names=names;self.values={n:[] for n in names}
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def updateData(self,values):
        for n in self.names:self.values[n].append(values[n])
    def __iter__(self):return iter(self.names)
    def __getitem__(self,n):return np.concatenate(self.values[n],axis=0)


def _compile_selected(relative,names,namespace):
    source=(UPSTREAM/relative).read_text(encoding='utf8')
    expected=json.loads((HERE/'upstream_provenance.json').read_text())['files'][relative]['sha256']
    if hashlib.sha256((UPSTREAM/relative).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Pinned author file changed: '+relative)
    parsed=ast.parse(source)
    # Last same-named class wins in the original module as well.
    picked={node.name:node for node in parsed.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names}
    missing=set(names)-set(picked)
    if missing:raise RuntimeError('Missing exact author definitions: '+str(missing))
    selected=[picked[name] for name in names]
    module=ast.Module(body=selected,type_ignores=[])
    exec(compile(module,str(UPSTREAM/relative),'exec'),namespace)
    return [{'file':relative,'definition':name,'line':picked[name].lineno,'end_line':picked[name].end_lineno} for name in names]


def author_namespace():
    namespace=dict(globals())
    manifest=[]
    manifest+=_compile_selected('models/models.py',['classifier','ProtoCLS','CLS','MemoryQueue','ClassMemoryQueue'],namespace)
    manifest+=_compile_selected('utils.py',['sinkhorn','ubot_CCD','ubot_CCD2','adaptive_filling','adaptive_filling2'],namespace)
    manifest+=_compile_selected('algorithms/algorithms.py',['Algorithm','PPOT','UniOT','UniJDOT'],namespace)
    namespace['selected_definition_manifest']=manifest
    return namespace


class CommonFeatureMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(config.common_dimension,config.final_out_channels),
            nn.ReLU(),nn.Linear(config.final_out_channels,config.final_out_channels))
    def forward(self,x):return self.layers(x.flatten(1))


class LabelBlindTargetDataset(Dataset):
    def __init__(self,x,indexed):
        self.x_data=torch.as_tensor(x,dtype=torch.float32)
        self.y_data=torch.zeros(len(x),dtype=torch.long) # fixed dummy, no target label supplied
        self.indexed=indexed
    def __len__(self):return len(self.x_data)
    def __getitem__(self,i):
        result=(self.x_data[i],self.y_data[i])
        return result+(i,) if self.indexed else result


class LabelledSourceDataset(LabelBlindTargetDataset):
    def __init__(self,x,y,indexed):
        super().__init__(x,indexed)
        self.y_data=torch.as_tensor(y,dtype=torch.long)


class Meter:
    def __init__(self):self.sum=0.;self.count=0
    def update(self,value,n=1):self.sum+=float(value)*n;self.count+=n
    @property
    def avg(self):return self.sum/max(1,self.count)


class LossMeters(defaultdict):
    def __init__(self):super().__init__(Meter)
    def __getitem__(self,key):
        # UniOT/UniJDOT native checkpoint code requests Src_cls_loss whereas
        # its training_epoch stores loss_cls. Alias logging only; fixed-last
        # selection remains independent of this private source diagnostic.
        if key=='Src_cls_loss' and key not in self and 'loss_cls' in self:key='loss_cls'
        return super().__getitem__(key)


DEFAULTS={
 'PPOT':dict(learning_rate=.005,weight_decay=1e-4,batch_size=32,
   num_epochs_pr=20,num_epochs=20,alpha=.01,beta=.01,tau=.5,tau1=.4,tau2=.5,
   ot=1.,neg=.2,p_entropy=.1,n_entropy=.1,reg=.02,thresh=.5),
 'UniOT':dict(learning_rate=1e-4,weight_decay=1e-4,batch_size=32,
   num_epochs_pr=20,num_epochs=20,K=20,MQ_size=320,temp=.05,gamma=.7,mu=.7,lam=.1),
 'UniJDOT':dict(learning_rate=5e-4,weight_decay=1e-4,batch_size=32,
   num_epochs_pr=20,num_epochs=20,K=20,n_batch=10,threshold_method='threshold_yen',
   alpha=1.,lamb=1.,src_weight=1.,joint_decision=True,trg_mem_size=320)}


def fit(method,source_x,source_y,target_x,num_classes,hparams=None,seed=75801,hidden=32):
    """Train on labelled source and UNLABELLED target; no evaluator is called.

    Caller must provide fixed public common features for both views. The
    author CUDA calls are retained; require GPU explicitly instead of mocking.
    No privacy guarantee applies to these author objectives or source states.
    """
    if method not in DEFAULTS:raise ValueError('Unsupported author algorithm')
    if not torch.cuda.is_available():raise RuntimeError('Unmodified author objectives require CUDA')
    source_x=np.asarray(source_x,dtype=np.float32);target_x=np.asarray(target_x,dtype=np.float32)
    source_y=np.asarray(source_y,dtype=int)
    if source_x.ndim!=2 or target_x.ndim!=2 or source_x.shape[1]!=target_x.shape[1]:
        raise ValueError('Explicit common-map adapter is required for feature-disjoint raw views')
    if not np.isfinite(source_x).all() or not np.isfinite(target_x).all():raise ValueError('Finite covariates required')
    if source_y.shape!=(len(source_x),) or set(source_y)!=set(range(num_classes)):
        raise ValueError('Every declared source class must occur; no target-derived class schema')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
    torch.set_num_threads(1)
    params=dict(DEFAULTS[method]);params.update(hparams or {})
    # PPOT's active base optimiser uses a hardcoded .005; reject a misleading
    # learning-rate sweep until an explicitly reviewed bugfix is introduced.
    if method=='PPOT' and params['learning_rate']!=.005:
        raise ValueError('Native PPOT port ignores hparams learning_rate; actual rate is .005')
    if params['batch_size']>min(len(source_x),len(target_x)):raise ValueError('Not enough full training batches')
    if method=='UniOT' and params['MQ_size']%params['batch_size']:
        raise ValueError('Native queue size must be a multiple of batch size')
    config=SimpleNamespace(common_dimension=source_x.shape[1],final_out_channels=hidden,
        features_len=1,num_classes=num_classes,isFNO=False,isTimesNet=False)
    namespace=author_namespace()
    algorithm=namespace[method](CommonFeatureMLP,config,params,'cuda').cuda()
    indexed=method!='PPOT'
    source=LabelledSourceDataset(source_x,source_y,indexed)
    target=LabelBlindTargetDataset(target_x,indexed)
    generator=torch.Generator().manual_seed(seed)
    source_loader=DataLoader(source,batch_size=params['batch_size'],shuffle=True,drop_last=True,num_workers=0,generator=generator)
    target_loader=DataLoader(target,batch_size=params['batch_size'],shuffle=True,drop_last=True,num_workers=0,generator=generator)
    meters=LossMeters();logger=logging.getLogger('author-baseline-no-metrics')
    final,best=algorithm.update(source_loader,target_loader,meters,logger)
    # No target scoring or private-loss-selected checkpoint: retain actual last.
    algorithm.network.load_state_dict(final)
    diagnostics={'method':method,'commit':COMMIT,'resource':'extra-resource NONPRIVATE contextual',
      'architecture':'shared CommonFeatureMLP on fixed public-map features; not raw-series author CNN/FNO',
      'target_label_input':'absent; fixed zero dummy IDs only','checkpoint':'fixed last',
      'hparams':params,'seed':seed,'source_n':len(source),'target_n':len(target),
      'source_passes':'source CE/prototype/memory accessed repeatedly; not single-DP-summary',
      'definition_manifest':namespace['selected_definition_manifest'],
      'dependencies_bypassed':'unrelated original trainer/backbones only; exact selected objective AST retained',
      'compatibility_adapter':'centroid-only train/eval context and concatenation accumulator; source meter key alias'}
    return algorithm,diagnostics


@torch.no_grad()
def _predict_unseeded(algorithm,method,x):
    """Covariate-only native decision components; separate full posterior/reject.

    UniOT's native rejection solves OT jointly over the supplied inference
    cohort. Declare this batch inference resource; never treat it as pointwise
    inductive rejection. Full-coverage metrics use unmasked probabilities.
    """
    x=torch.as_tensor(x,dtype=torch.float32,device='cuda')
    algorithm.eval();feature=algorithm.feature_extractor(x)
    if method=='PPOT':
        logits=algorithm.classifier(feature);probability=F.softmax(logits,dim=1)
        reject=probability.max(dim=1).values<algorithm.hparams['thresh']
    else:
        projected,logits=algorithm.classifier(feature);probability=F.softmax(logits,dim=1)
        if method=='UniJDOT':
            if algorithm.hparams['joint_decision']:
                distance=algorithm.compute_cluster_distance(F.normalize(projected))
                confidence=algorithm.joint_decision(logits,distance).max(dim=1).values
            else:confidence=probability.max(dim=1).values
            reject=confidence<algorithm.final_threshold
        elif method=='UniOT':
            namespace=author_namespace()
            similarities,fake=namespace['adaptive_filling'](F.normalize(projected),
                algorithm.classifier.ProtoCLS.fc.weight,algorithm.hparams['gamma'],algorithm.beta,0,stopThr=1e-6)
            _,_,label,_=namespace['ubot_CCD'](similarities,algorithm.beta,
                fake_size=fake,fill_size=0,mode='minibatch',stopThr=1e-6)
            reject=label==algorithm.nb_classes
        else:raise ValueError(method)
    p=probability.cpu().numpy();r=reject.cpu().numpy()
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6):
        raise RuntimeError('Nonfinite/unnormalised author posterior; retain failure')
    return {'probabilities':p,'reject':r,'label':p.argmax(1),
        'inference':'batch-cohort native OT rejection' if method=='UniOT' else 'fixed-state covariate-only prediction'}


def predict(algorithm,method,x,inference_seed=75802):
    """Fix native adaptive-filling randomness without changing caller RNG state."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(inference_seed);torch.cuda.manual_seed_all(inference_seed)
        return _predict_unseeded(algorithm,method,x)


def save_checkpoint(path,algorithm,diagnostics):
    """Preserve all prediction state, including native unregistered attributes."""
    extra={}
    for name in ['beta','src_prototype','src_latent_cluster','trg_latent_cluster',
                 'trg_feats_mem','trg_preds_mem','alpha','class_weight']:
        if hasattr(algorithm,name):
            value=getattr(algorithm,name)
            extra[name]=value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
    state={key:value.detach().cpu().clone() for key,value in algorithm.state_dict().items()}
    for name in ['memqueue','memqueue_feat']:
        module=getattr(algorithm,name,None)
        if module is not None and hasattr(module,'next_write'):extra[name+'.next_write']=module.next_write
    torch.save({'diagnostics':diagnostics,'config':vars(algorithm.configs),
        'state_dict':state,'nonregistered_prediction_state':extra},path)


def load_checkpoint(path):
    """Restore a trusted local wrapper checkpoint without accessing any dataset."""
    payload=torch.load(path,map_location='cpu')
    diagnostic=payload['diagnostics'];method=diagnostic['method']
    if diagnostic['commit']!=COMMIT:raise RuntimeError('Checkpoint author source mismatch')
    namespace=author_namespace();config=SimpleNamespace(**payload['config'])
    algorithm=namespace[method](CommonFeatureMLP,config,diagnostic['hparams'],'cuda').cuda()
    algorithm.load_state_dict(payload['state_dict'])
    for name,value in payload['nonregistered_prediction_state'].items():
        if '.' in name:
            parent,key=name.split('.',1);setattr(getattr(algorithm,parent),key,value)
        else:setattr(algorithm,name,value.cuda() if torch.is_tensor(value) else value)
    return algorithm,diagnostic
