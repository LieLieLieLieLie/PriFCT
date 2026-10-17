"""Disjoint identities/subjects, no public cross-view pairs."""
import itertools
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from core.preprocessing import fixed_record_map
from revision.run_study import read_mf, FAMILIES
from revision.har_data import load_har_raw
from revision.daily_sports_data import load_daily, VIEWS

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[42,123,456,789,2026,3141,2718,8080,9001,10007]
DEVELOPMENT_SEEDS=[1701,1702,1703]
EPS=[.1,.5,1.,2.,5.,10.,float('inf')]
_CACHE={}

def load(name):
    if name in _CACHE:return _CACHE[name]
    if name=='MF':
        views,y=read_mf();subjects=None;meta={'classes':10,'view_relation':'different descriptor families'}
    elif name=='HAR':views,y,subjects,meta=load_har_raw()
    elif name=='DSA':views,y,subjects,meta=load_daily()
    elif name=='WDBC':
        raw=np.loadtxt(ROOT/'data/wdbc/wdbc.data',delimiter=',',dtype=str)
        x=raw[:,2:].astype(float);y=(raw[:,1]=='M').astype(int);subjects=None
        views={'mean':x[:,:10],'worst':x[:,20:30]};meta={'view_relation':'comparable nuclear descriptors'}
    elif name=='Office':
        views={};ys=[]
        for v in ['art','clipart','product','real_world']:
            d=np.load(ROOT/'data/office_home/.cache'/(v+'_resnet50_all65.npz'),allow_pickle=True)
            views[v]=d['features'];ys.append(d['labels'])
        y=ys;subjects=None;meta={'view_relation':'same frozen ImageNet ResNet50 representation, domain shift','classes':65}
    elif name=='CWRU':
        views={};ys=[]
        for i in range(4):
            raw=np.load(ROOT/'data/cwru/.cache'/f'load{i}_stat64_v2.npz')
            views[str(i)]=raw['features'][:,32:64];ys.append(raw['labels'])
        y=ys;subjects=None;meta={'view_relation':'same 32 coordinates across load conditions','correlated_windows':True}
    else:raise ValueError(name)
    _CACHE[name]=(views,y,subjects,meta);return _CACHE[name]

def pairs(name,full=True):
    if name=='MF':v=FAMILIES
    elif name=='HAR':v=['accel','gyro']
    elif name=='DSA':v=VIEWS
    elif name=='WDBC':v=['mean','worst']
    else:v=list(map(str,range(4)))
    all_pairs=list(itertools.permutations(v,2))
    return all_pairs if full else all_pairs[:2]

def case(name,source,target,seed,source_map='original'):
    views,y,subjects,meta=load(name)
    rng=np.random.default_rng(seed+90100)
    if name=='CWRU':
        sy=y[int(source)];ty=y[int(target)];src=[];tr=[];te=[]
        for c in range(4):
            ix=np.flatnonzero(sy==c);src.extend(ix[:int(.7*len(ix))-1])
            ix=np.flatnonzero(ty==c);cut=int(.7*len(ix));tr.extend(ix[:cut-1]);te.extend(ix[cut:])
        src,tr,te=map(np.asarray,[src,tr,te]);source_y=sy[src];train_y=ty[tr];test_y=ty[te]
    else:
        if subjects is None:
            src,tmp=train_test_split(np.arange(len(y)),test_size=.5,stratify=y,random_state=seed)
            tr,te=train_test_split(tmp,test_size=.4,stratify=y[tmp],random_state=seed+11)
        else:
            order=rng.permutation(np.unique(subjects))
            if name=='HAR':groups=[order[:12],order[12:22],order[22:]]
            else:groups=[order[:3],order[3:6],order[6:]]
            src,tr,te=[np.flatnonzero(np.isin(subjects,g)) for g in groups]
        assert not (set(src)&set(tr) or set(src)&set(te) or set(tr)&set(te))
        source_y,train_y,test_y=y[src],y[tr],y[te]
    # Source map is fixed, row-local and bounded. Target-only standardisation
    # is fit later using target training observations; target DP is not claimed.
    sx=fixed_record_map(views[source][src],log_transform=True)
    if source_map!='original':
        # Every operation is fixed/row-local; no private cohort normalisation.
        sx=sx-sx.mean(1,keepdims=True)
        if sx.shape[1]>16:
            q,_=np.linalg.qr(np.random.default_rng(87123).normal(size=(sx.shape[1],16)))
            sx=sx@q
        sx=sx/np.maximum(np.linalg.norm(sx,axis=1,keepdims=True),1e-12)
    tx=np.asarray(views[target][tr],float);test=np.asarray(views[target][te],float)
    # Comparable mode uses the SAME row-local map on both sides.
    comparable=name in ['WDBC','CWRU'] and source_map=='original'
    return {'source':sx,'source_y':source_y,'train':tx,'train_y':train_y,'test':test,'test_y':test_y,
        'k':{'MF':10,'HAR':6,'DSA':19,'WDBC':2,'CWRU':4,'Office':65}[name],'comparable':comparable,'source_indices':src,'train_indices':tr,'test_indices':te,
        'subjects_test':None if subjects is None else subjects[te],'metadata':meta,
        'source_view':source,'target_view':target,'dataset':name,'seed':seed}

def anchors(y,k,budget,seed):
    rng=np.random.default_rng(seed+90600)
    return np.concatenate([rng.permutation(np.flatnonzero(y==c))[:budget] for c in range(k)]).astype(int)
