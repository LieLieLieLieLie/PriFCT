"""Original unlabelled-target contract with independent single-view calibration.

Calibration rows/subjects are excluded from the private cohort and target
training/testing. Their features are explicitly public; labels are unused.
Unlike FedKAPT-O, no paired public observations train a cross-view bridge.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from prifct.data import load

class FrozenMap:
    def __init__(self,public,rank=16,pca=False,mode='log_standard'):
        self.mode=mode
        z=self.record_transform(public)
        self.scaler=StandardScaler(with_std=mode=='log_standard').fit(z)
        self.rank=min(rank,z.shape[1])
        if pca and z.shape[1]>self.rank:
            import torch
            device='cuda' if torch.cuda.is_available() else 'cpu'
            matrix=torch.as_tensor(self.scaler.transform(z),dtype=torch.float64,device=device)
            _,_,v=torch.linalg.svd(matrix,full_matrices=False)
            self.projection=v[:self.rank].T.cpu().numpy()
        elif z.shape[1]>self.rank:
            self.projection,_=np.linalg.qr(np.random.default_rng(88117).normal(size=(z.shape[1],self.rank)))
        else:self.projection=np.eye(z.shape[1])
    def record_transform(self,x):
        mode=getattr(self,'mode','log_standard')
        if mode=='log_standard':return np.sign(x)*np.log1p(np.abs(x))
        if mode=='unit_center':return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        if mode=='raw_center':return x
        raise ValueError(mode)
    def transform(self,x):
        z=self.scaler.transform(self.record_transform(x))@self.projection
        # All transformations are frozen BEFORE accessing private records.
        return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1.)

def case(name,a,b,seed,target_local=True,rank=16,image_mode='log_standard'):
    views,y,subjects,meta=load(name);rng=np.random.default_rng(seed+93300)
    if name=='Office':
        domain=['art','clipart','product','real_world'];sy=y[domain.index(a)];ty=y[domain.index(b)]
        order=rng.permutation(len(sy));pub=order[:min(400,int(.2*len(sy)))];src=order[len(pub):]
        tr,te=train_test_split(np.arange(len(ty)),test_size=.4,stratify=ty,random_state=seed+11)
        source_y,train_y,test_y=sy[src],ty[tr],ty[te]
    elif name=='CWRU':
        sy=y[int(a)];ty=y[int(b)];pub=[];src=[];tr=[];te=[]
        for c in range(4):
            ix=np.flatnonzero(sy==c);cut=int(.1*len(ix));stop=int(.7*len(ix));pub.extend(ix[:cut]);src.extend(ix[cut+1:stop-1])
            ix=np.flatnonzero(ty==c);stop=int(.7*len(ix));tr.extend(ix[:stop-1]);te.extend(ix[stop:])
        pub,src,tr,te=map(np.asarray,[pub,src,tr,te]);source_y=sy[src];train_y=ty[tr];test_y=ty[te]
    else:
        if subjects is None:
            order=rng.permutation(len(y));pub=order[:max(60,int(.1*len(y)))];available=order[len(pub):]
            src,tmp=train_test_split(available,test_size=.5,stratify=y[available],random_state=seed)
            tr,te=train_test_split(tmp,test_size=.4,stratify=y[tmp],random_state=seed+11)
        else:
            order=rng.permutation(np.unique(subjects))
            groups=[order[:4],order[4:14],order[14:22],order[22:]] if name=='HAR' else [order[:1],order[1:3],order[3:6],order[6:]]
            pub,src,tr,te=[np.flatnonzero(np.isin(subjects,g)) for g in groups]
        assert all(not(set(u)&set(v)) for i,u in enumerate([pub,src,tr,te]) for v in [pub,src,tr,te][i+1:])
        source_y,train_y,test_y=y[src],y[tr],y[te]
    mode=image_mode if name=='Office' else 'log_standard'
    sm=FrozenMap(views[a][pub],rank,pca=name=='Office',mode=mode)
    source=sm.transform(views[a][src])
    comparable=name in ['WDBC','CWRU','Office']
    if comparable and not target_local:
        target=sm.transform(views[b][tr]);test=sm.transform(views[b][te])
        tm=sm
    else:
        # Target map learned locally; corresponding descriptor positions use
        # the same predeclared projection, with separate domain calibration.
        # No private source statistics are fitted or sent in this step.
        # This is not counted as feature correspondence or a shared projection.
        tm=FrozenMap(views[b][tr],rank,mode=mode)
        if comparable:tm.projection=sm.projection.copy()
        target=tm.transform(views[b][tr]);test=tm.transform(views[b][te])
    return {'source':source,'source_y':source_y,'train':target,'train_y':train_y,'test':test,'test_y':test_y,
       'k':{'MF':10,'HAR':6,'DSA':19,'WDBC':2,'CWRU':4,'Office':65}[name],'comparable':comparable,'source_map':sm,'target_map':tm,
       'source_indices':src,'train_indices':tr,'test_indices':te,'public_source_indices':pub,
       'subjects_test':None if subjects is None else subjects[te],'metadata':meta,'input_prepared':True}
