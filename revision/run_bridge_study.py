"""Public-bridge extension; all alternatives share its additional information.

Uses a randomly drawn public proxy pool WITHOUT accessing labels for selection.
Its identities are disjoint from source/train/test. Public pairs simulate an
available independent resource; original private cohorts are never paired.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import json,itertools,argparse,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from revision.public_bridge import fit_bridge
from revision.run_study import ROOT,OUT,SEEDS,EPS,FAMILIES,PRIMARY_PAIRS,read_mf,truth_mapping,safe_json
from revision.protocol import release,align,assignment,screen,classify
from revision.joint_release import release_joint,screen_joint


def partitions(n,y,seed,proxy_size,reserve_size=400):
    # Same fixed random permutation prefixes across different pool sizes.
    perm=np.random.default_rng(seed+11000).permutation(n)
    reserve=max(proxy_size,reserve_size)
    proxy=perm[:proxy_size];remaining=perm[reserve:]
    fit=proxy[:int(.8*proxy_size)];cal=proxy[int(.8*proxy_size):]
    src,tgt=train_test_split(remaining,test_size=.5,stratify=y[remaining],random_state=seed)
    tr,te=train_test_split(tgt,test_size=.3,stratify=y[tgt],random_state=seed+11)
    assert len(set(perm[:reserve])|set(src)|set(tr)|set(te))==n
    assert sum(map(len,[perm[:reserve],src,tr,te]))==n
    return fit,cal,src,tr,te


def evaluate(xs,xt,y,seed,eps,pool=400,rho=.5,dim=16,shuffle=False,screening=True,reserve=400,partition_override=None,public_k=10):
    fit,cal,src,tr,te=(partitions(len(y),y,seed,pool,reserve) if partition_override is None else partition_override)
    if partition_override is not None:
        pool=len(fit)+len(cal)
        sets=[set(part) for part in partition_override]
        assert all(not a&b for a,b in itertools.combinations(sets,2))
    ps=xs[fit];pt=xt[fit]
    cv=xt[cal]
    if shuffle:
        pt=pt[np.random.default_rng(seed+31000).permutation(len(pt))]
        cv=cv[np.random.default_rng(seed+31001).permutation(len(cv))]
    bridge=fit_bridge(ps,pt,xs[cal],cv,dimension=dim)
    source=bridge.source(xs[src]);xtr=bridge.target(xt[tr]);xte=bridge.target(xt[te])
    # Declared benchmark class universe, never inferred from protected labels.
    k=int(public_k)
    if np.any((y<0)|(y>=k)):raise ValueError('Labels outside the declared public class universe')
    km=KMeans(n_clusters=k,n_init=20,max_iter=500,random_state=seed).fit(xtr)
    clusters=km.labels_;centers=km.cluster_centers_;testc=km.predict(xte)
    joint=release_joint(source,y[src],k,eps,seed+80000)
    message,noisy_counts=joint.decode()
    s,m,g,nw=align(centers,message,rho,tau=.05,declared_coordinates=True)
    ref=truth_mapping(clusters,y[tr],k)
    cond=s@message.prototypes;pseudo=m[clusters]
    results={'FedKAPT-Bridge':classify(np.c_[xtr,cond[clusters]],np.c_[xte,cond[testc]],pseudo,y[te]),
             'Bridge-Assignment':classify(xtr,xte,pseudo,y[te]),
             'Bridge-OneHot':classify(np.c_[xtr,np.eye(k)[pseudo]],np.c_[xte,np.eye(k)[m[testc]]],pseudo,y[te])}
    for mode,label in [('full','Bridge-FullOT'),('uot','Bridge-UOT')]:
        _,mb,_,_=align(centers,message,rho,mode=mode,tau=.05,declared_coordinates=True)
        results[label]=classify(xtr,xte,mb[clusters],y[te]);results[label]['mapping']=mb.tolist()
    md,_=assignment(-cdist(centers,message.prototypes)**2)
    results['Bridge-Direct']=classify(xtr,xte,md[clusters],y[te])
    # Strong prototype classifier: no clustering, no transport, same common codes.
    pred=cdist(xte,message.prototypes).argmin(1)
    from sklearn.metrics import accuracy_score,f1_score
    results['Bridge-NearestProto']={'accuracy':float(accuracy_score(y[te],pred)),
           'macro_f1':float(f1_score(y[te],pred,average='macro',zero_division=0)),
           'predictions':pred.tolist(),'n_test':len(te),'n_correct':int((pred==y[te]).sum())}
    # Strong same-message continuous-information controls.
    teacher=cdist(xtr,message.prototypes).argmin(1)
    results['Bridge-ProtoST']=classify(xtr,xte,teacher,y[te])
    from sklearn.neighbors import NearestNeighbors
    from scipy.special import softmax
    import scipy.sparse as sparse
    q=softmax(-cdist(xtr,message.prototypes)**2/.1,axis=1)
    nbr=NearestNeighbors(n_neighbors=11).fit(xtr)
    distance,index=nbr.kneighbors(xtr);distance,index=distance[:,1:],index[:,1:]
    affinity=np.exp(-distance**2/.1);affinity/=affinity.sum(1,keepdims=True)
    graph=sparse.csr_matrix((affinity.ravel(),(np.repeat(np.arange(len(xtr)),10),index.ravel())),shape=(len(xtr),len(xtr)))
    qt=q.copy()
    for iteration in range(50):qt=.8*q+.2*graph@qt
    dte,ite=nbr.kneighbors(xte,n_neighbors=10);weight=np.exp(-dte**2/.1);weight/=weight.sum(1,keepdims=True)
    pq=(weight[:,:,None]*qt[ite]).sum(1);pred_graph=pq.argmax(1)
    results['Bridge-GraphProto']={'accuracy':float(accuracy_score(y[te],pred_graph)),
          'macro_f1':float(f1_score(y[te],pred_graph,average='macro',zero_division=0)),
          'predictions':pred_graph.tolist(),'n_test':len(te),'n_correct':int((pred_graph==y[te]).sum())}
    clean,_=release_joint(source,y[src],k,float('inf'),seed+80000).decode()
    sc,mc,_,_=align(centers,clean,rho,tau=.05,declared_coordinates=True)
    scr=screen_joint(centers,joint,rho,s,m,g,seed+90000,tau=.05,
          multipliers=(.5,1.,2.) if eps in [.5,2.,10.] else (1.,),counts=(25,50,100) if eps in [.5,2.,10.] else (100,)) if screening else []
    return {'seed':seed,'epsilon':eps,'proxy_size':pool,'rho':rho,'tau':.05,'dimension':len(message.prototypes[0]),
            'shuffle_public_pairs':shuffle,'calibration_error':bridge.calibration_error,
            'residual_variance':bridge.residual_variance,'mapping':m,'reference_mapping':ref,
            'correct_assignment':bool(np.array_equal(m,ref)),'fraction_correct_clusters':float(np.mean(m==ref)),
            'oracle_cluster_test_accuracy':float(np.mean(ref[testc]==y[te])),
            'pseudo_label_accuracy':float(np.mean(pseudo==y[tr])),
            'noisy_counts':noisy_counts,'sigma_query':joint.sigma_query,'joint_raw':joint.raw,
            'privacy_scope':'full-labelled-record replace-one; public N and K','results':results,'screen':scr,
            'source_n':len(src),'target_train_n':len(tr),'target_test_n':len(te),
            'private_clean_plan_error':float(np.max(abs(s-sc))),
            'sigma':message.sigma,'payload_bytes':joint.raw.nbytes,
            'partition':{'public_fit':fit,'public_calibration':cal,'source':src,'target_train':tr,'target_test':te},
            'source_basis':bridge.source_basis,'target_basis':bridge.target_basis,
            'source_mean':bridge.source_mean,'target_mean':bridge.target_mean,
            'source_scale':bridge.source_scale,'target_scale':bridge.target_scale,
            'prototypes':message.prototypes,'centers':centers,'plan':s,'solver_warnings':nw}


def save(stage,ident,row):
    payload_fields=['source_basis','target_basis','source_mean','target_mean','source_scale','target_scale','prototypes','centers','plan','joint_raw']
    filename=stage+'_'+ident.replace('|','_')+'.npz'
    np.savez_compressed(OUT/'messages'/filename,**{k:row.pop(k) for k in payload_fields})
    row['id']=ident;row['message_file']='messages/'+filename
    with (OUT/(stage+'.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(safe_json(row),allow_nan=False)+'\n')


def run(stage='bridge_joint',pilot=False):
    mf,y=read_mf();path=OUT/(stage+'.jsonl')
    completed={json.loads(x)['id'] for x in path.read_text().splitlines()} if path.exists() else set()
    pairs=PRIMARY_PAIRS if pilot else list(itertools.permutations(FAMILIES,2))
    for a,b in pairs:
        primary=(a,b) in PRIMARY_PAIRS
        for seed in SEEDS[:2] if pilot else SEEDS:
            for eps in [2.,float('inf')] if pilot else (EPS if primary else [2.]):
                ident=f'MF-{a}-{b}|{seed}|{eps}'
                if ident in completed:continue
                r=evaluate(mf[a],mf[b],y,seed,eps)
                r['dataset']='MF-'+a+'-'+b;save(stage,ident,r)
            print(f'{stage} {a}-{b} seed {seed}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--sensitivity',action='store_true');parser.add_argument('--clinical',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.clinical:
            w=np.loadtxt(ROOT/'data'/'wdbc'/'wdbc.data',delimiter=',',dtype=str);x=w[:,2:].astype(float);y=(w[:,1]=='M').astype(int)
            stage='bridge_clinical';path=OUT/(stage+'.jsonl');completed={json.loads(x)['id'] for x in path.read_text().splitlines()} if path.exists() else set()
            for seed in SEEDS:
                for eps in EPS:
                    ident=f'WDBC|{seed}|{eps}'
                    if ident in completed:continue
                    r=evaluate(x[:,:10],x[:,20:30],y,seed,eps,pool=100,reserve=100,dim=8,public_k=2)
                    r['dataset']='WDBC';save(stage,ident,r)
                print(stage+' seed '+str(seed),flush=True)
        elif args.sensitivity:
            mf,y=read_mf();stage='bridge_sensitivity';path=OUT/(stage+'.jsonl');completed={json.loads(x)['id'] for x in path.read_text().splitlines()} if path.exists() else set()
            configurations=[(100,16,.5,False),(200,16,.5,False),(400,8,.5,False),(400,32,.5,False),
                            (400,16,0.,False),(400,16,1.,False),(400,16,.5,True)]
            for seed in SEEDS:
                for pool,dim,rho,shuffle in configurations:
                    ident=f'MF-fac-kar|{seed}|{pool}|{dim}|{rho}|{shuffle}'
                    if ident in completed:continue
                    r=evaluate(mf['fac'],mf['kar'],y,seed,2.,pool=pool,dim=dim,rho=rho,shuffle=shuffle)
                    r['dataset']='MF-fac-kar';save(stage,ident,r)
                print(stage+' seed '+str(seed),flush=True)
        else:run('bridge_joint_pilot' if args.pilot else 'bridge_joint',args.pilot)
