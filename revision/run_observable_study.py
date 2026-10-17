"""Frozen, resumable complete study. Each mechanism is a separate DP run.

These alternative messages must NOT be jointly published under one epsilon.
Saved clean comparators and ground truth are trusted evaluation artifacts.
"""
import argparse, hashlib, itertools, json, pickle, time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef, log_loss
from threadpoolctl import threadpool_limits
from revision.run_study import ROOT, SEEDS, EPS, FAMILIES, PRIMARY_PAIRS, read_mf, safe_json
from revision.run_bridge_study import partitions
from revision.kernel_function import KernelBridge, KernelMap, release_function, function_predict
from revision.public_bridge import fit_bridge
from revision.joint_release import release_joint
from revision.observable_transfer import predict_graph
from revision.prototype_certificate import prototype_radii
from revision.private_proxy_selection import select_proxies, predict_selected
from revision.freeze_round3 import freeze, DEST

METHODS=['FedKAPT-O','Linear-Graph','Kernel-Graph','Kernel-Nearest','Kernel-Moment','Public-Proxy']

def metrics(truth, probability, k, subjects=None):
    probability=np.maximum(probability,1e-15);probability/=probability.sum(1,keepdims=True)
    prediction=probability.argmax(1);confidence=probability.max(1);correct=prediction==truth
    ece=0.
    for lo in np.arange(0,1,.1):
        mask=(confidence>=lo)&(confidence<lo+.1 if lo<.9 else confidence<=1)
        if mask.any():ece+=mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    out={'accuracy':float(correct.mean()),'macro_f1':float(f1_score(truth,prediction,labels=np.arange(k),average='macro',zero_division=0)),
       'balanced_accuracy':float(balanced_accuracy_score(truth,prediction)), 'mcc':float(matthews_corrcoef(truth,prediction)),
       'nll':float(log_loss(truth,probability,labels=np.arange(k))),
       'brier':float(np.mean(np.sum((probability-np.eye(k)[truth])**2,axis=1))),
       'ece':float(ece),'n_test':len(truth),'n_correct':int(correct.sum())}
    if subjects is not None:
        out['per_subject_accuracy']={str(s):float(correct[subjects==s].mean()) for s in np.unique(subjects)}
        out['subject_macro_accuracy']=float(np.mean(list(out['per_subject_accuracy'].values())))
    return out

def evaluate(xs,xt,y,indices,eps,seed,k,model_path,subjects=None,variant='full',shuffle=False):
    started=time.perf_counter();fit,cal,src,tr,te=indices
    # Public resource caps apply identically to all controls; unused rows reserved.
    # Dataset files are sorted by activity/subject. A prefix would silently
    # restrict the public resource to early activities. Subsample WITHOUT
    # inspecting labels; keep already capped MF pools exactly unchanged.
    if len(fit)>320:fit=fit[np.random.default_rng(seed+83000).permutation(len(fit))[:320]]
    if len(cal)>80:cal=cal[np.random.default_rng(seed+83001).permutation(len(cal))[:80]]
    dimension=8 if eps<=1 else 32
    if variant.startswith('fixed'):dimension=int(variant[5:])
    dimension=min(dimension,64,len(fit)-1)
    ps,pt=xs[fit],xt[fit];cs,ct=xs[cal],xt[cal]
    if shuffle:
        pt=pt[np.random.default_rng(seed+123000).permutation(len(pt))]
        ct=ct[np.random.default_rng(seed+123001).permutation(len(ct))]
    weighted=KernelBridge(ps,pt,cs,ct,dimension=dimension,ridge=.001,weighted=True)
    if variant=='no_weight':weighted.weights[:]=1.
    source=weighted.source(xs[src]);xtr=weighted.target(xt[tr]);xte=weighted.target(xt[te])
    joint=release_joint(source,y[src],k,eps,seed+110000)
    parameters={'beta':0. if variant=='no_diffusion' else .2,'hard_teacher':variant=='hard_teacher'}
    main=predict_graph(xtr,xte,joint,**parameters)
    clean=release_joint(source,y[src],k,float('inf'),seed+110000)
    clean_output=predict_graph(xtr,xte,clean,certify=False,**parameters)
    # Clean artifacts exist only inside the research evaluator.
    pclean,_=clean.decode();pnoise,_=joint.decode()
    event=bool(np.all(np.linalg.norm(pclean.prototypes-pnoise.prototypes,axis=1)<=main['prototype_radii']+1e-12))
    probabilities={'FedKAPT-O':main['probabilities']}
    payload={'main_query':joint.raw,'main_clean_probabilities':clean_output['probabilities'],
       'main_lower':main['lower'],'main_upper':main['upper'],'main_accepted':main['accepted'],
       'main_certificate_gap':main['certificate_gap'],'prototype_radii':main['prototype_radii'],
       'y_test':y[te],'target_test_indices':te,'source_indices':src,'public_fit_indices':fit,'public_calibration_indices':cal,
       'main_target_train_codes':xtr,'main_target_test_codes':xte}
    extra={}
    if variant=='full':
        linear=fit_bridge(ps,pt,cs,ct,dimension=16,ridge=.1)
        ls=linear.source(xs[src]);ltr=linear.target(xt[tr]);lte=linear.target(xt[te])
        lj=release_joint(ls,y[src],k,eps,seed+110000)
        probabilities['Linear-Graph']=predict_graph(ltr,lte,lj,certify=False)['probabilities'];payload['linear_query']=lj.raw
        unweighted=KernelBridge(ps,pt,cs,ct,dimension=dimension,ridge=.001,weighted=False)
        us=unweighted.source(xs[src]);utr=unweighted.target(xt[tr]);ute=unweighted.target(xt[te])
        uj=release_joint(us,y[src],k,eps,seed+110000);um,_=uj.decode()
        probabilities['Kernel-Graph']=predict_graph(utr,ute,uj,certify=False)['probabilities'];payload['kernel_query']=uj.raw
        probabilities['Kernel-Nearest']=softmax(-cdist(ute,um.prototypes)**2/.1,axis=1)
        # DP-MERF-style moment release with public-Hessian function recovery.
        # It is explicitly an adaptation, not DP-MERF's generative algorithm.
        sm,tm=KernelMap(ps,64),KernelMap(pt,64)
        fq,fsigma=release_function(sm.transform(xs[src]),y[src],k,eps,seed+110000)
        scores,fd=function_predict(sm.public,tm.public,tm.transform(xt[tr]),tm.transform(xt[te]),fq,ridge=.01,variant='two_stage')
        probabilities['Kernel-Moment']=softmax(scores,axis=1);payload['moment_query']=fq
        payload['moment_source_coefficients']=fd['source_coefficients'];payload['moment_target_coefficients']=fd['target_coefficients']
        # Public-prototype exponential selection is DPPL-inspired; exact tuple
        # utility/sensitivity given in private_proxy_selection.py, not DPPL clone.
        proxy_source,proxy_target=weighted.source(ps),weighted.target(pt)
        selected,pmeta=select_proxies(source,y[src],proxy_source,k,eps,seed+112000)
        probabilities['Public-Proxy']=predict_selected(xtr,xte,proxy_target,selected)
        payload['proxy_indices']=selected;extra={'proxy_privacy':pmeta,'moment_sigma':fsigma}
        model={'main_bridge':weighted,'linear_bridge':linear,'unweighted_bridge':unweighted,
           'moment_source_map':sm,'moment_target_map':tm,'moment_coefficients':fd,
           'main_query':joint,'linear_query':lj,'unweighted_query':uj,'proxy_indices':selected}
    else:model={'main_bridge':weighted,'main_query':joint}
    for name,p in probabilities.items():payload[name+'_probability']=p
    result={name:metrics(y[te],p,k,subjects[te] if subjects is not None else None) for name,p in probabilities.items()}
    disagreement=main['predictions']!=clean_output['predictions'];accepted=main['accepted']
    out={'epsilon':eps,'seed':seed,'dimension':dimension,'source_n':len(src),'target_train_n':len(tr),'target_test_n':len(te),
       'public_fit_n':len(fit),'public_calibration_n':len(cal),'sigma_query':joint.sigma_query,'payload_bytes':joint.raw.nbytes,
       'weights':weighted.weights,'canonical':weighted.canonical,'results':result,
       'certificate':{'alpha':.05,'inclusion_event':event,'coverage':float(accepted.mean()),
           'accepted_n':int(accepted.sum()),'accepted_clean_disagreements':int((accepted&disagreement).sum()),
           'accepted_wrong_labels':int((accepted&(main['predictions']!=y[te])).sum()),
           'clean_disagreement_rate':float(disagreement.mean()),'meaning':'agreement with clean same pipeline; no semantic correctness guarantee'},
       'partition':dict(zip(['public_fit_reserved','public_calibration_reserved','source','target_train','target_test'],indices)),
       'runtime_seconds':time.perf_counter()-started,'variant':variant,**extra}
    # Append extensions: epsilon decimal points must not be treated as a
    # filename suffix, otherwise 0.1/0.5/1.0 and ablation variants collide.
    np.savez_compressed(str(model_path)+'.npz',**payload)
    with open(str(model_path)+'.pkl','wb') as f:pickle.dump(model,f,protocol=4)
    return out

def run(stage):
    config=freeze();folder=DEST/stage;folder.mkdir(exist_ok=True);path=DEST/(stage+'.jsonl')
    done={r['id']:r for r in (json.loads(l) for l in path.read_text().splitlines())} if path.exists() else {}
    if stage in ['mf','ablation','sensitivity','public_sizes','fixed_privacy']:
        views,y=read_mf();subjects=None;k=10
        pairs=list(itertools.permutations(FAMILIES,2)) if stage=='mf' else PRIMARY_PAIRS
        def get_parts(seed):return partitions(len(y),y,seed,400)
    elif stage=='har':
        from revision.har_data import load_har_raw,subject_partition
        views,y,subjects,_=load_har_raw();k=6;pairs=list(itertools.permutations(['accel','gyro'],2))
        def get_parts(seed):return list(subject_partition(subjects,seed)[0].values())
    elif stage=='daily':
        from revision.daily_sports_data import load_daily,daily_partition,VIEWS
        views,y,subjects,_=load_daily();k=19;pairs=list(itertools.permutations(VIEWS,2))
        def get_parts(seed):return list(daily_partition(subjects,seed)[0].values())
    elif stage=='clinical':
        w=np.loadtxt(ROOT/'data'/'wdbc'/'wdbc.data',delimiter=',',dtype=str);x=w[:,2:].astype(float);y=(w[:,1]=='M').astype(int)
        views={'mean':x[:,:10],'worst':x[:,20:30]};subjects=None;k=2;pairs=[('mean','worst')]
        def get_parts(seed):return partitions(len(y),y,seed,100,100)
    else:raise ValueError(stage)
    primary=set(map(tuple,config['new_daily_sports']['primary_directions'])) if stage=='daily' else set(PRIMARY_PAIRS)
    for a,b in pairs:
        for seed in SEEDS:
            indices=get_parts(seed)
            budgets=EPS if stage in ['har','clinical'] or (a,b) in primary else [2.]
            variants=['full']
            if stage=='ablation':budgets=[.5,2.,float('inf')];variants=['no_weight','no_diffusion','hard_teacher']
            if stage=='sensitivity':budgets=[2.];variants=['shuffled_pairs']
            if stage=='public_sizes':budgets=[2.];variants=['pool100','pool200']
            if stage=='fixed_privacy':budgets=EPS;variants=['fixed8','fixed32']
            for eps in budgets:
                for variant in variants:
                    ident=f'{stage}-{a}-{b}|{seed}|{eps}|{variant}'
                    file=folder/ident.replace('|','_')
                    if ident in done and file.parent.joinpath(file.name+'.npz').exists() and file.parent.joinpath(file.name+'.pkl').exists():continue
                    current_indices=partitions(len(y),y,seed,int(variant[4:]),400) if variant.startswith('pool') else indices
                    record=evaluate(views[a],views[b],y,current_indices,eps,seed,k,file,subjects,
                        variant='resource' if variant.startswith('pool') else ('full' if variant=='shuffled_pairs' else variant),shuffle=variant=='shuffled_pairs')
                    record.update(id=ident,dataset=f'{stage.upper()}-{a}-{b}',model_file=str(file.relative_to(DEST)),classes=k,
                        primary=stage in ['har','clinical'] or (a,b) in primary,privacy_unit='labelled provided descriptor/window replacement')
                    if ident in done:
                        for name in record['results']:
                            assert abs(record['results'][name]['accuracy']-done[ident]['results'][name]['accuracy'])<1e-12, ('Regeneration changed accuracy',ident,name)
                    else:
                        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(safe_json(record),allow_nan=False)+'\n')
                    print(ident,[(n,round(r['accuracy'],4)) for n,r in record['results'].items()],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['mf','har','daily','clinical','ablation','sensitivity','public_sizes','fixed_privacy'],required=True)
    args=p.parse_args()
    with threadpool_limits(limits=1):run(args.stage)
