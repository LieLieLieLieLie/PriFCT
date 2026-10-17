"""Prospectively frozen MotionSense evaluation; no outcome-based selection."""
import argparse,contextlib,hashlib,itertools,json,math,pickle,re,sys,time,zipfile,io
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from prifct.v7_public_bridge import fit_public_bridge
from prifct.v7_functional_bridge import fit_functional_bridge
from prifct.v7_semantic_transport import MultiScaleSignature
from prifct.v7_anchored_distribution import ProjectedSignature,component_means
from prifct.v15_maxent_teacher import fit as maxent,features
from prifct.v14_semantic_functions import public_operators,calibrated,log_probability,offset,predict,OPTIONS
from prifct.v7_public_quadratic import public_covariance
from prifct.v7_spectral_contrast import reconstruct
from prifct.v7_core_target import CoreTarget
from revision.joint_release import release_joint
from prifct.v6_development_pilot import score
from prifct.v7_core_checkpoints import digest
from prifct.v7_lossless_probability import pack

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/models/eswa_fedkapt'
OUT=BASE/'semantic_v47_uci_har'
DEV=BASE/'maxent_v19_development'
ZIP=ROOT/'data/uci_har/official_complete.zip'
ROSTER=['PriFCT','DP-HP','DP-NTK','DP-KIP','KPG-RL','m-PPOT','UniOT','UniJDOT','AGW','UCOOT']
DESIGNS=['Full maxent semantic transport','Ridge source teacher','No semantic calibration','Hard component teacher','No marginal coupling','Posterior hard component teacher']
SEEDS=[91331,91332,91333]

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def register():
    OUT.mkdir(parents=True,exist_ok=True)
    people=np.random.default_rng(91301).permutation(np.arange(1,31)).tolist()
    roles=dict(public=people[:6],source=people[6:16],adaptation=people[16:21],test=people[21:])
    choices=read(BASE/'original_v7_research/confirmation_v7/protocol.json')['selections']
    options={d:read(DEV/'DSA/summary.json')['rankings'][d][0]['options'] for d in DESIGNS}
    assert read(DEV/'retention_gate.json')['passes']
    protocol=dict(scope='Additional fixed-configuration UCI HAR participant evaluation; dataset was acquired earlier but these outcomes have not been used for model selection. Dataset extension follows mixed MotionSense results; no pristine project-wide holdout or unconditional superiority claim',
        rationale='Additional public benchmark on thirty participants and independent accelerometer/gyroscope coordinate groups; all original activity classes, no class selection or utility filtering',
        dataset_sha256=sha(ZIP),roles=roles,classes=6,directions=[['accel','gyro'],['gyro','accel']],methods=ROSTER,
        public='Uniform96 fitting and24 held windows per public person, independent of labels;256 target landmarks; six held-covariate candidate maps',
        source='Every finite source-person descriptor; no utility/class filtering or source subsampling',
        windows='Official128sample2.56second windows at50Hz with50percent overlap. All role separation by person, with person-level inference accounting for dependent windows; record-DP is descriptor-level and not raw-stream privacy',
        features='Three axes and magnitude; mean, populationSD, RMS, range and low/high positive-frequency power fractions =24 coordinates per group',
        modality='Official total accelerometer coordinates and body gyroscope coordinates. Distinct sensor columns; not statistical independence. No mixed561-coordinate features.',
        activity_catalog=['walking','walking_upstairs','walking_downstairs','sitting','standing','laying'],
        options=options,private_baselines={m:choices[m] for m in ROSTER[1:4]},context_baselines={m:choices[m] for m in ROSTER[4:]},
        selection='Proposed and removal parameters frozen from DSA development; all nine comparator settings frozen from complete MF development. No UCI HAR epoch/config tuning.',
        budgets=[2.,10.],seeds=SEEDS,primary='Mean supported-class balanced accuracy within each of9 test people, then directions/draws/budgets and people',
        statistical_unit='People; intervals conditional on the fixed source and public cohorts; not window/bootstrap independence',
        privacy='Complete labelled descriptor-record replacement at fixedN/K,delta1e-5; not person/raw-stream privacy. Nonprivate six labelled-source objectives explicitly separated.',
        missing='Failing fits remain in the ledger; no success-only mean or default replacement',
        code={p:sha(ROOT/p) for p in ['prifct/v47_uci_har.py','prifct/v15_maxent_teacher.py','prifct/v14_semantic_functions.py','prifct/v7_public_bridge.py','prifct/v7_functional_bridge.py']})
    pp=OUT/'protocol.json'
    if pp.exists():assert read(pp)==protocol
    else:pp.write_text(json.dumps(protocol,indent=2))
    return protocol

def descriptors(block):
    a=np.c_[block,np.linalg.norm(block,axis=1)]
    power=abs(np.fft.rfft(a-a.mean(0),axis=0))**2
    positive=power[1:];total=np.maximum(positive.sum(0),1e-20)
    low=positive[:len(positive)//2].sum(0)/total
    high=positive[len(positive)//2:].sum(0)/total
    return np.stack([a.mean(0),a.std(0),np.sqrt((a*a).mean(0)),np.ptp(a,axis=0),low,high],axis=1).ravel()

def data(protocol):
    cache=OUT/'descriptors.npz'
    if not cache.exists():
        groups={n:[] for n in ['accel','gyro','y','subjects','identities']}
        files={}
        with zipfile.ZipFile(ZIP) as outer:
            with zipfile.ZipFile(io.BytesIO(outer.read('UCI HAR Dataset.zip'))) as z:
                for si,split in enumerate(['train','test']):
                    prefix='UCI HAR Dataset/'+split+'/'
                    def read_array(name):
                        body=z.read(prefix+name);files[prefix+name]=hashlib.sha256(body).hexdigest()
                        return np.loadtxt(io.BytesIO(body))
                    y=read_array('y_'+split+'.txt').astype(int)-1
                    people=read_array('subject_'+split+'.txt').astype(int)
                    acc=np.stack([read_array('Inertial Signals/total_acc_'+axis+'_'+split+'.txt') for axis in 'xyz'],axis=2)
                    gyro=np.stack([read_array('Inertial Signals/body_gyro_'+axis+'_'+split+'.txt') for axis in 'xyz'],axis=2)
                    assert len(y)==len(acc)==len(gyro)==len(people)
                    for j in range(len(y)):
                        assert np.isfinite(acc[j]).all() and np.isfinite(gyro[j]).all()
                        groups['accel'].append(descriptors(acc[j]));groups['gyro'].append(descriptors(gyro[j]))
                        groups['y'].append(y[j]);groups['subjects'].append(people[j]);groups['identities'].append([people[j],si,j])
        arrays={n:np.array(v) for n,v in groups.items()};np.savez_compressed(cache,**arrays)
        (OUT/'extraction.json').write_text(json.dumps(dict(files=files,rows=len(arrays['y']),protocol_sha256=sha(OUT/'protocol.json'))))
        print('Extracted official UCI HAR windows',len(arrays['y']),'nonoverlapping windows',flush=True)
    with np.load(cache) as z:arrays={n:z[n] for n in z.files}
    arrays['roles']={r:np.flatnonzero(np.isin(arrays['subjects'],v)) for r,v in protocol['roles'].items()}
    rng=np.random.default_rng(91311);fit=[];held=[]
    for person in protocol['roles']['public']:
        eligible=np.flatnonzero(arrays['subjects']==person);assert len(eligible)>=120
        idx=rng.permutation(eligible);fit.extend(idx[:96]);held.extend(idx[96:120])
    arrays['fit']=np.array(fit);arrays['held']=np.array(held)
    return arrays

def maps(arrays,s,t):
    folder=OUT/'public_maps';folder.mkdir(exist_ok=True);file=folder/(s+'_'+t+'.pkl')
    if file.exists():
        with file.open('rb') as f:return pickle.load(f)
    fit=arrays['fit'];held=arrays['held'];sx=arrays[s][fit];tx=arrays[t][fit];sh=arrays[s][held];th=arrays[t][held]
    choices=[]
    for o in OPTIONS:
        point=fit_public_bridge(sx,tx,rank=32,landmarks=256,seed=91312,**o)
        error=float(np.mean(np.sum((point.target(th)-point.source(sh))**2,1)))
        choices.append((error,point))
    point=min(choices,key=lambda v:v[0])[1]
    original=MultiScaleSignature.public_fit(point.source(sx),seed=91313)
    signature=ProjectedSignature.public_fit(original,point.source(sx),rank=32)
    choices=[]
    for o in OPTIONS:
        full=fit_functional_bridge(point,signature,sx,tx,**o)
        error=float(np.mean(np.sum((full.target(th)-full.source(sh))**2,1)));choices.append((error,full))
    result=dict(point=point,full=min(choices,key=lambda v:v[0])[1])
    with file.open('wb') as f:pickle.dump(result,f,protocol=4)
    return result

def append(row,p,a,test,model):
    p=np.asarray(p,float);p/=p.sum(1,keepdims=True)
    folder=OUT/'runs';folder.mkdir(exist_ok=True)
    output=folder/(row['id']+'.npz')
    np.savez_compressed(output,y=a['y'][test],subjects=a['subjects'][test],identities=a['identities'][test],**pack(p))
    row.update(status='success',results=score(a['y'][test],p,a['subjects'][test],6),probability_sha256=digest(p),
        output_sha256=sha(output),model_sha256=sha(model),protocol_sha256=sha(OUT/'protocol.json'))
    with (OUT/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(row['id'],row['results']['subject_balanced_accuracy'],flush=True)

def proposed(protocol,a):
    file=OUT/'results.jsonl';done={json.loads(v)['id'] for v in file.read_text().splitlines()} if file.exists() else set()
    models=OUT/'models';models.mkdir(exist_ok=True);queries=OUT/'queries';queries.mkdir(exist_ok=True)
    source=a['roles']['source'];adapt=a['roles']['adaptation'];test=a['roles']['test']
    for s,t in protocol['directions']:
        m=maps(a,s,t);mapping=m['full'];point=m['point'];fit=a['fit'];held=a['held']
        sx=mapping.source(a[s][fit]);sh=mapping.source(a[s][held]);x=mapping.source(a[s][source])
        ax=mapping.target(a[t][adapt]);vx=mapping.target(a[t][test]);cov=public_covariance(sx)
        generic_geometry=CoreTarget.build(ax,vx,6,91321)
        operators=public_operators(point,a[t][fit],a[t][held],a[t][adapt],a[t][test])
        for e,seed in itertools.product(protocol['budgets'],SEEDS):
            qp=queries/f'{s}_{t}_e{e}_s{seed}.npz'
            if not qp.exists():
                packet=release_joint(x,a['y'][source],6,e,seed=seed)
                np.savez_compressed(qp,raw=packet.raw,sigma=packet.sigma_query,n_public=len(source))
            with np.load(qp) as z:raw=z['raw'];sigma=float(z['sigma'])
            q=np.maximum(raw[:,-1]*len(source),1.);q/=q.sum()
            for design,o in protocol['options'].items():
                ident=f'{s}_{t}_e{e}_s{seed}_{design.replace(" ","_")}'
                if ident in done:continue
                if design=='Ridge source teacher':
                    w,_=reconstruct(raw[:,:-1],cov,sigma,o['ridge'],no_denoising=True)
                    w=np.r_[w,np.zeros((1,6))];mu=np.zeros(sx.shape[1])
                else:w,mu,solver=maxent(sx,raw,len(source),o['ridge'])
                if design=='No semantic calibration':
                    ap=softmax(features(ax,mu)@w/o['teacher_temperature'],axis=1)
                    vp=softmax(features(vx,mu)@w/o['teacher_temperature'],axis=1);state=None
                else:
                    state=calibrated(operators,features(sx,mu),features(sh,mu),w,o['teacher_temperature'])
                    ap=state['adaptation'];vp=state['validation']
                hard=design in ['Hard component teacher','Posterior hard component teacher']
                geometry=CoreTarget.build(ap,vp,6,91321) if design=='Posterior hard component teacher' else generic_geometry
                cl=component_means(ap,geometry.labels,len(geometry.centers))
                scores=log_probability(cl) if hard else log_probability(ap)
                v,diagnostic=offset(scores,o['receiver_temperature'],o['balance_strength'],q,geometry.target_mass if hard else None)
                scores=log_probability(cl)[geometry.test_responsibilities.argmax(1)] if hard else log_probability(vp)
                p=predict(scores,v,o['receiver_temperature']);cp=models/(ident+'.pkl')
                with cp.open('wb') as f:pickle.dump(dict(weight=w,mean=mu,semantic_state=None if state is None else
                    {k:state[k] for k in ['coefficients','selected','public_errors']},offset=v,options=o,
                    geometry_centers=geometry.centers,component_probability=cl,query_sha256=sha(qp),design=design),f,protocol=4)
                append(dict(id=ident,case_id=s+'_'+t,method='PriFCT' if design==DESIGNS[0] else design,design=design,
                    epsilon=str(e),seed=seed,privacy='complete-record DP',solver=diagnostic),p,a,test,cp)

def private_baselines(protocol,a):
    from prifct.v7_streamed_private_features import release
    from prifct.v6_private_generation import PrivateFeatureMap
    from prifct.v7_generator_weighted import fit_receiver
    from prifct.v7_dp_ntk import NTKFeatureMap
    from prifct.v7_large_cohort_kip import fit as fit_kip
    from prifct.v7_dp_kip import predict as predict_kip
    source=a['roles']['source'];test=a['roles']['test'];models=OUT/'models';models.mkdir(exist_ok=True)
    for s,t in protocol['directions']:
        m=maps(a,s,t)
        for method,selection in protocol['private_baselines'].items():
            o=selection['options'];mapping=m['full' if o['representation']=='function' else 'point']
            x=mapping.source(a[s][source]);xt=mapping.target(a[t][test])
            if method!='DP-KIP':
                pub=mapping.source(a[s][a['fit']]);chosen=np.random.default_rng(91351).permutation(len(pub))[:256]
                d=np.sum((pub[chosen,None,:]-pub[None,chosen,:])**2,axis=2);width=max(float(np.median(d[np.triu_indices(256,1)])),1e-4)
                feature=PrivateFeatureMap.hermite(x.shape[1],width,order=o['order'],product_order=o['product_order']) if method=='DP-HP' else NTKFeatureMap(x.shape[1],6,hidden=o['hidden'],seed=77271)
            for e,seed in itertools.product(protocol['budgets'],SEEDS):
                ident=f'{s}_{t}_{method}_e{e}_s{seed}'
                rows=[json.loads(v) for v in (OUT/'results.jsonl').read_text().splitlines()]
                if any(v['id']==ident for v in rows):continue
                cp=models/(ident+'.pkl')
                if method=='DP-KIP':
                    settings={k:v for k,v in o.items() if k not in ['representation','initialization','temperature','calibration']}
                    scale=1. if o['initialization']=='native_gaussian' else 1/math.sqrt(x.shape[1])
                    model=fit_kip(x,a['y'][source],6,epsilon=e,seed=seed,initialization_scale=scale,accountant='gdp',**settings)
                    p=predict_kip(model,xt,o['temperature'])
                    with cp.open('wb') as f:pickle.dump(model,f,protocol=4)
                else:
                    packet=release(x,a['y'][source],6,feature,e,seed=seed,device='cuda')
                    weights=np.ones(packet.raw.shape[1]-1)
                    if method=='DP-HP':weights[x.shape[1]*(o['order']+1):]=o['gamma']
                    generator=fit_receiver(packet,feature,steps=o['steps'],hidden=128,samples_per_class=16,seed=77281,device='cuda',loss_weights=weights)
                    synthetic,labels=generator.sample(n_per_class=128,seed=77291)
                    classifier=LogisticRegression(C=1.,multi_class='multinomial',max_iter=1000).fit(synthetic,labels)
                    p=classifier.predict_proba(xt)
                    with cp.open('wb') as f:pickle.dump(dict(classifier=classifier,synthetic=synthetic,labels=labels,
                        feature=(dict(kind=feature.kind,dimension=feature.dimension,classes=feature.classes,hidden=feature.hidden,seed=feature.seed,author_sha256=feature.author_sha256,network_state={n:v.detach().cpu() for n,v in feature.network.state_dict().items()}) if hasattr(feature,'network') else feature),packet=packet,generator_state=generator.generator.state_dict(),options=o),f,protocol=4)
                    del generator
                p=p.astype(np.float32).astype(float);p/=p.sum(1,keepdims=True)
                append(dict(id=ident,case_id=s+'_'+t,method=method,epsilon=str(e),seed=seed,
                    privacy='complete-record DP',options=o),p,a,test,cp)

def context(protocol,a):
    sys.path.insert(0,str(ROOT/'prifct/author_baselines_v6'))
    from unidabench_core import fit as train,predict as infer,save_checkpoint
    from prifct.v7_native_structural import kpg,barycentric_classifier
    from prifct.v7_native_structural_checked import agw
    from prifct.v7_native_structural_checkpoint import ucoot
    src=a['roles']['source'];adapt=a['roles']['adaptation'];test=a['roles']['test'];models=OUT/'models'
    for s,t in protocol['directions']:
        m=maps(a,s,t)
        for method,selection in protocol['context_baselines'].items():
            ident=s+'_'+t+'_'+method;rows=[json.loads(v) for v in (OUT/'results.jsonl').read_text().splitlines()]
            if any(r['id']==ident for r in rows):continue
            o=selection['options'];mapping=m['full' if o['representation']=='function' else 'point']
            x=mapping.source(a[s][src]);tx=mapping.target(a[t][adapt]);xt=mapping.target(a[t][test])
            if method in ['m-PPOT','UniOT','UniJDOT']:
                internal='PPOT' if method=='m-PPOT' else method;cp=models/(ident+'.pt')
                with (models/(ident+'.log')).open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    model,diagnostic=train(internal,x,a['y'][src],tx,6,hparams=o['hparams'],seed=91361,hidden=o['hidden'])
                    result=infer(model,internal,xt,inference_seed=91362)
                p=result['probabilities'];save_checkpoint(cp,model,diagnostic);rejection=float(result['reject'].mean());del model
            else:
                settings={k:v for k,v in o.items() if k not in ['representation','classifier_C']}
                if method=='AGW':plan,diagnostic=agw(x,tx,**settings)
                elif method=='UCOOT':plan,diagnostic=ucoot(x,tx,**settings)
                else:
                    indices=np.random.default_rng(91363).permutation(a['fit'])[:32]
                    plan,diagnostic=kpg(x,tx,mapping.source(a[s][indices]),mapping.target(a[t][indices]),**settings)
                classifier,head=barycentric_classifier(plan,tx,a['y'][src],classes=6,C=o['classifier_C'],seed=91364)
                p=classifier.predict_proba(xt);cp=models/(ident+'.pkl');rejection=0.
                with cp.open('wb') as f:pickle.dump(dict(classifier=classifier,diagnostics=diagnostic,head=head,options=o),f,protocol=4)
                del plan
            append(dict(id=ident,case_id=s+'_'+t,method=method,epsilon='none',seed=91361,
                privacy='NONPRIVATE labelled source',options=o,native_rejection_rate=rejection),p,a,test,cp)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['proposed','private','context']);mode=parser.parse_args().mode
    protocol=register();a=data(protocol)
    if mode=='proposed':proposed(protocol,a)
    elif mode=='private':private_baselines(protocol,a)
    else:context(protocol,a)

if __name__=='__main__':
    with threadpool_limits(1):main()
