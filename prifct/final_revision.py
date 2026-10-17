"""Frozen catalogue-calibrated FedKAPT candidate; original target contract.

The completed original_v4 evaluation is reported with matched controls and
explicit limitations; frozen settings must not be retuned on evaluation scores.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,json,itertools,pickle,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from prifct.calibrated_data import case
from prifct.data import load,SEEDS,EPS,FAMILIES
from prifct.method import TargetProblem,gpu_sqdist,certify_plan,DEVICE
from prifct.denoise import catalogue,CatalogueQuery
from prifct.original_final import run_control,METHODS,PRIMARY
from revision.joint_release import release_joint
from revision.run_observable_study import metrics
from revision.run_study import safe_json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/models/eswa_fedkapt/original_v4'
CONFIG=dict(relational_weight=1.,semantic_weight=0.,mass=.95,tau=.01,blend=1.,ordered=False,graph_mix=1.,rho=.5,prototype_mix=.25,prototype_temperature=.1)
POSTPROCESS={'mixing':.5,'width':.2,'graph_route':.5,'office_rank':128,'other_rank':16,'image_mode':'raw_center'}
REFINEMENT=dict(strength=.5,steps=10,temperature=.01,threshold=0.,transport_mix=.5)
METHODS=METHODS+['DP-Refine']

@lru_cache(maxsize=32)
def build(n,a,b,seed):
    d=case(n,a,b,seed,rank=128 if n=='Office' else 16,image_mode='raw_center')
    views,_,_,_=load(n);public=d['source_map'].transform(views[a][d['public_source_indices']])
    codebook=catalogue(public,d['k'])
    problem=TargetProblem.build(d['train'],d['test'],d['k'],np.array([],int),np.array([],int),seed,d['comparable'],prepared=True)
    return d,problem,codebook

def probability(problem,query,config,route=.5,variant='full'):
    output=problem.predict(query,variant='hard' if variant=='hard' else 'full',**config)
    if variant=='no_graph':
        cluster=problem.test_responsibilities@output['teacher']
        if problem.comparable:
            msg,_=query.decode();direct=softmax(-gpu_sqdist(problem.xtest,msg.prototypes)/config['prototype_temperature'],axis=1)
            output['probability']=(1-config['prototype_mix'])*cluster+config['prototype_mix']*direct
        else:output['probability']=cluster
    if problem.comparable:
        msg,_=query.decode();direct=softmax(-gpu_sqdist(problem.xtest,msg.prototypes)/config['prototype_temperature'],axis=1)
        output['probability']=route*output['probability']+(1-route)*direct
    return output

def revision_probability(n,problem,query,config,route=.5,variant='full'):
    output=probability(problem,query,config,route,variant)
    if n=='Office' and variant not in ['no_condition','relational_only']:
        from prifct.target_refinement import refine
        options=REFINEMENT.copy()
        if variant=='no_transport':options['transport_mix']=0.
        if variant=='no_graph':options['graph']=False
        if variant=='no_refinement':options['steps']=0
        if variant.startswith('pull'):options['strength']=float(variant[4:])
        if variant.startswith('steps'):options['steps']=int(variant[5:])
        p,detail=refine(problem,query,config,hard=variant=='hard',**options)
        output['probability']=p;output['refinement']=detail
    return output

def evaluate(n,a,b,seed,eps,variant='full',clusters=1):
    start=time.perf_counter();d,problem,book=build(n,a,b,seed)
    if clusters!=1:
        problem=TargetProblem.build(d['train'],d['test'],d['k'],np.array([],int),np.array([],int),seed,d['comparable'],clusters=clusters,prepared=True)
    base=release_joint(d['source'],d['source_y'],d['k'],eps,seed+110000)
    mixing=0. if variant=='no_catalogue' else POSTPROCESS['mixing']
    query=CatalogueQuery(base,book,mixing,POSTPROCESS['width'])
    config=CONFIG.copy();route=POSTPROCESS['graph_route']
    if not d['comparable']:config.update(rho=0.,prototype_mix=0.)
    if variant=='no_condition':config['prototype_mix']=0.;route=1.
    if variant=='no_transport':config['prototype_mix']=1.
    if variant=='relational_only':config.update(rho=0.,prototype_mix=0.);route=1.
    if variant.startswith('rho'):config['rho']=float(variant[3:])
    if variant.startswith('mass'):config['mass']=float(variant[4:])
    if variant.startswith('length'):config['signature_length']=int(variant[6:])
    output=revision_probability(n,problem,query,config,route,variant)
    probabilities={'FedKAPT':output['probability']}
    tuning_file=OUT/'baseline_config.json'
    baseline_config=json.loads(tuning_file.read_text()) if tuning_file.exists() else {}
    if variant=='full' and d['comparable']:
        for method in METHODS[1:]:
            c=baseline_config.get(method,{}).get('config',CONFIG)
            mix=baseline_config.get(method,{}).get('mixing',POSTPROCESS['mixing'])
            q=CatalogueQuery(base,book,mix,POSTPROCESS['width'])
            if method=='DP-Refine':
                from prifct.target_refinement import refine
                opt=baseline_config.get(method,{}).get('refinement',dict(REFINEMENT,transport_mix=0.))
                p=refine(problem,q,c,**opt)[0] if n=='Office' else run_control(problem,q,'DP-NCM',c)
                cal=baseline_config.get(method,{}).get('calibration',1.) if n=='Office' else 1.
                p=softmax(np.log(np.maximum(p,1e-12))/cal,axis=1)
            else:p=run_control(problem,q,method,c)
            if method not in ['DP-NCM','DP-Graph','DP-Refine']:
                r=baseline_config.get(method,{}).get('route',POSTPROCESS['graph_route'])
                message,_=q.decode();direct=softmax(-gpu_sqdist(problem.xtest,message.prototypes)/c['prototype_temperature'],axis=1)
                p=r*p+(1-r)*direct
            probabilities[method]=p
    clean_base=release_joint(d['source'],d['source_y'],d['k'],float('inf'),seed+110000)
    clean_query=CatalogueQuery(clean_base,book,mixing,POSTPROCESS['width'])
    clean_output=revision_probability(n,problem,clean_query,config,route,variant)
    raw_probabilities={m:p.copy() for m,p in probabilities.items()};clean_raw=clean_output['probability'].copy()
    calibration_file=OUT/'probability_calibration_config.json'
    calibration=json.loads(calibration_file.read_text()) if n=='Office' and calibration_file.exists() and variant!='no_calibration' else {}
    if calibration:
        from prifct.tune_probability_calibration import calibrate
        probabilities={m:calibrate(p,**calibration[m]) for m,p in probabilities.items()}
        clean_output['probability']=calibrate(clean_output['probability'],**calibration['FedKAPT'])
    cert=certify_plan(problem,query,output,config['tau'],config['relational_weight'],rho=config['rho'])
    winner=output['conditional_plan'].argmax(1);cleanwinner=clean_output['conditional_plan'].argmax(1)
    truth=np.array([np.bincount(d['train_y'][problem.labels==c],minlength=d['k']).argmax() for c in range(len(problem.centers))])
    ident=f'{n}_{a}_{b}_s{seed}_e{eps}_{variant}_c{clusters}'
    folder=OUT/'runs';folder.mkdir(parents=True,exist_ok=True)
    with (folder/(ident+'.pkl')).open('wb') as f:pickle.dump({'problem':problem,'query':query,'parameters':config,'route':route,'refinement':REFINEMENT if n=='Office' else None,'probability_calibration':calibration,'source_map':d['source_map'],'target_map':d['target_map']},f,protocol=4)
    np.savez_compressed(folder/(ident+'.npz'),**{m+'_probability':p for m,p in probabilities.items()},**{m+'_raw_probability':p for m,p in raw_probabilities.items()},clean_raw_probability=clean_raw,
        y_test=d['test_y'],source_indices=d['source_indices'],train_indices=d['train_indices'],test_indices=d['test_indices'],public_indices=d['public_source_indices'],
        query=base.raw,clean_query=clean_base.raw,codebook=book,plan=output['plan'],clean_plan=clean_output['plan'],clean_probability=clean_output['probability'],
        train_cluster_truth=truth,train_cluster_winner=winner,certified_rows=cert['accepted_rows'],
        refined_centers=output.get('refinement',{}).get('centers',np.empty((0,d['source'].shape[1]))))
    msg,_=query.decode();clearm,_=clean_query.decode()
    return safe_json({'id':ident,'dataset':n,'source':a,'target':b,'seed':seed,'epsilon':eps,'variant':variant,'clusters':clusters,
        'source_n':len(d['source']),'target_train_n':len(d['train']),'target_test_n':len(d['test']),'public_calibration_n':len(d['public_source_indices']),
        'source_dimension':d['source'].shape[1],'target_semantic_training_labels':0,'comparable':d['comparable'],'probability_calibration':calibration,'query_bytes':base.raw.nbytes,
        'public_setup_bytes':int(book.nbytes+d['source_map'].projection.nbytes+sum(v.nbytes for v in [d['source_map'].scaler.mean_,d['source_map'].scaler.scale_] if v is not None)),
        'results':{m:metrics(d['test_y'],p,d['k'],d['subjects_test']) for m,p in probabilities.items()},'certificate':cert,
        'certified_clean_disagreements':int((cert['accepted_rows']&(winner!=cleanwinner)).sum()),
        'prototype_error':float(np.mean(np.linalg.norm(msg.prototypes-clearm.prototypes,axis=1))),
        'row_semantic_accuracy':float((winner==truth).mean()),'runtime_seconds':time.perf_counter()-start,'device':DEVICE})

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen={'version':'original-catalogue-4.0','config':CONFIG,'postprocess':POSTPROCESS,'refinement':REFINEMENT,'epsilon':safe_json(EPS),'seeds':SEEDS,
      'main_information':'No target training labels; no paired public bridge; independent source-view public calibration, target-local standardisation, one joint labelled-record Gaussian query',
      'selection':'Source catalogue prior/temperature/routing and image raw-centred public PCA128/anchored target refinement validated on seeds1701-1703, strict held-out target-training validation; rho=.5 retains original mixture as an explicit design choice; historical pilot/test results observed, exploratory research',
      'privacy':'Known public class universe and N, full feature-label replace-one sensitivity2/N, C=1, delta1e-5; not subject/participation DP',
      'scope':'Corresponding descriptor positions required for main direct cost. MF/HAR/DSA zero-anchor strict-view results are stress tests, not a successful general feature-disjoint claim.'}
    f=OUT/'frozen_config.json'
    if f.exists():assert json.loads(f.read_text())==frozen
    else:f.write_text(json.dumps(frozen,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['main','privacy','ablation','scope','scope_nonprivate','cardinality','sensitivity'],default='main');args=parser.parse_args();freeze()
    file=OUT/(args.stage+'.jsonl');done={json.loads(l)['id'] for l in file.read_text().splitlines()} if file.exists() else set()
    if args.stage=='main':
        directions=[('WDBC',a,b) for a,b in [('mean','mean'),('worst','worst'),('mean','worst'),('worst','mean')]]+[('CWRU',a,b) for a,b in itertools.permutations(map(str,range(4)),2)]+[('Office',a,b) for a,b in itertools.permutations(['art','clipart','product','real_world'],2)]
        jobs=[(n,a,b,s,2.,'full',1) for n,a,b in directions for s in SEEDS]
    elif args.stage=='privacy':jobs=[(n,a,b,s,e,'full',1) for n,a,b in PRIMARY for s in SEEDS for e in EPS]
    elif args.stage=='ablation':jobs=[(n,a,b,s,2.,v,1) for n,a,b in PRIMARY for s in SEEDS for v in ['no_catalogue','no_condition','no_transport','hard','no_graph','relational_only','rho0.0','rho0.25','rho0.75','rho1.0','mass0.5','mass0.8','mass1.0','no_refinement','no_calibration']]
    elif args.stage=='cardinality':jobs=[(n,a,b,s,2.,'full',c) for n,a,b in PRIMARY for s in SEEDS for c in [2,3]]
    elif args.stage=='sensitivity':jobs=[(n,a,b,s,2.,v,1) for n,a,b in PRIMARY for s in SEEDS for v in ['length4','length8','length32','pull0.0','pull0.25','pull0.75','pull1.0','steps3','steps20']]
    else:jobs=[(n,a,b,s,float('inf') if args.stage=='scope_nonprivate' else 2.,'full',1) for n,a,b in [('MF',a,b) for a,b in itertools.permutations(FAMILIES,2)]+[('HAR','accel','gyro'),('HAR','gyro','accel'),('DSA','torso','right_arm'),('DSA','right_arm','torso')] for s in SEEDS]
    for i,(n,a,b,s,e,v,c) in enumerate(jobs):
        ident=f'{n}_{a}_{b}_s{s}_e{e}_{v}_c{c}'
        if ident in done:continue
        row=evaluate(n,a,b,s,e,v,c)
        with file.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(args.stage,i+1,'/',len(jobs),ident,round(row['results']['FedKAPT']['accuracy'],4),flush=True)
if __name__=='__main__':
    with threadpool_limits(1):main()
