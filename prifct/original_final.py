"""Original FedKAPT architecture, explicit fixed source calibration and soft
prototype condition. No public pairs or target semantic training labels.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,json,itertools,pickle,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import ot
from prifct.calibrated_data import case
from prifct.method import TargetProblem,fixed_transport,gpu_sqdist,certify_plan,DEVICE
from prifct.data import SEEDS,EPS,FAMILIES
from revision.joint_release import release_joint
from revision.run_observable_study import metrics
from revision.run_study import safe_json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/models/eswa_fedkapt/original_v2'
CONFIG=dict(relational_weight=1.,semantic_weight=0.,mass=.95,tau=.01,blend=1.,ordered=False,graph_mix=1.,rho=1.,prototype_mix=.25,prototype_temperature=.1)
METHODS=['FedKAPT','DP-NCM','DP-Graph','Rel-FullOT','Rel-UOT','EGW-Private']
PRIMARY=[('WDBC','mean','mean'),('WDBC','mean','worst'),('CWRU','0','2'),('CWRU','2','0'),('Office','art','clipart'),('Office','product','real_world')]

def run_control(problem,joint,name,config):
    msg,_=joint.decode();temp=config['prototype_temperature']
    direct=softmax(-gpu_sqdist(problem.x,msg.prototypes)/temp,axis=1)
    if name=='DP-NCM':return softmax(-gpu_sqdist(problem.xtest,msg.prototypes)/temp,axis=1)
    if name=='DP-Graph':return problem.test_transition@problem.propagate(direct)
    if name=='Rel-FullOT':
        c=config.copy();c['mass']=1.;return problem.predict(joint,**c)['probability']
    cost=problem.cost(joint,config['relational_weight'],0.,False,config['rho'])
    a=np.full(len(cost),1/len(cost));b=np.full(problem.k,1/problem.k)
    if name=='Rel-UOT':plan=ot.unbalanced.sinkhorn_unbalanced(a,b,cost,config['tau'],.1,numItermax=3000)
    elif name=='EGW-Private':
        from prifct.stable_fgw import solve
        plan,_=solve(cost,cdist(problem.centers,problem.centers)/2,cdist(msg.prototypes,msg.prototypes)/2,
                     tau=config['tau'],alpha=.5)
    else:raise ValueError(name)
    assert np.isfinite(plan).all() and np.min(plan.sum(1))>1e-15
    teacher=plan/plan.sum(1,keepdims=True)
    initial=(1-config['prototype_mix'])*(problem.responsibilities@teacher)+config['prototype_mix']*direct
    return problem.test_transition@problem.propagate(initial)

def evaluate(n,a,b,seed,eps,variant='full',clusters=1):
    start=time.perf_counter();d=case(n,a,b,seed)
    problem=TargetProblem.build(d['train'],d['test'],d['k'],np.array([],int),np.array([],int),seed,d['comparable'],clusters=clusters,prepared=True)
    joint=release_joint(d['source'],d['source_y'],d['k'],eps,seed+110000)
    config=CONFIG.copy()
    if not d['comparable']:config.update(rho=0.,prototype_mix=0.)
    if variant.startswith('rho'):config['rho']=float(variant[3:])
    if variant.startswith('mass'):config['mass']=float(variant[4:])
    if variant=='no_condition':config['prototype_mix']=0.
    if variant=='no_transport':config['prototype_mix']=1.
    output=problem.predict(joint,variant='hard' if variant=='hard' else 'full',**config)
    if variant=='no_graph':
        teacher=output['teacher'];msg,_=joint.decode()
        output['probability']=(1-config['prototype_mix'])*(problem.test_responsibilities@teacher)+config['prototype_mix']*softmax(-gpu_sqdist(problem.xtest,msg.prototypes)/config['prototype_temperature'],axis=1)
    probabilities={'FedKAPT':output['probability']}
    if variant=='full' and d['comparable']:
        for method in METHODS[1:]:probabilities[method]=run_control(problem,joint,method,CONFIG)
    clean=release_joint(d['source'],d['source_y'],d['k'],float('inf'),seed+110000)
    clean_output=problem.predict(clean,**config)
    certificate=certify_plan(problem,joint,output,config['tau'],config['relational_weight'],rho=config['rho'])
    truth=np.array([np.bincount(d['train_y'][problem.labels==c],minlength=d['k']).argmax() for c in range(len(problem.centers))])
    winner=output['conditional_plan'].argmax(1);cleanwinner=clean_output['conditional_plan'].argmax(1)
    ident=f'{n}_{a}_{b}_s{seed}_e{eps}_{variant}_c{clusters}'
    folder=OUT/'runs';folder.mkdir(parents=True,exist_ok=True)
    state={'problem':problem,'query':joint,'parameters':config,'source_map':d['source_map'],'target_map':d['target_map']}
    with (folder/(ident+'.pkl')).open('wb') as f:pickle.dump(state,f,protocol=4)
    np.savez_compressed(folder/(ident+'.npz'),**{m+'_probability':p for m,p in probabilities.items()},
        y_test=d['test_y'],source_indices=d['source_indices'],train_indices=d['train_indices'],test_indices=d['test_indices'],
        public_indices=d['public_source_indices'],query=joint.raw,plan=output['plan'],clean_plan=clean_output['plan'],
        clean_probability=clean_output['probability'],train_cluster_truth=truth,
        train_cluster_winner=winner,certified_rows=certificate['accepted_rows'])
    return safe_json({'id':ident,'dataset':n,'source':a,'target':b,'seed':seed,'epsilon':eps,'variant':variant,'clusters':clusters,
        'source_n':len(d['source']),'target_train_n':len(d['train']),'target_test_n':len(d['test']),
        'public_calibration_n':len(d['public_source_indices']),'source_dimension':d['source'].shape[1],
        'target_semantic_training_labels':0,'comparable':d['comparable'],'query_bytes':joint.raw.nbytes,
        'results':{m:metrics(d['test_y'],p,d['k'],d['subjects_test']) for m,p in probabilities.items()},
        'certificate':certificate,'certified_clean_disagreements':int((certificate['accepted_rows']&(winner!=cleanwinner)).sum()),
        'row_semantic_accuracy':float((winner==truth).mean()),'runtime_seconds':time.perf_counter()-start,'device':DEVICE})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['main','privacy','ablation','scope','cardinality'],default='main');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    frozen={'version':'original-unlabelled-target-2.0','parameters':CONFIG,'primary_epsilon':2.,'seeds':SEEDS,'budgets':safe_json(EPS),
        'information':'one joint Gaussian source message; independent public source-view calibration; no paired bridge or target training labels',
        'development':'combined local-calibration and full65 Office development; all failures retained; historical datasets observed',
        'privacy':'full labelled-record replace-one, fixed N,K, C=1, delta=1e-5; conditional on public source map',
        'comparison':'DP-NCM, DP-Graph, full/UOT and EGW mathematical message controls; not claimed faithful top-journal systems',
        'scope':'declared descriptor correspondence is main; strict disjoint MF/HAR/DSA are boundary tests'}
    path=OUT/'frozen_config.json'
    if path.exists():assert json.loads(path.read_text())==frozen
    else:path.write_text(json.dumps(frozen,indent=2))
    file=OUT/(args.stage+'.jsonl');done={json.loads(l)['id'] for l in file.read_text().splitlines()} if file.exists() else set()
    if args.stage=='main':
        directions=[('WDBC',a,b) for a,b in [('mean','mean'),('worst','worst'),('mean','worst'),('worst','mean')]]
        directions += [('CWRU',a,b) for a,b in itertools.permutations(map(str,range(4)),2)]
        directions += [('Office',a,b) for a,b in itertools.permutations(['art','clipart','product','real_world'],2)]
        jobs=[(n,a,b,s,2.,'full',1) for n,a,b in directions for s in SEEDS]
    elif args.stage=='privacy':jobs=[(n,a,b,s,e,'full',1) for n,a,b in PRIMARY for s in SEEDS for e in EPS]
    elif args.stage=='ablation':jobs=[(n,a,b,s,2.,v,1) for n,a,b in PRIMARY for s in SEEDS for v in ['no_condition','no_transport','hard','no_graph','rho0.0','rho0.5','mass0.5','mass0.8','mass1.0']]
    elif args.stage=='cardinality':jobs=[(n,a,b,s,2.,'full',c) for n,a,b in PRIMARY for s in SEEDS for c in [2,3]]
    else:
        directions=[('MF',a,b) for a,b in itertools.permutations(FAMILIES,2)]+[('HAR','accel','gyro'),('HAR','gyro','accel'),('DSA','torso','right_arm'),('DSA','right_arm','torso')]
        jobs=[(n,a,b,s,2.,'full',1) for n,a,b in directions for s in SEEDS]
    for i,(n,a,b,s,e,v,c) in enumerate(jobs):
        ident=f'{n}_{a}_{b}_s{s}_e{e}_{v}_c{c}'
        if ident in done:continue
        row=evaluate(n,a,b,s,e,v,c)
        with file.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(args.stage,i+1,'/',len(jobs),ident,round(row['results']['FedKAPT']['accuracy'],4),flush=True)
if __name__=='__main__':
    with threadpool_limits(1):main()
