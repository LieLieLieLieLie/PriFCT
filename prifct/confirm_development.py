"""Small predeclared development grid, excluding every external test identity."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import hashlib,itertools,json,time
from datetime import datetime,timezone
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from prifct.confirm_protocol import OUT,ROOT,case,freeze_roles,DEV_SEEDS,DIRECTIONS
from prifct.final_revision import CONFIG
from prifct.method import TargetProblem,gpu_sqdist
from prifct.denoise import catalogue,CatalogueQuery
from prifct.original_final import run_control
from prifct.dp_linear_head import fit_private_head,PUBLIC_GRID
from revision.joint_release import release_joint
from revision.protocol import clip
from revision.run_observable_study import metrics

FED_GRID=[dict(transport_mix=t,style=s,graph=g,temperature=.01,steps=10,strength=.5)
          for t,s,g in itertools.product([.25,.5],['constant','count_gate','count_gate_and_pull'],[False,True])]
REF_GRID=[dict(transport_mix=0.,style=s,graph=g,temperature=t,steps=10,strength=.5)
          for s,t,g in itertools.product(['constant','count_pull'],[.01,.05,.1],[False,True])]
CONTROL_GRID=[dict(rho=r,route=route,prototype_temperature=t) for r,route,t in itertools.product([.5,1.],[0.,.5,1.],[.01,.1])]

def refine(problem,query,options,config=None,teacher=None):
    config=CONFIG if config is None else config
    source,counts=query.decode();anchor=source.prototypes;centers=anchor.copy()
    uncertainty=query.n_public*query.sigma_query*np.sqrt(anchor.shape[1]+1)/np.maximum(counts,1.)
    precision=1/(1+uncertainty**2)
    if options['transport_mix']:
        if teacher is None:teacher=problem.predict(query,**config)['teacher']
        transported=problem.responsibilities@teacher
        gate=transported@precision if options['style']!='constant' else np.ones(len(problem.x))
        mixing=options['transport_mix']*gate[:,None]
    else:transported=np.zeros((len(problem.x),problem.k));mixing=np.zeros((len(problem.x),1))
    source_pull=np.full(problem.k,1-options['strength'])
    if options['style'] in ['count_pull','count_gate_and_pull']:source_pull*=precision
    history=[]
    for _ in range(options['steps']):
        direct=softmax(-gpu_sqdist(problem.x,centers)/options['temperature'],axis=1)
        initial=mixing*transported+(1-mixing)*direct
        smooth=problem.propagate(initial,steps=30) if options['graph'] else initial
        p=.5*smooth+.5*direct;weights=p**2;denominator=weights.sum(0)
        target=weights.T@problem.x/np.maximum(denominator[:,None],1e-12)
        target[denominator<1]=anchor[denominator<1]
        update=clip(source_pull[:,None]*anchor+(1-source_pull[:,None])*target)
        history.append({'max_update':float(np.linalg.norm(update-centers,axis=1).max()),'max_drift':float(np.linalg.norm(update-anchor,axis=1).max()),'inactive_classes':int((denominator<1).sum())})
        centers=update
    direct=softmax(-gpu_sqdist(problem.x,centers)/options['temperature'],axis=1)
    initial=mixing*transported+(1-mixing)*direct
    direct_test=softmax(-gpu_sqdist(problem.xtest,centers)/options['temperature'],axis=1)
    if options['graph']:pred=problem.test_transition@problem.propagate(initial,steps=30)
    else:
        if options['transport_mix']:
            target_mix=options['transport_mix']*(problem.test_responsibilities@teacher@precision)[:,None] if options['style']!='constant' else options['transport_mix']
            pred=target_mix*(problem.test_responsibilities@teacher)+(1-target_mix)*direct_test
        else:pred=direct_test
    return .5*pred+.5*direct_test,{'centers':centers,'precision_proxy':precision,'source_pull':source_pull,'history':history,'mean_transport_weight':float(np.mean(mixing))}

def prepared(name,a,b,seed,condition,development):
    d=case(name,a,b,seed,condition,development)
    problem=TargetProblem.build(d['train'],d['test'],d['k'],np.array([],int),np.array([],int),seed,True,prepared=True)
    book=catalogue(d['public'],d['k'])
    return d,problem,book

def preregister():
    freeze_roles()
    spec={'date_utc':datetime.now(timezone.utc).isoformat(),'version':'review-round1-v5',
       'FedKAPT_candidates':FED_GRID,'DP-Refine_candidates':REF_GRID,'mechanism_candidates':CONTROL_GRID,
       'DP-Linear_candidates':PUBLIC_GRID,'development_directions':DIRECTIONS,'development_seeds':DEV_SEEDS,
       'development_conditions':['full','quarter','LT10'],'development_epsilon':2.,
       'development_primary':'equal case-weighted accuracy; tie break macro-F1 then NLL; no external labels',
       'external_primary':'Office31 all six directions, epsilon2, 20 noise/clustering repeats; first equal-average directions within each repeat, then single two-sided paired exact sign-flip contrast FedKAPT minus DP-Refine',
       'practical_effect_accuracy':.01,'significance':.05,'population':'fixed cohorts conditional noise/clustering sensitivity; not independent domain sampling',
       'source_conditions':['full','quarter','half','LT10'],'stress_directions':DIRECTIONS,'stress_epsilon':[.5,2.,10.],
       'rank_rule':'nominal128 visual/16HAR; actual source-public SVD available rank copied to target without label-based adaptation',
       'precision_proxy':'1/(1+(N*sigma_query*sqrt(d+1)/max(noisy_class_count,1))^2), heuristic not certificate',
       'margins':'unchanged public uniform target/catalogue and class margins; no count-dependent marginals',
       'role_manifest_sha256':(OUT/'identity_roles.sha256').read_text().strip(),
       'external_selection_forbidden':True,'v4_status':'preserved exploratory; its Holm family is not reduced retrospectively'}
    path=OUT/'preregistration.json'
    if not path.exists():
        data=json.dumps(spec,indent=2);path.write_bytes(data.encode('utf8'))
        (OUT/'preregistration.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    else:
        old=json.loads(path.read_text(encoding='utf8'));old.pop('date_utc');spec.pop('date_utc');assert old==spec

def main():
    preregister();file=OUT/'development.jsonl';done={json.loads(l)['id'] for l in file.read_text().splitlines()} if file.exists() else set()
    for name,a,b in [('OfficeHome',a,b) for a,b in DIRECTIONS]+[('HAR','accel','accel')]:
      conditions=['full','quarter','LT10'] if name=='OfficeHome' else ['full']
      for seed,condition in itertools.product(DEV_SEEDS,conditions):
        d,problem,book=prepared(name,a,b,seed,condition,True)
        base=release_joint(d['source'],d['source_y'],d['k'],2.,seed+91710)
        query=CatalogueQuery(base,book,.5,.2)
        teacher=problem.predict(query,**CONFIG)['teacher']
        jobs=[('FedKAPT',i,c) for i,c in enumerate(FED_GRID)]+[('DP-Refine',i,c) for i,c in enumerate(REF_GRID)]+[('DP-Linear',i,c) for i,c in enumerate(PUBLIC_GRID)]
        jobs += [(m,i,c) for m in ['Rel-FullOT','Rel-UOT','EGW-Private'] for i,c in enumerate(CONTROL_GRID)]
        jobs += [('DP-NCM',i,{'prototype_temperature':t}) for i,t in enumerate([.01,.05,.1])]
        for method,i,c in jobs:
            ident=f'{name}_{a}_{b}_{seed}_{condition}_{method}_{i}'
            if ident in done:continue
            start=time.perf_counter()
            if method in ['FedKAPT','DP-Refine']:p,detail=refine(problem,query,c,teacher=teacher)
            elif method=='DP-Linear':
                head=fit_private_head(d['source'],d['source_y'],d['k'],2.,seed=seed+71920,**c);p=head.predict_proba(d['test']);detail={'accounting':head.accounting}
            else:
                cfg=dict(CONFIG,**{key:value for key,value in c.items() if key!='route'});p=run_control(problem,query,method,cfg);detail={}
                if 'route' in c:
                    msg,_=query.decode();direct=softmax(-gpu_sqdist(problem.xtest,msg.prototypes)/cfg['prototype_temperature'],axis=1)
                    p=c['route']*p+(1-c['route'])*direct
            row={'id':ident,'dataset':name,'source':a,'target':b,'seed':seed,'condition':condition,'method':method,'candidate':c,
              'source_n':len(d['source']),'results':metrics(d['test_y'],p,d['k'],d['subjects_test']),'runtime_seconds':time.perf_counter()-start}
            with file.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        print(name,a,b,seed,condition,'development complete',flush=True)
    rows=[json.loads(l) for l in file.read_text().splitlines()];selected={};ranking={}
    for name in ['OfficeHome','HAR']:
      selected[name]={};ranking[name]={}
      for method in ['FedKAPT','DP-Refine','DP-Linear','DP-NCM','Rel-FullOT','Rel-UOT','EGW-Private']:
        rr=[r for r in rows if r['dataset']==name and r['method']==method]
        candidates={json.dumps(r['candidate'],sort_keys=True) for r in rr};summary=[]
        for candidate in candidates:
            cases=[r for r in rr if json.dumps(r['candidate'],sort_keys=True)==candidate]
            summary.append({'candidate':json.loads(candidate),'cases':len(cases),'accuracy':float(np.mean([r['results']['accuracy'] for r in cases])),
             'macro_f1':float(np.mean([r['results']['macro_f1'] for r in cases])),'nll':float(np.mean([r['results']['nll'] for r in cases]))})
        summary.sort(key=lambda r:(-r['accuracy'],-r['macro_f1'],r['nll'],json.dumps(r['candidate'],sort_keys=True)))
        selected[name][method]=summary[0]['candidate'];ranking[name][method]=summary
    (OUT/'development_rankings.json').write_text(json.dumps(ranking,indent=2))
    (OUT/'selected_config.json').write_text(json.dumps(selected,indent=2))
    print('Development selection done; external scores not accessed.',json.dumps(selected),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
