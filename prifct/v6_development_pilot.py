"""Single mechanism family, MHEALTH-only development; no external evaluation.

Protocol and code are frozen before scores. Controls share public pair IDs,
bounded maps, source query, target clusters and validation access. Repeated
noise draws on three validation people are not population confirmation.
"""
import hashlib
import itertools
import json
import pickle
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from prifct.v6_mhealth_data import ROOT,OUT as CACHE,cohort
from prifct.v6_public_relation import fit_public_relation,squared_distance
from prifct.v6_robust_transport import robust_transport
from prifct.v6_robust_numerics import audit_robust_transport
from prifct.method import TargetProblem,fixed_transport
from prifct.confirm_development import refine
from revision.joint_release import release_joint
from revision.prototype_certificate import prototype_radii
from revision.run_observable_study import metrics

OUT=ROOT/'results/models/eswa_fedkapt/original_v6_research/development_pilot'
DIRECTIONS=[('ankle_accel','ankle_gyro'),('ankle_gyro','ankle_accel'),
            ('wrist_accel','wrist_gyro'),('wrist_gyro','wrist_accel')]
SEEDS=[75101,75102,75103]
TEMPERATURES=[.001,.01,.05]
METHODS=['FedKAPT-shared','Point-OT','Edge-OT','DP-NCM','DP-Refine']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    dependencies=['prifct/v6_development_pilot.py','prifct/v6_mhealth_data.py',
        'prifct/v6_public_relation.py','prifct/v6_robust_transport.py','prifct/v6_robust_numerics.py',
        'prifct/transport_numerics.py','prifct/method.py','prifct/confirm_development.py',
        'revision/joint_release.py','revision/prototype_certificate.py','core/privacy.py',
        'revision/run_observable_study.py']
    config={'registered_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'single class-shared noise adversary pilot; MHEALTH development only',
        'directions':DIRECTIONS,'budgets':[2.,10.],'seeds':SEEDS,'source_condition':'full',
        'public_pairs':'uniform fixed128 per public person; class-blind; separate64/person residual diagnostic',
        'pair_sampling_seed':75013,'landmarks':64,'rank':8,'map':'known public kernel CCA',
        'cca_ridge':.01,'target_clusters':'2K=24','target_fit':'adaptation people only',
        'row_margin':'positive target KMeans occupancy','class_margin':'uniform classes, fixedmass.95+reject.05',
        'tau_candidates':TEMPERATURES,'ncm_cost_temperature_candidates':TEMPERATURES,
        'refine_candidates':[{'temperature':4*t,'strength':s,'steps':10,'graph':False,
                             'style':'constant','transport_mix':0.} for t,s in itertools.product(TEMPERATURES,[.25,.5])],
        'transport_decoder':'target test soft cluster responsibilities times conditional class teacher; no graph/private source access',
        'alpha':.05,'uncertainty':'unshrunk simultaneous decoder radii, noise-only',
        'selection':'independent per method, equal case mean of per-person supported-class balanced accuracy; tie macro-F1 then NLL',
        'retention_gate':{'each_budget_gain_vs_strongest_NCM_Refine_pp':2.,
            'each_budget_gain_vs_point_OT_pp':1.,'worst_direction_allowed_degradation_pp':2.},
        'failure':'retain failure; do not evaluate WISDM or widen this pilot grid after scores',
        'privacy_unit':'fixed2s labelled-window replace-one, N/Kpublic, not subject; public benchmark seeds not deployment',
        'code':{name:sha(ROOT/name) for name in dependencies},
        'cache':{p.name:sha(p) for p in CACHE.glob('*.npz')},
        'feature_protocol_sha256':sha(CACHE/'feature_protocol.json')}
    path=OUT/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text());existing.pop('registered_utc');config.pop('registered_utc')
        assert existing==config,'Frozen protocol/code changed; do not overwrite scored version'
    else:
        path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        (OUT/'protocol.sha256').write_text(sha(path)+'\n')
    return json.loads(path.read_text())


def score(y,p,subjects,k):
    result=metrics(y,p,k,subjects);by_subject={}
    for subject in np.unique(subjects):
        mask=subjects==subject;ys=y[mask];pred=p[mask].argmax(1)
        available=np.unique(ys)
        recall=[float((pred[ys==c]==c).mean()) for c in available]
        by_subject[str(subject)]={'balanced_accuracy':float(np.mean(recall)),
            'classes_present':available.tolist(),'windows':int(mask.sum())}
    result['per_subject_balanced']=by_subject
    result['subject_balanced_accuracy']=float(np.mean([v['balanced_accuracy'] for v in by_subject.values()]))
    return result


def teacher(plan,a,k):
    conditional=plan/np.asarray(a)[:,None]
    result=conditional[:,:k].copy();mass=result.sum(1)
    result/=np.maximum(mass[:,None],1e-15)
    empty=mass<=1e-15;result[empty]=1/k
    return result,conditional,np.flatnonzero(empty)


def prepared(source,target,seed):
    ps,_,ss,pi=cohort('public',source);pt,_,st,ti=cohort('public',target)
    np.testing.assert_array_equal(pi,ti);np.testing.assert_array_equal(ss,st)
    pair_ids=[];calibration_ids=[]
    rng=np.random.default_rng(75013)
    for subject in np.unique(ss):
        indices=rng.permutation(np.flatnonzero(ss==subject))
        pair_ids.extend(indices[:128]);calibration_ids.extend(indices[128:192])
    pair_ids=np.asarray(pair_ids);calibration_ids=np.asarray(calibration_ids)
    mapping=fit_public_relation(ps[pair_ids],pt[pair_ids],mode='cca',landmarks=64,
        rank=8,ridge=.01,seed=75013)
    xs,ys,subjects_source,source_ids=cohort('source',source)
    adaptation,_,subjects_adaptation,adaptation_ids=cohort('adaptation',target)
    test,y,subjects,validation_ids=cohort('validation',target)
    for a,b in itertools.combinations([set(ss),set(subjects_source),set(subjects_adaptation),set(subjects)],2):assert not a&b
    sx=mapping.source(xs);tx=mapping.target(adaptation);testx=mapping.target(test)
    problem=TargetProblem.build(tx,testx,12,np.array([],int),np.array([],int),seed,
        comparable=True,clusters=2,prepared=True)
    weights=np.maximum(np.bincount(problem.labels,minlength=24),1)
    return {'source':sx,'source_y':ys,'problem':problem,'validation_y':y,'validation_subjects':subjects,
        'source_ids':source_ids,'adaptation_ids':adaptation_ids,'validation_ids':validation_ids,
        'public_pair_ids':pi[pair_ids],'public_calibration_ids':pi[calibration_ids],
        'mapping':mapping,'bridge_residual':mapping.calibration(ps[calibration_ids],pt[calibration_ids]),
        'target_integer_weights':weights}


def main():
    config=register();path=OUT/'results.jsonl';runs=OUT/'runs';runs.mkdir(exist_ok=True)
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done={row['id'] for row in rows};protocol_hash=sha(OUT/'protocol.json')
    for source,target in DIRECTIONS:
        for seed in SEEDS:
            data=prepared(source,target,seed);problem=data['problem'];weights=data['target_integer_weights']
            for epsilon in [2.,10.]:
                query=release_joint(data['source'],data['source_y'],12,epsilon,seed+31720)
                prototypes,radii,radius_details=prototype_radii(query,.05)
                cost=squared_distance(problem.centers,prototypes)/4
                difference=prototypes[None,:,:]-problem.centers[:,None,:]
                edge_cost=cost+(2*radii[None,:]*np.linalg.norm(difference,axis=2)+radii[None,:]**2)/4
                jobs=[(m,{'temperature':t}) for m,t in itertools.product(METHODS[:-1],TEMPERATURES)]
                jobs += [('DP-Refine',options) for options in config['refine_candidates']]
                for method,options in jobs:
                    ident=f'{source}_{target}_s{seed}_e{epsilon}_{method}_'+sha_text(options)[:12]
                    if ident in done:continue
                    started=time.perf_counter();details={}
                    if method=='DP-Refine':p,details=refine(problem,query,options)
                    elif method=='DP-NCM':p=softmax(-squared_distance(problem.xtest,prototypes)/(4*options['temperature']),axis=1)
                    elif method=='FedKAPT-shared':
                        result=robust_transport(problem.centers,prototypes,radii,mass=.95,tau=options['temperature'],target_mass=weights)
                        p=problem.test_responsibilities@result.teacher
                        numerical=audit_robust_transport(problem.centers,prototypes,radii,result.plan,
                            result.adversary,result.dual_row,result.dual_column,mass=.95,tau=options['temperature'],
                            target_mass=weights,conditional_plan=result.conditional_plan)
                        details={'result':result,'numerical':numerical}
                    else:
                        plan,_=fixed_transport(edge_cost if method=='Edge-OT' else cost,.95,options['temperature'],target_mass=weights)
                        soft,conditional,empty=teacher(plan,weights/weights.sum(),12)
                        p=problem.test_responsibilities@soft
                        details={'plan':plan,'conditional':conditional,'teacher':soft,'reject_only_rows':empty}
                    p=np.maximum(p,1e-12);p/=p.sum(1,keepdims=True)
                    evaluation=score(data['validation_y'],p,data['validation_subjects'],12)
                    with (runs/(ident+'.pkl')).open('wb') as stream:pickle.dump({'data':data,'query':query,
                        'options':options,'method':method,'details':details},stream,protocol=4)
                    np.savez_compressed(runs/(ident+'.npz'),probability=p,y_validation=data['validation_y'],
                        subjects=data['validation_subjects'],query=query.raw,radii=radii,
                        source_ids=data['source_ids'],adaptation_ids=data['adaptation_ids'],
                        validation_ids=data['validation_ids'],public_pair_ids=data['public_pair_ids'])
                    row={'id':ident,'source':source,'target':target,'seed':seed,'epsilon':epsilon,
                        'method':method,'options':options,'results':evaluation,
                        'radius_min':float(radii.min()),'radius_max':float(radii.max()),
                        'saturated_radii':int((radii>=2).sum()),'runtime_seconds':time.perf_counter()-started,
                        'protocol_sha256':protocol_hash,'target_training_labels':0,
                        'numerical_status':details.get('numerical',{}).get('status')}
                    with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                    rows.append(row);done.add(ident)
                print(source,target,'seed',seed,'epsilon',epsilon,'all candidates complete',flush=True)
    summarize(rows)


def sha_text(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def summarize(rows):
    expected=4*3*2*(4*3+6)
    assert len(rows)==expected and len({r['id'] for r in rows})==expected
    ranking={};chosen={}
    for method in METHODS:
        rr=[r for r in rows if r['method']==method];candidates={sha_text(r['options']):r['options'] for r in rr}
        values=[]
        for key,options in candidates.items():
            records=[r for r in rr if sha_text(r['options'])==key]
            values.append({'options':options,'cases':len(records),
                'subject_balanced_accuracy':float(np.mean([r['results']['subject_balanced_accuracy'] for r in records])),
                'macro_f1':float(np.mean([r['results']['macro_f1'] for r in records])),
                'nll':float(np.mean([r['results']['nll'] for r in records]))})
        values.sort(key=lambda r:(-r['subject_balanced_accuracy'],-r['macro_f1'],r['nll']))
        ranking[method]=values;chosen[method]=values[0]['options']
    selected=[r for r in rows if r['options']==chosen[r['method']]]
    effects={};gate={}
    for epsilon in [2.,10.]:
        means={m:float(np.mean([r['results']['subject_balanced_accuracy'] for r in selected if r['method']==m and r['epsilon']==epsilon])) for m in METHODS}
        strongest=max(['DP-NCM','DP-Refine'],key=lambda m:means[m])
        directions={}
        for source,target in DIRECTIONS:
            per_method={m:float(np.mean([r['results']['subject_balanced_accuracy'] for r in selected
                if r['method']==m and r['epsilon']==epsilon and r['source']==source and r['target']==target])) for m in METHODS}
            directions[source+'->'+target]=100*(per_method['FedKAPT-shared']-per_method[strongest])
        effects[str(epsilon)]={'mean_subject_balanced_accuracy':means,'strongest_control':strongest,
            'gain_vs_control_pp':100*(means['FedKAPT-shared']-means[strongest]),
            'gain_vs_point_pp':100*(means['FedKAPT-shared']-means['Point-OT']),
            'direction_gain_vs_control_pp':directions}
        gate[str(epsilon)]={'gain_vs_strongest_at_least2pp':effects[str(epsilon)]['gain_vs_control_pp']>=2.,
            'gain_vs_point_at_least1pp':effects[str(epsilon)]['gain_vs_point_pp']>=1.,
            'all_directions_no_more_than2pp_worse':min(directions.values())>=-2.}
    result={'version':'single shared-prototype adversary development pilot','rows':len(rows),
        'cases':24,'selected':chosen,'rankings':ranking,'effects':effects,'gate':gate,
        'gate_passed':all(all(v.values()) for v in gate.values()),
        'inference':'Development selection on three fixed validation people. Not independent population confirmation.',
        'protocol_sha256':sha(OUT/'protocol.json')}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'gate_passed':result['gate_passed'],'effects':effects},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
