"""Inference only on300 reserved records, after all10 configurations freeze.

No model fitting, configuration search or source query is permitted here.
Baseline model states are the exact independently selected development states.
All main predictions and six removals plus direct control are saved for paired
record-level analysis across the correlated30 family directions.
"""
import json,pickle,itertools,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from prifct.v7_confirmation_registry import main as freeze,ROOT,BASE,MF,OUT,ROSTER
from prifct.v7_multiple_features import load as load_views,roles,FAMILIES
from prifct.v7_core_target import CoreTarget
from prifct.v7_core_checkpoints import probability,digest
from prifct.v7_lossless_probability import pack,load as load_saved,decode
from prifct.v6_development_pilot import score
from prifct.v7_dp_kip import predict as predict_kip


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def records(folder):return [json.loads(v) for v in (folder/'results.jsonl').read_text().splitlines()]


def choose_row(rows,case,options,method=None,epsilon=None):
    candidates=[r for r in rows if r['case_id']==case and r['options']==options and
        (method is None or r['method']==method) and (epsilon is None or r['epsilon']==epsilon)]
    if len(candidates)!=1 or candidates[0]['status']!='success':raise RuntimeError('Selected complete configuration missing: '+str((case,options,method,epsilon)))
    return candidates[0]


def rounded(p):
    p=np.maximum(p,1e-12);p/=p.sum(1,keepdims=True)
    p=p.astype(np.float32).astype(np.float64);p/=p.sum(1,keepdims=True)
    return p


def native_rounded(p):
    # Exactly the baseline development precision sequence, without inserting
    # the core's extra numerical floor into a named comparator.
    p=p.astype(np.float32).astype(np.float64);p/=p.sum(1,keepdims=True)
    return p


def main():
    protocol=freeze();pp=OUT/'protocol.json';assert all(sha(ROOT/p)==h for p,h in protocol['frozen_code'].items())
    assert sha(pp)==(OUT/'protocol.sha256').read_text().strip()
    run=OUT/'MF_reserved';run.mkdir(exist_ok=True);saved=run/'runs';saved.mkdir(exist_ok=True)
    file=run/'results.jsonl';rows=records(run) if file.exists() else [];done={r['id'] for r in rows}
    views,y,raw_hashes=load_views();partition=roles();reserved=partition['reserved'];adapt=partition['adaptation'];subjects=np.zeros(len(reserved),int)
    assert len(reserved)==300 and set(reserved).isdisjoint(partition['validation'])
    folders={v['folder'] for v in protocol['selections'].values() if not v['folder'].startswith('..')}
    development={folder:records(MF/folder) for folder in folders}
    core_rows=records(BASE/'core_mix_refinement/MF');direct_rows=records(BASE/'direct_core_control/MF')
    sys.path.insert(0,str(ROOT/'prifct/author_baselines_v6'))
    from unidabench_core import load_checkpoint,predict as predict_context
    expected=30*(4*2+6+6*2+2+2) #10 main algorithms,6 removals,direct,count-only control.
    for s,t in itertools.permutations(FAMILIES,2):
        case=s+'_'+t
        with (MF/'spectral_contrast/public_maps'/(case+'.pkl')).open('rb') as stream:maps=pickle.load(stream)
        point=maps['point'];full=maps['full'];unprojected=maps['unprojected']
        prepared={k:mp.target(views[t][reserved]) for k,mp in [('point',point),('function',full),('unprojected',unprojected)]}
        target_inputs=dict(full=prepared['function'],linear=prepared['point'],point=full.signature.transform(prepared['point']),unprojected=prepared['unprojected'])
        adaptation_inputs=dict(full=full.target(views[t][adapt]),linear=point.target(views[t][adapt]),
            point=full.signature.transform(point.target(views[t][adapt])),unprojected=unprojected.target(views[t][adapt]))
        targets={}
        for key in target_inputs:
            p=CoreTarget.build(adaptation_inputs[key],target_inputs[key],10,77021)
            targets[key]=dict(xtest=p.xtest,test_responsibilities=p.test_responsibilities,y=y[reserved],subjects=subjects)
        def append(ident,method,group,epsilon,options,p,evidence):
            if ident in done:return
            assert np.isfinite(p).all() and p.shape==(300,10) and np.allclose(p.sum(1),1,atol=1e-12)
            np.savez_compressed(saved/(ident+'.npz'),y=y[reserved],subjects=subjects,record_indices=reserved,**pack(p))
            row=dict(id=ident,case_id=case,method=method,group=group,epsilon=epsilon,options=options,status='success',
                results=score(y[reserved],p,subjects,10),probability_sha256=digest(p),protocol_sha256=sha(pp),evidence=evidence)
            with file.open('a') as stream:stream.write(json.dumps(row)+'\n')
            rows.append(row);done.add(ident)
        for method,options in protocol['ablations']['MF'].items():
            label='FedKAPT' if method=='FedKAPT-candidate' else method;group='primary_private' if label=='FedKAPT' else 'mechanism'
            for e in protocol['main_budgets']:
                ident=f'{case}_{label.replace(" ","_")}_e{e}'
                if ident in done:continue
                chosen=choose_row(core_rows,case,options,method,e);checkpoint=ROOT/chosen['parent_model']
                assert sha(checkpoint)==chosen['parent_model_sha256']
                with np.load(checkpoint) as f:state={n:f[n] for n in f.files}
                state['transport_mix']=options['transport_mix'];p=probability(state,targets[chosen['target_key']])
                append(ident,label,group,e,options,p,dict(development_row=chosen['id'],model_sha256=sha(checkpoint),source='Immutable selected core state; only new record-local responsibilities'))
        for e in protocol['main_budgets']:
            ident=case+'_Counts_only_e'+str(e)
            if ident in done:continue
            chosen=choose_row(core_rows,case,protocol['ablations']['MF']['FedKAPT-candidate'],'FedKAPT-candidate',e)
            checkpoint=ROOT/chosen['parent_model']
            with np.load(checkpoint) as state:q=state['class_prior']
            count_p=native_rounded(np.broadcast_to(q,(300,10)).copy())
            append(ident,'No class function moments','mechanism',e,dict(information='Same released class prior only'),count_p,
                dict(development_row=chosen['id'],source='Constant released-prior classifier; source/target class-function scores completely erased'))
        options=protocol['direct']['MF']
        for e in protocol['main_budgets']:
            ident=f'{case}_Direct_e{e}'
            if ident in done:continue
            chosen=choose_row(direct_rows,case,options,epsilon=e);checkpoint=BASE/'direct_core_control/MF/models'/(chosen['id']+'.npz')
            with np.load(checkpoint) as f:p=rounded(softmax(prepared['function']@f['weight']/float(f['temperature']),axis=1))
            append(ident,'No transport or component interpolation','mechanism',e,options,p,dict(development_row=chosen['id'],model_sha256=sha(checkpoint)))
        for method in ROSTER[1:]:
            selection=protocol['selections'][method];folder=MF/selection['folder'];options=selection['options']
            eps=protocol['main_budgets'] if method.startswith('DP-') else [None]
            for e in eps:
                ident=f'{case}_{method}_e{e}'
                if ident in done:continue
                internal='PPOT' if method=='m-PPOT' else method
                chosen=choose_row(development[selection['folder']],case,options,internal,e)
                evidence=dict(development_row=chosen['id'],development_protocol_sha256=chosen['protocol_sha256'])
                if method in ['KPG-RL','AGW','UCOOT']:
                    checkpoint=folder/'models'/(chosen['id']+'.pkl')
                    with checkpoint.open('rb') as stream:model=pickle.load(stream)
                    representation=options.get('representation','raw')
                    target=point.target_scaler.transform(views[t][reserved])/np.sqrt(FAMILIES[t]) if representation=='raw' else prepared[representation]
                    p=native_rounded(model['classifier'].predict_proba(target))
                elif method in ['m-PPOT','UniOT','UniJDOT']:
                    checkpoint=folder/'runs'/(chosen['id']+'.pt');model,_=load_checkpoint(checkpoint)
                    output=predict_context(model,internal,prepared[options['representation']],inference_seed=77232)
                    # Retain native posterior normalization without float32 re-rounding, matching development.
                    p=output['probabilities'].astype(np.float64);p/=p.sum(1,keepdims=True)
                    evidence['native_rejection_rate']=float(output['reject'].mean());del model
                elif method=='DP-KIP':
                    if 'temperature' in options:base=chosen['id'].rsplit('_t',1)[0];temp=options['temperature']
                    else:base=chosen['id'].rsplit('_',1)[0];temp=options['calibration']
                    checkpoint=folder/'models'/(base+'.pkl')
                    with checkpoint.open('rb') as stream:model=pickle.load(stream)
                    p=native_rounded(predict_kip(model,prepared[options['representation']],temp))
                else:
                    checkpoint=folder/'models'/(chosen['id']+'_classifier.pkl')
                    with checkpoint.open('rb') as stream:model=pickle.load(stream)
                    p=native_rounded(model.predict_proba(prepared[options['representation']]))
                evidence['model_sha256']=sha(checkpoint)
                append(ident,method,'primary_private' if e is not None else 'nonprivate_context',e,options,p,evidence)
        print(case,len(rows),'/',expected,'reserved predictions',flush=True)
    assert len(rows)==expected==900
    assert set(r['method'] for r in rows if r['group']!='mechanism')==set(ROSTER)
    assert len({r['id'] for r in rows})==expected
    checks=0
    for r in rows:
        with load_saved(saved/(r['id']+'.npz')) as f:p=decode(f);actual=score(f['y'],p,f['subjects'],10)
        for metric in ['accuracy','macro_f1','balanced_accuracy','mcc','nll','brier','ece','subject_balanced_accuracy']:
            assert actual[metric]==r['results'][metric];checks+=1
    report=dict(outcomes=expected,failures=0,primary_methods=ROSTER,primary_outcomes=420,mechanism_outcomes=480,
        exact_metric_replay_checks=checks,max_metric_difference=0.,protocol_sha256=sha(pp),
        reserved_records=300,raw_sha256=raw_hashes,scope='Frozen state conditional record holdout, NOT independent writer/source/noise population')
    (run/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
