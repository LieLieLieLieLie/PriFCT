"""Equal decision-preserving probability calibration on MF development only."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from prifct.v14_semantic_functions import predict
from prifct.v7_lossless_probability import load,decode
from prifct.v7_mf_reserved_confirmation import choose_row

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/models/eswa_fedkapt'
OLD=BASE/'original_v7_research'
GRID=[dict(temperature=t,uniform_mass=a) for t,a in itertools.product([.5,1.,2.],[0.,.01,.03,.1,.3])]

def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(v) for v in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def apply(p,option):
    p=np.asarray(p,float)
    if option==dict(temperature=1.,uniform_mass=0.):return p
    value=softmax(np.log(np.maximum(p,1e-30))/option['temperature'],axis=1)
    value=(1-option['uniform_mass'])*value+option['uniform_mass']/p.shape[1]
    value/=value.sum(1,keepdims=True)
    assert np.array_equal(value.argmax(1),p.argmax(1)),'Calibration must preserve every decision'
    return value

def nll(p,y):return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())

def main(branch):
    development=BASE/branch;out=BASE/'semantic_probability_calibration'/branch;out.mkdir(parents=True,exist_ok=True)
    assert (development/'MF/summary.json').exists() and (development/'DSA/summary.json').exists()
    manifest=dict(scope='MF validation labels only, all30 directions/both budgets, no reserved/WISDM/MotionSense utility',
        rule='Each proposed/removal/comparator receives the same15 decision-preserving temperature/uniform-smoothing options; choose NLL only after its predictive hyperparameters freeze',
        accuracy='Every transformation is checked to preserve all argmax decisions; never chooses a predictive checkpoint',
        transfer='Proposed/removals evaluated on MF at their separately frozen DSA predictive parameters; comparators use their independently selected MF states',
        grid=GRID,development_sha256={d:sha(development/d/'summary.json') for d in ['MF','DSA']},
        comparator_registry_sha256=sha(OLD/'confirmation_v7/protocol.json'),code_sha256=sha(Path(__file__)))
    file=out/'protocol.json'
    if file.exists():assert read(file)==manifest
    else:file.write_text(json.dumps(manifest,indent=2))
    mf=read(development/'MF/summary.json')['rankings'];dsa=read(development/'DSA/summary.json')['rankings']
    records=rows(development/'MF/results.jsonl');core={}
    for mode,rankings in [('MF',mf),('transfer',dsa)]:
        core[mode]={}
        for design,candidates in rankings.items():
            options=candidates[0]['options'];selected=[r for r in records if r['design']==design and r['options']==options]
            assert len(selected)==60 and all(r['status']=='success' for r in selected)
            probs=[]
            for row in selected:
                with np.load(development/row['target']) as target,np.load(development/row['model']) as model:
                    if bool(model['hard']):logs=target['component_log'][target['hard_assignment']]
                    else:logs=target['validation_log']
                    p=predict(logs,model['offset'],float(model['temperature']))
                with np.load(development/row['truth']) as truth:y=truth['y']
                probs.append((p,y))
            core[mode][design]=probs
    registry=read(OLD/'confirmation_v7/protocol.json')['selections'];baseline={}
    for method,selection in registry.items():
        if method=='FedKAPT':continue
        folder=OLD/'multiple_features_development'/selection['folder'];ledger=rows(folder/'results.jsonl')
        cases=sorted({r['case_id'] for r in records});items=[]
        for case in cases:
            for e in ([2.,10.] if method.startswith('DP-') else [None]):
                internal='PPOT' if method=='m-PPOT' else method
                row=choose_row(ledger,case,selection['options'],internal,e)
                with load(folder/'runs'/(row['id']+'.npz')) as saved:items.append((decode(saved),saved['y']))
        baseline[method]=items
    result={}
    for mode,groups in core.items():
        result[mode]={}
        for method,items in dict(groups,**baseline).items():
            ranking=[]
            for option in GRID:
                scores=[nll(apply(p,option),y) for p,y in items]
                ranking.append(dict(options=option,nll=float(np.mean(scores)),cases=len(scores)))
            ranking.sort(key=lambda r:(r['nll'],r['options']['uniform_mass'],abs(np.log(r['options']['temperature']))))
            result[mode][method]=dict(selected=ranking[0],ranking=ranking)
    (out/'summary.json').write_text(json.dumps(dict(protocol_sha256=sha(file),settings=result,all_decisions_preserved=True,
        external_probabilities_opened=False),indent=2))
    print(json.dumps({mode:{m:r['selected'] for m,r in value.items()} for mode,value in result.items()},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('branch')
    with threadpool_limits(1):main(p.parse_args().branch)
