"""All methods receive the same development-only probability calibration.

Monotone temperature and a uniform floor preserve every class decision.
The candidate is not adopted or written to the active calibration config here.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import json,itertools
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from prifct.calibrated_data import case,FrozenMap
from prifct.data import load,DEVELOPMENT_SEEDS
from prifct.method import TargetProblem,gpu_sqdist
from prifct.denoise import catalogue,CatalogueQuery
from prifct.final_revision import OUT,CONFIG,POSTPROCESS,REFINEMENT,METHODS,revision_probability
from prifct.original_final import run_control
from prifct.target_refinement import refine
from revision.joint_release import release_joint
from revision.run_observable_study import metrics

def calibrate(p,t=1.,floor=0.):
    result=softmax(np.log(np.maximum(p,1e-12))/t,axis=1)
    return (1-floor)*result+floor/p.shape[1]

def main():
    configs=json.loads((OUT/'baseline_config.json').read_text());file=OUT/'probability_calibration_development.jsonl';rows=[]
    candidates=[{'t':t,'floor':floor} for t,floor in itertools.product([.5,.75,1.,1.25,1.5,2.],[0.,.01,.05,.1,.2])]
    for a,b in [('art','clipart'),('product','real_world')]:
      for seed in DEVELOPMENT_SEEDS:
        d=case('Office',a,b,seed,rank=128,image_mode='raw_center');views,_,_,_=load('Office')
        tr,val=train_test_split(np.arange(len(d['train_y'])),test_size=.25,stratify=d['train_y'],random_state=seed+33)
        tm=FrozenMap(views[b][d['train_indices'][tr]],128,mode='raw_center');tm.projection=d['source_map'].projection.copy()
        x=tm.transform(views[b][d['train_indices'][tr]]);v=tm.transform(views[b][d['train_indices'][val]])
        problem=TargetProblem.build(x,v,d['k'],np.array([],int),np.array([],int),seed,True,prepared=True)
        book=catalogue(d['source_map'].transform(views[a][d['public_source_indices']]),d['k']);base=release_joint(d['source'],d['source_y'],d['k'],2.,seed+110000)
        for method in METHODS:
            if method=='FedKAPT':p=revision_probability('Office',problem,CatalogueQuery(base,book,.5,.2),CONFIG,.5)['probability']
            else:
                cfg=configs[method];q=CatalogueQuery(base,book,cfg['mixing'],.2);c=cfg['config']
                if method=='DP-Refine':p=calibrate(refine(problem,q,c,**cfg['refinement'])[0],t=cfg['calibration'])
                else:p=run_control(problem,q,method,c)
                if method not in ['DP-NCM','DP-Graph','DP-Refine']:
                    message,_=q.decode();direct=softmax(-gpu_sqdist(problem.xtest,message.prototypes)/c['prototype_temperature'],axis=1);p=cfg['route']*p+(1-cfg['route'])*direct
            for candidate in candidates:
                prediction=calibrate(p,**candidate);assert np.array_equal(prediction.argmax(1),p.argmax(1))
                rows.append({'id':f'{a}_{b}_{seed}_{method}_{candidate}','pair':[a,b],'seed':seed,'method':method,'candidate':candidate,'results':metrics(d['train_y'][val],prediction,d['k'])})
        print('Probability calibration',a,b,seed,flush=True)
    file.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');selected={};summary={}
    for method in METHODS:
        ranking=[]
        for candidate in candidates:
            rr=[r for r in rows if r['method']==method and r['candidate']==candidate];assert len(rr)==6
            ranking.append({'candidate':candidate,'nll':float(np.mean([r['results']['nll'] for r in rr])),'brier':float(np.mean([r['results']['brier'] for r in rr])),'ece':float(np.mean([r['results']['ece'] for r in rr]))})
        ranking.sort(key=lambda r:(r['nll'],r['brier']));selected[method]=ranking[0]['candidate'];summary[method]=ranking
        print(method,'development probability choice',ranking[0],flush=True)
    (OUT/'probability_calibration_candidate.json').write_text(json.dumps(selected,indent=2));(OUT/'probability_calibration_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
