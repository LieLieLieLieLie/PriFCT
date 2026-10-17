"""Target-only conservative prototype refinement of the original soft teacher.

No target labels, validation observations or additional private queries are
accepted by this function. Confidence filtering is heuristic, not a semantic
correctness certificate. The source pull prevents unconstrained drift.
"""
import numpy as np
from scipy.special import softmax
from prifct.method import gpu_sqdist
from revision.protocol import clip

def refine(problem,query,config,strength=.5,steps=5,temperature=.05,threshold=.0,clusters=None,transport_mix=.25,graph=True,hard=False):
    message,_=query.decode();source=message.prototypes;centers=source.copy()
    out=problem.predict(query,variant='hard' if hard else 'full',**config) if transport_mix else {'teacher':np.zeros((len(problem.centers),problem.k)),'plan':None}
    transported=problem.responsibilities@out['teacher']
    histories=[]
    for _ in range(steps):
        direct=softmax(-gpu_sqdist(problem.x,centers)/temperature,axis=1)
        initial=transport_mix*transported+(1-transport_mix)*direct
        smooth=problem.propagate(initial,steps=30) if graph else initial
        p=.5*smooth+.5*direct
        confidence=p.max(1)
        weights=p**2*(confidence>=threshold)[:,None]
        denominator=weights.sum(0)
        target=weights.T@problem.x/np.maximum(denominator[:,None],1e-12)
        target[denominator<1.]=source[denominator<1.]
        centers=clip((1-strength)*source+strength*target)
        histories.append({'max_drift':float(np.max(np.linalg.norm(centers-source,axis=1))),'active_classes':int((denominator>=1).sum())})
    direct=softmax(-gpu_sqdist(problem.x,centers)/temperature,axis=1)
    initial=transport_mix*transported+(1-transport_mix)*direct
    direct_test=softmax(-gpu_sqdist(problem.xtest,centers)/temperature,axis=1)
    propagated=problem.test_transition@problem.propagate(initial,steps=30) if graph else transport_mix*(problem.test_responsibilities@out['teacher'])+(1-transport_mix)*direct_test
    result=.5*propagated+.5*direct_test
    return result,{'centers':centers,'history':histories,'plan':out['plan']}
