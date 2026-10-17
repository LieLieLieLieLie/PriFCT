"""Small exact receiver checkpoints instead of repeated validation matrices.

All candidate predictions can be reconstructed from shared target functions,
responsibilities, exact weights and teachers. Float32 roundtrip is the same as
the scored studies. Saving this state is not quantization of research scores.
"""
import hashlib
import numpy as np
from scipy.special import softmax


def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def probability(state,target):
    direct=softmax(target['xtest']@state['weight']/float(state['temperature']),axis=1)
    mixture=float(state['transport_mix'])
    result=(1-mixture)*direct+mixture*(target['test_responsibilities']@state['teacher'])
    result=np.maximum(result,1e-12);result/=result.sum(1,keepdims=True)
    result=result.astype(np.float32).astype(np.float64);result/=result.sum(1,keepdims=True)
    return result


def save(path,model,options,target,scored):
    state={name:model[name] for name in ['weight','teacher','plan','class_prior','cost']}
    state.update(temperature=options['temperature'],transport_mix=options['transport_mix'])
    restored=probability(state,target)
    if digest(restored)!=digest(scored):raise RuntimeError('Small-checkpoint prediction is not bit identical')
    np.savez_compressed(path,**state,probability_sha256=np.array(digest(scored)))


def load(checkpoint,target):
    with np.load(checkpoint) as state:
        output=probability(state,target)
        assert digest(output)==str(state['probability_sha256'])
    return output
