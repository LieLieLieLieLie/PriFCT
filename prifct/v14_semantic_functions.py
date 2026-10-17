"""DP-packet-conditioned public posterior functions and samplewise transport.

Every input is a released DP packet, frozen public maps, independent unlabelled
public pairs, or unlabelled adaptation covariates. No fitting API accepts labels.
Semantic sufficiency across views is an assumption, never implied by pairing.
"""
import numpy as np
from scipy.special import softmax
from scipy.linalg import solve
from prifct.v6_public_relation import squared_distance
from prifct.v12_cell_functions import simplex
from prifct.v7_anchored_checked import transport
from prifct.v9_density_receiver import probability

OPTIONS=[dict(bandwidth_factor=b,ridge=r) for b in [.25,1.] for r in [.01,.1,1.]]

def public_operators(point, target_fit, target_held, adaptation, validation):
    """Prepare six label-blind public kernel operators before packet access."""
    inputs=[target_fit,target_held,adaptation,validation]
    distance=[squared_distance(point.target_scaler.transform(x),point.target_landmarks) for x in inputs]
    result=[]
    for option in OPTIONS:
        kernels=[np.exp(-v/(point.target_bandwidth*option['bandwidth_factor'])) for v in distance]
        a=kernels[0]
        regression=solve(a.T@a+option['ridge']*np.eye(a.shape[1]),a.T,assume_a='pos')
        result.append(dict(option=option,regression=regression,held=kernels[1],adaptation=kernels[2],validation=kernels[3]))
    return result

def calibrated(operators,source_fit,source_held,weight,teacher_temperature):
    fit=softmax(source_fit@weight/teacher_temperature,axis=1)
    held=softmax(source_held@weight/teacher_temperature,axis=1)
    choices=[]
    for operator in operators:
        coefficients=operator['regression']@fit
        prediction=simplex(operator['held']@coefficients)
        error=float(np.mean(np.sum((prediction-held)**2,axis=1)))
        choices.append((error,coefficients))
    selected=min(range(len(choices)),key=lambda j:choices[j][0])
    coefficients=choices[selected][1];operator=operators[selected]
    return dict(adaptation=simplex(operator['adaptation']@coefficients),
                validation=simplex(operator['validation']@coefficients),coefficients=coefficients,
                selected=selected,public_errors=[v[0] for v in choices])

def log_probability(p):
    p=np.maximum(np.asarray(p,float),1e-12);p=p/p.sum(1,keepdims=True)
    return np.log(p)

def offset(logscores,temperature,balance_strength,q,mass=None):
    n,k=logscores.shape
    mass=np.ones(n)/n if mass is None else np.asarray(mass,float)
    q=np.maximum(np.asarray(q,float),1e-12);q/=q.sum()
    if balance_strength==0:return np.zeros(k),dict(dual_gradient_infinity=0.,duality_gap=0.,iterations=0)
    cost=-logscores+temperature*np.log(q)[None,:]
    plan,solver=transport(cost,mass,q,temperature,balance_strength*temperature)
    conditional=plan/(mass/mass.sum())[:,None]
    correction=temperature*np.log(conditional[0])+cost[0]-temperature*np.log(q)
    correction-=correction.mean()
    reconstruction=softmax((logscores+correction)/temperature,axis=1)
    if not np.allclose(reconstruction,conditional,atol=2e-11,rtol=2e-9):
        raise RuntimeError('Transport dual reconstruction mismatch')
    return correction,solver

def predict(logscores,correction,temperature):
    return probability(logscores,correction,temperature)
