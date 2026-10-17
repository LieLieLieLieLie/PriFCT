"""Kernel-distribution transport with anchored conditional target moments.

Empirical target kernel means are computed from records inside each target
component, not by applying a nonlinear map to a component's centroid. All
updates are postprocessing. This exploratory alternating procedure does not
claim a global optimum or semantic certificate.
"""
import numpy as np
from scipy.special import softmax
from prifct.method import fixed_transport


def component_means(features,labels,n_components):
    counts=np.bincount(labels,minlength=n_components)
    if np.any(counts==0):raise ValueError('Nonempty empirical target components required')
    result=np.zeros((n_components,features.shape[1]))
    np.add.at(result,labels,features)
    result/=counts[:,None]
    return result


def predict(problem,signature,message,*,temperature=.02,transport_mix=.75,
            source_pull=.5,steps=5,hard=False,point_component=False):
    if temperature<=0 or not 0<=transport_mix<=1 or not 0<=source_pull<=1 or steps<1:
        raise ValueError('Finite declared refinement settings required')
    train=signature.transform(problem.x);test=signature.transform(problem.xtest)
    anchor=message.decode()[0].prototypes
    means=anchor.copy()
    target=signature.transform(problem.centers) if point_component else component_means(train,problem.labels,len(problem.centers))
    changes=[];teacher=None
    for _ in range(steps):
        local=softmax(train@means.T/temperature,axis=1)
        if transport_mix:
            cost=-np.log(np.maximum(softmax(target@means.T/temperature,axis=1),1e-12))
            plan,conditional=fixed_transport(cost,mass=1.,tau=1.,target_mass=problem.target_mass)
            teacher=conditional[:,:problem.k]
            if hard:teacher=np.eye(problem.k)[teacher.argmax(1)]
            local=(1-transport_mix)*local+transport_mix*(problem.responsibilities@teacher)
        # Confidence-weighted moment estimates, identical in every ablation.
        weights=local**2;mass=weights.sum(axis=0)
        update=weights.T@train/np.maximum(mass[:,None],1e-12)
        update[mass<1e-8]=anchor[mass<1e-8]
        update=source_pull*anchor+(1-source_pull)*update
        changes.append(float(np.max(np.linalg.norm(update-means,axis=1))))
        means=update
    probability=softmax(test@means.T/temperature,axis=1)
    # Recompute final transport consistently with the FINAL moment state.
    if transport_mix:
        cost=-np.log(np.maximum(softmax(target@means.T/temperature,axis=1),1e-12))
        plan,conditional=fixed_transport(cost,mass=1.,tau=1.,target_mass=problem.target_mass)
        teacher=conditional[:,:problem.k]
        if hard:teacher=np.eye(problem.k)[teacher.argmax(1)]
        probability=(1-transport_mix)*probability+transport_mix*(problem.test_responsibilities@teacher)
    probability=np.maximum(probability,1e-12);probability/=probability.sum(1,keepdims=True)
    return probability,dict(moment_change_history=changes,means=means,teacher=teacher,
        point_component=point_component,target_training_labels=0,
        caveat='Iterative postprocessing heuristic, no monotonic objective or semantic certificate claimed')
