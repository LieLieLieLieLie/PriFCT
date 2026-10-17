"""Public spectral class-contrast denoising and convex distribution transport.

Positive-part James--Stein shrinkage is an established estimator, not a new
standalone contribution. For K>=4 its K-1 class-contrast risk improves Gaussian
moment estimation. Classification/semantic correctness do not follow from this
MSE result; this coupled receiver remains a development hypothesis.
"""
import numpy as np
from scipy.special import softmax
from prifct.v7_anchored_distribution import component_means
from prifct.v7_anchored_checked import transport


def reconstruct(first,covariance,sigma,ridge,*,no_denoising=False,no_preconditioner=False):
    first=np.asarray(first,float);covariance=np.asarray(covariance,float)
    k,d=first.shape
    if covariance.shape!=(d,d) or ridge<=0 or sigma<0:raise ValueError('Frozen public covariance and calibrated Gaussian scale required')
    values,basis=np.linalg.eigh((covariance+covariance.T)/2)
    if values.min() < -1e-10:raise ValueError('Public covariance must be positive semidefinite')
    values=np.maximum(values,0.)
    coordinates=first@basis
    common=coordinates.mean(0,keepdims=True)
    contrast=coordinates-common
    radius_squared=np.sum(contrast**2,axis=0)
    threshold=max(k-3,0)*sigma*sigma
    shrink=np.ones(d) if no_denoising or k<4 or sigma==0 else np.maximum(1-threshold/np.maximum(radius_squared,1e-300),0.)
    filtered=common+contrast*shrink
    weight=basis@filtered.T if no_preconditioner else basis@(filtered.T/(values[:,None]+ridge))
    return weight,dict(public_spectrum=values,shrink=shrink,contrast_radius_squared=radius_squared,
        gaussian_contrast_dimensions=k-1,positive_part_threshold=threshold,
        scope='Known spectral James-Stein contrast estimator; risk dominance for Gaussian moments, NOT classification accuracy')


def predict(problem,message,covariance,*,ridge=.01,temperature=.02,prior_strength=.5,transport_mix=.75,
            no_denoising=False,no_preconditioner=False,hard_condition=False):
    weight,diagnostic=reconstruct(message.raw[:,:-1],covariance,message.sigma_query,ridge,
        no_denoising=no_denoising,no_preconditioner=no_preconditioner)
    count=np.maximum(message.raw[:,-1]*message.n_public,1.);q=count/count.sum()
    components=component_means(problem.x,problem.labels,len(problem.centers))
    cost=-components@weight+temperature*np.log(q)[None,:]
    plan,solver=transport(cost,problem.target_mass,q,temperature,prior_strength)
    teacher=plan/problem.target_mass[:,None]
    if hard_condition:teacher=np.eye(problem.k)[teacher.argmax(1)]
    direct=softmax(problem.xtest@weight/temperature,axis=1)
    probability=(1-transport_mix)*direct+transport_mix*(problem.test_responsibilities@teacher)
    probability=np.maximum(probability,1e-12);probability/=probability.sum(1,keepdims=True)
    return probability,dict(weight=weight,plan=plan,teacher=teacher,class_prior=q,cost=cost,solver=solver,denoising=diagnostic,
        private_source_records_at_receiver=0,private_source_second_moment_at_receiver=False)
