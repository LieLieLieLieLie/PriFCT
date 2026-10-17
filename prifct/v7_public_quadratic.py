"""Receiver-derived semantic scores from private first moments/public geometry.

The quadratic public-covariance preconditioner is known ridge regression. Its
coupling to paired function transport is a DEVELOPMENT hypothesis, not an
independently validated novelty/superiority claim. No privately fitted source
model or source second moment is transmitted.
"""
import numpy as np
from scipy.special import softmax
from prifct.v7_anchored_distribution import component_means
from prifct.v7_anchored_checked import transport


def public_covariance(functions):
    functions=np.asarray(functions,float)
    if functions.ndim!=2 or len(functions)<2 or not np.isfinite(functions).all():raise ValueError('Independent public source functions required')
    return functions.T@functions/len(functions)


def predict(problem,message,covariance,*,ridge=.01,temperature=.02,prior_strength=.1,transport_mix=.75,
            no_preconditioner=False,hard_condition=False):
    if ridge<=0 or not 0<=transport_mix<=1:raise ValueError('Positive public regularization and convex mixture required')
    # Raw query entries are private first moments (1/N)*sum phi(x) onehot(y),
    # not true counts, private covariance, or a source-trained classifier.
    first=message.raw[:,:-1]
    count=np.maximum(message.raw[:,-1]*message.n_public,1.);prior=count/count.sum()
    covariance=np.asarray(covariance,float)
    if covariance.shape!=(first.shape[1],first.shape[1]):raise ValueError('Same independent public function geometry required')
    # Unique minimizer of Tr(W.T Cpub W)/2 -Tr(W.T Q.T)+ridge||W||F^2/2.
    weight=first.T if no_preconditioner else np.linalg.solve(covariance+ridge*np.eye(len(covariance)),first.T)
    components=component_means(problem.x,problem.labels,len(problem.centers))
    semantic=components@weight
    # Cancel the source prior in the ROW reference measure. Only the explicit
    # column-KL penalty uses the released prior, avoiding a double prior.
    cost=-semantic+temperature*np.log(prior)[None,:]
    plan,diagnostic=transport(cost,problem.target_mass,prior,temperature,prior_strength)
    teacher=plan/problem.target_mass[:,None]
    if hard_condition:teacher=np.eye(problem.k)[teacher.argmax(1)]
    direct=softmax((problem.xtest@weight)/temperature,axis=1)
    probability=(1-transport_mix)*direct+transport_mix*(problem.test_responsibilities@teacher)
    probability=np.maximum(probability,1e-12);probability/=probability.sum(1,keepdims=True)
    return probability,dict(weight=weight,plan=plan,teacher=teacher,class_prior=prior,cost=cost,solver=diagnostic,
        private_source_records_at_receiver=0,private_source_second_moment_at_receiver=False,
        public_quadratic_objective='Independent public covariance, DP first moments, positive ridge; exact unique minimizer')
