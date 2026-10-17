"""Same native AGW objective with explicit complete LP solves.

The earlier full-size development emitted a maximum-network-simplex-iteration
warning. Its predictions are preserved but excluded from final comparisons.
Only the LP iteration budget is increased; nonoptimal native solves fail.
"""
import warnings
import numpy as np
import ot
from scipy.spatial.distance import cdist
from prifct.v7_native_structural import load_body,AGW


def strict_emd(a,b,cost,*args,**kwargs):
    kwargs.pop('log',None);kwargs['numItermax']=max(int(kwargs.get('numItermax',0)),2000000)
    with warnings.catch_warnings():
        warnings.filterwarnings('error',message='numItermax reached before optimality.*',category=UserWarning)
        plan,diagnostic=ot.emd(a,b,cost,*args,log=True,**kwargs)
    if diagnostic.get('warning') or diagnostic.get('result_code')!=1:
        raise RuntimeError('Native LP did not report optimality '+str(diagnostic))
    return plan


def agw(source,target,*,alpha=.5):
    utils=load_body(AGW/'utils.py')
    optim=load_body(AGW/'optim.py',skip=('utils',),injected=utils)
    optim['emd']=strict_emd
    namespace=load_body(AGW/'agw_scootr.py',skip=('utils','optim'),injected={**utils,**optim})
    d1=cdist(source,source,'sqeuclidean');d2=cdist(target,target,'sqeuclidean')
    d1/=max(float(d1.max()),1e-12);d2/=max(float(d2.max()),1e-12)
    with warnings.catch_warnings():
        warnings.filterwarnings('error',message='numItermax reached before optimality.*',category=UserWarning)
        plan,feature_plan,value,log=namespace['agw_scootr'](source,target,d1,d2,alpha=alpha,niter=10,
            algo='cg',algo2='emd',scaleCost=True,supervised=False,verbose=False,log=True)
    row_error=float(np.max(abs(plan.sum(1)-1/len(source))));column_error=float(np.max(abs(plan.sum(0)-1/len(target))))
    if max(row_error,column_error)>1e-6:raise RuntimeError('Native AGW marginal mismatch')
    return plan,dict(feature_plan=feature_plan,objective=float(value),objective_history=log['cost'],
        sample_row_error=row_error,sample_column_error=column_error,
        numerical_adapter='Same native objective, network simplex2million iteration ceiling; nonoptimal solves fail')
