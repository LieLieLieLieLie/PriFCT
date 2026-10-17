"""Serialize native UCOOT numerical diagnostics without backend RNG objects.

Couplings, features and classifier inputs remain numerically unchanged. POT's
log includes a backend object holding a torch Generator, which is not a model
parameter and cannot be pickled. Preserve numeric entries and its type name.
"""
import numpy as np
import torch
from prifct.v7_native_structural import ucoot as native_ucoot


def numerical(value):
    if value is None or isinstance(value,(str,int,float,bool,np.ndarray,np.generic)):return value
    if torch.is_tensor(value):return value.detach().cpu().numpy()
    if isinstance(value,dict):return {key:numerical(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [numerical(item) for item in value]
    return dict(non_numeric_diagnostic_type=type(value).__module__+'.'+type(value).__name__)


def ucoot(*args,**kwargs):
    plan,diagnostic=native_ucoot(*args,**kwargs)
    return plan,numerical(diagnostic)
