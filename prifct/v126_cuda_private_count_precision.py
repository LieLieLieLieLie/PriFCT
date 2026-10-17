"""Correct explicit float64 class counts, preserving failed v125 evidence.

PyTorch integer division otherwise creates float32 counts before assignment
to a float64 query. The mechanism must cast before division, including the
feature-independent count coordinate used in its sensitivity argument.
"""
import torch
from prifct.v124_cuda_private_baselines import *
from prifct import v124_cuda_private_baselines as original


def feature_query(source,y,k,feature,batch_size=128):
    query=original.feature_query(source,y,k,feature,batch_size)
    labels=torch.as_tensor(y,device='cuda',dtype=torch.long)
    query[:,-1]=torch.bincount(labels,minlength=k).to(torch.float64)/len(source)
    return query


def release(source,y,k,feature,epsilon,seed,delta=1e-5):
    clean=feature_query(source,y,k,feature);sigma=float(compute_sigma(2./len(source),epsilon,delta))
    rng=torch.Generator(device='cuda').manual_seed(seed)
    raw=clean+sigma*torch.randn(clean.shape,device='cuda',dtype=torch.float64,generator=rng)
    return JointMessage(gpu.host(raw),sigma,epsilon,len(source),delta)
