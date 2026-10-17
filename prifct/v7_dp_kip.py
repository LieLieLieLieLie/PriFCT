"""DP-KIP tabular FC-NTK objective, GPU port with corrected record privacy.

Pinned author architecture is Dense(sqrt2,.1)->ReLU->Dense(sqrt2,.1).
Exact infinite-width kernel and trace-regularized KRR MSE are implemented.
Adaptations: fixed public support labels, independent Gaussian draws at every
step, frozen public preprocessing, full-cohort gradients and explicit zCDP
composition for full labelled-record replacement. The native source's private
class-balanced label initialization and repeated epoch noise are not reused.
"""
import hashlib,math
from pathlib import Path
import numpy as np
import torch
from torch.func import grad,vmap
from scipy.special import softmax

AUTHOR=Path(__file__).resolve().parent/'author_baselines_v7/DP-KIP/da198c8b4bfb145b1d19314a91030c476547a238/dpkip_tab_data.py'


def fc_ntk(x,y):
    covariance=2*(x@y.T)/x.shape[1]+.01
    dx=2*torch.sum(x*x,dim=1)/x.shape[1]+.01
    dy=2*torch.sum(y*y,dim=1)/y.shape[1]+.01
    scale=torch.sqrt(dx[:,None]*dy[None,:])
    cosine=torch.clamp(covariance/scale,-1+1e-12,1-1e-12)
    angle=torch.acos(cosine)
    relu=scale*(torch.sin(angle)+(math.pi-angle)*cosine)/(2*math.pi)
    derivative=(math.pi-angle)/(2*math.pi)
    kernel=2*relu+.01+2*covariance*derivative
    if x is y:
        # Exact diagonal and its finite derivative; acos at one is singular.
        mask=torch.eye(len(x),device=x.device,dtype=torch.bool)
        kernel=torch.where(mask,2*dx[:,None]+.01,kernel)
    return kernel


def coefficients(support,labels,reg):
    gram=fc_ntk(support,support)
    regularized=gram+abs(reg)*torch.trace(gram)/len(gram)*torch.eye(len(gram),device=support.device,dtype=support.dtype)
    return torch.linalg.solve(regularized,labels)


def record_loss(support,row,label,support_labels,reg):
    prediction=fc_ntk(row[None,:],support)@coefficients(support,support_labels,reg)
    return .5*torch.mean((prediction[0]-label)**2)


def noise_scale(n,steps,clip,epsilon,delta):
    if n<1 or steps<1 or clip<=0 or epsilon<=0 or not 0<delta<1:raise ValueError('Public positive accountant parameters required')
    log_delta=math.log(1/delta)
    rho=(epsilon/(math.sqrt(log_delta+epsilon)+math.sqrt(log_delta)))**2
    sigma=(2*clip/n)*math.sqrt(steps/(2*rho))
    return sigma,rho


def fit(source,source_y,classes,*,epsilon,steps=100,clip=.1,lr=.1,reg=1e-6,per_class=2,seed=76701,delta=1e-5):
    if not torch.cuda.is_available():raise RuntimeError('DP-KIP CUDA port requires GPU')
    x=torch.as_tensor(source,device='cuda',dtype=torch.float64)
    y=torch.nn.functional.one_hot(torch.as_tensor(source_y,device='cuda',dtype=torch.long),classes).to(x.dtype)
    generator=torch.Generator(device='cuda').manual_seed(seed)
    # Fixed catalog, not source class counts/presence or sampled private labels.
    support_y=torch.nn.functional.one_hot(torch.arange(classes,device='cuda').repeat_interleave(per_class),classes).to(x.dtype)
    support=torch.nn.Parameter(torch.randn((len(support_y),x.shape[1]),device='cuda',dtype=x.dtype,generator=generator))
    optimizer=torch.optim.Adam([support],lr=lr)
    sigma,rho=noise_scale(len(x),steps,clip,epsilon,delta)
    gradient=vmap(grad(record_loss),in_dims=(None,0,0,None,None))
    for _ in range(steps):
        total=torch.zeros_like(support)
        # Microbatching computes the SAME complete cohort query each step.
        for start in range(0,len(x),128):
            individual=gradient(support.detach(),x[start:start+128],y[start:start+128],support_y,reg)
            norm=torch.linalg.vector_norm(individual.flatten(1),dim=1)
            individual=individual/torch.clamp(norm/clip,min=1.)[:,None,None]
            total+=individual.sum(0)
        noisy=total/len(x)+sigma*torch.randn(support.shape,device='cuda',dtype=x.dtype,generator=generator)
        if not torch.isfinite(noisy).all():raise RuntimeError('Nonfinite clipped DP-KIP gradient')
        optimizer.zero_grad();support.grad=noisy;optimizer.step()
    return dict(support=support.detach().cpu().numpy(),support_labels=support_y.cpu().numpy(),reg=reg,
        diagnostics=dict(sigma_averaged_gradient=sigma,rho_total=rho,steps=steps,epsilon=epsilon,delta=delta,
            full_record_sensitivity_each_step=2*clip/len(x),private_initialization=False,
            noise='Fresh independent benchmark-simulation Gaussian draw each step; deployment seed not public',
            accountant='Full cohort composed Gaussian zCDP; no subsampling amplification claimed',
            author_sha256=hashlib.sha256(AUTHOR.read_bytes()).hexdigest(),native_objective='FC infinite-NTK trace-ridge KRR per-record MSE'))


def predict(model,target,temperature=1.):
    with torch.no_grad():
        support=torch.as_tensor(model['support'],device='cuda',dtype=torch.float64)
        labels=torch.as_tensor(model['support_labels'],device='cuda',dtype=torch.float64)
        alpha=coefficients(support,labels,model['reg'])
        target=torch.as_tensor(target,device='cuda',dtype=torch.float64)
        scores=(fc_ntk(target,support)@alpha).cpu().numpy()
    return softmax(scores/temperature,axis=1)
