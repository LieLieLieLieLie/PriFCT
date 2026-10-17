"""CUDA arithmetic for the unchanged private-feature generator protocols.

No CPU optimization or data-dependent feature arithmetic is permitted here.
The synthetic classifier solves sklearn's multinomial C-regularized objective;
stored sklearn coefficients are numerical references, never fitted here.
"""
import math
import numpy as np
import torch
from core.privacy import compute_sigma
from revision.joint_release import JointMessage
from prifct import v111_cuda_semantic_core as gpu
from prifct.v6_private_generation import ConditionalGenerator,ReceiverGenerator
from prifct.v7_dp_kip import fc_ntk,coefficients


def feature_query(source,y,k,feature,batch_size=128):
    x=gpu.tensor(source);labels=torch.as_tensor(y,device='cuda',dtype=torch.long)
    assert len(x)>0 and labels.shape==(len(x),) and ((labels>=0)&(labels<k)).all()
    assert torch.linalg.vector_norm(x,dim=1).max()<=1+1e-9
    dimension=feature.transform_torch(torch.zeros((1,x.shape[1]),device='cuda',dtype=torch.float64)).shape[1]
    q=torch.zeros((k,dimension+1),device='cuda',dtype=torch.float64)
    for start in range(0,len(x),batch_size):
        with torch.enable_grad():value=gpu.clip(feature.transform_torch(x[start:start+batch_size])).detach()
        q[:,:-1].index_add_(0,labels[start:start+batch_size],value/len(x))
    q[:,-1]=torch.bincount(labels,minlength=k)/len(x)
    return q


def release(source,y,k,feature,epsilon,seed,delta=1e-5):
    clean=feature_query(source,y,k,feature);sigma=float(compute_sigma(2./len(source),epsilon,delta))
    rng=torch.Generator(device='cuda').manual_seed(seed)
    raw=clean+sigma*torch.randn(clean.shape,device='cuda',dtype=torch.float64,generator=rng)
    return JointMessage(gpu.host(raw),sigma,epsilon,len(source),delta)


def fit_receiver(message,feature,*,steps=500,learning_rate=.001,samples_per_class=16,
                 hidden=128,seed=77281,loss_weights=None):
    gpu.require_cuda();k=len(message.raw)
    raw=gpu.tensor(message.raw);count=raw[:,-1].clamp_min(message.count_floor/message.n_public)
    mass=count/count.sum()
    probe=feature.transform_torch(torch.zeros((1,feature.dimension),device='cuda',dtype=torch.float64))
    assert raw.shape==(k,probe.shape[1]+1)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed);model=ConditionalGenerator(feature.dimension,k,hidden=hidden).cuda()
    rng=torch.Generator(device='cuda').manual_seed(seed+1)
    target=raw[:,:-1].float();weights=mass.float()[:,None]
    objective=torch.ones(target.shape[1],device='cuda') if loss_weights is None else torch.as_tensor(loss_weights,dtype=torch.float32,device='cuda')
    labels=torch.arange(k,device='cuda').repeat_interleave(samples_per_class)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate);history=[]
    for step in range(steps):
        noise=torch.randn(len(labels),model.code,device='cuda',generator=rng)
        value=feature.transform_torch(model(noise,labels)).reshape(k,samples_per_class,-1).mean(1)
        loss=(((weights*value-target)**2)*objective).sum()
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step in [0,steps-1]:history.append(loss.item())
    return ReceiverGenerator(model,gpu.host(mass),feature,
        dict(steps=steps,learning_rate=learning_rate,samples_per_class=samples_per_class,hidden=hidden,seed=seed,device='cuda:0'),
        dict(postprocessing_loss_endpoints=history,query_bytes=int(message.raw.nbytes),cpu_fallback=False,
             objective='Original noisy unconditional labelled feature mean; public sum/product weights'))


def lr_objective_gradient(x,y,w,C):
    z=torch.cat([x,torch.ones((len(x),1),device=x.device,dtype=x.dtype)],1)
    logits=z@w;logp=torch.log_softmax(logits,1);p=logp.exp()
    penalty=torch.ones_like(w)/(C*len(x));penalty[-1]=0
    objective=-logp[torch.arange(len(x),device=x.device),y].mean()+.5*(penalty*w*w).sum()
    gradient=z.T@(p-torch.nn.functional.one_hot(y,w.shape[1]))/len(x)+penalty*w
    return objective,gradient,z,p,penalty


def fit_lr(x,y,k,C=1.):
    x=gpu.tensor(x);y=torch.as_tensor(y,device='cuda',dtype=torch.long)
    assert set(gpu.host(torch.unique(y)).tolist())==set(range(k)) and C>0
    w=torch.nn.Parameter(torch.zeros((x.shape[1]+1,k),device='cuda',dtype=torch.float64))
    optimizer=torch.optim.LBFGS([w],lr=1.,max_iter=1000,tolerance_grad=1e-11,tolerance_change=1e-15,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();value=lr_objective_gradient(x,y,w,C)[0];value.backward();return value
    optimizer.step(closure)
    with torch.no_grad():
        w=w.detach();w[-1]-=w[-1].mean()
        iterations=0
        for iterations in range(15):
            value,gradient,z,p,penalty=lr_objective_gradient(x,y,w,C)
            if gradient.abs().max()<2e-10:break
            curvature=torch.diag_embed(p)-p[:,:,None]*p[:,None,:]
            hessian=torch.einsum('ni,nj,nab->iajb',z,z,curvature).reshape(w.numel(),w.numel())/len(x)
            hessian+=torch.diag(penalty.flatten())
            # Fix the unidentifiable common intercept without penalising it.
            hessian[-k:,-k:]+=torch.ones((k,k),device='cuda',dtype=x.dtype)/k
            step=torch.linalg.solve(hessian,gradient.flatten()).reshape_as(w)
            rate=1.
            while rate>2**-20:
                candidate=w-rate*step;candidate[-1]-=candidate[-1].mean()
                if lr_objective_gradient(x,y,candidate,C)[0]<=value+1e-15:break
                rate*=.5
            w=candidate
        value,gradient,*_=lr_objective_gradient(x,y,w,C)
        assert gradient.abs().max()<2e-9,gradient.abs().max().item()
    return dict(weight=gpu.host(w),C=C,classes=k,diagnostics=dict(objective=value.item(),
        gradient_infinity_norm=gradient.abs().max().item(),newton_steps=iterations,
        objective_definition='Mean multinomial cross entropy + ||W||_F^2/(2*C*N); unpenalised intercept',
        compute_device='cuda:0',cpu_fallback=False))


def predict_lr(model,x):
    x=gpu.tensor(x);z=torch.cat([x,torch.ones((len(x),1),device='cuda',dtype=x.dtype)],1)
    return torch.softmax(z@gpu.tensor(model['weight']),1)


def predict_kip(model,x,temperature):
    assert temperature>0
    with torch.no_grad():
        support=gpu.tensor(model['support']);labels=gpu.tensor(model['support_labels'])
        alpha=coefficients(support,labels,model['reg'])
        return torch.softmax((fc_ntk(gpu.tensor(x),support)@alpha)/temperature,1)


def export(p):
    p=p.float().double();return p/p.sum(1,keepdim=True)
