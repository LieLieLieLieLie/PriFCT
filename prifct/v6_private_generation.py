"""DP-MERF/DP-HP receiver adaptations for bounded public common features.

These retain the single noisy labelled feature mean / generator matching
objective. They are disclosed adaptations, not native benchmark reproduction.
An added noisy count coordinate uses the same full-record joint release.
Public maps are independently fixed; the receiver never accepts source rows.
No dataset utility is run by this module.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from revision.joint_release import release_joint, JointMessage


def bounded_rows(value):
    return value/torch.clamp(torch.linalg.vector_norm(value,dim=1,keepdim=True),min=1.)


@dataclass(frozen=True)
class PrivateFeatureMap:
    kind: str
    dimension: int
    bandwidth: float
    frequencies: object = None
    rho: float = 0.
    order: int = 0
    product_order: int = 0
    product_coordinates: tuple = (0,1)

    @classmethod
    def rff(cls, dimension, bandwidth, width=128, seed=75501):
        if dimension<1 or width<2 or width%2 or not math.isfinite(bandwidth) or bandwidth<=0:
            raise ValueError('Positive public bandwidth and even feature width required')
        frequencies=np.random.default_rng(seed).normal(size=(width//2,dimension))/math.sqrt(bandwidth)
        return cls('DP-MERF-RF',dimension,float(bandwidth),frequencies)

    @classmethod
    def hermite(cls,dimension,bandwidth,order=10,product_order=3,product_coordinates=(0,1)):
        if dimension<2 or not math.isfinite(bandwidth) or bandwidth<=0 or order<1 or product_order<1:
            raise ValueError('Positive fixed public Hermite configuration required')
        coordinates=tuple(product_coordinates)
        if len(coordinates)!=2 or len(set(coordinates))!=2 or min(coordinates)<0 or max(coordinates)>=dimension:
            raise ValueError('Two distinct public product coordinates required')
        # Stable positive root of rho^2+2*bandwidth*rho-1=0.
        rho=1./(math.sqrt(1.+bandwidth**2)+bandwidth)
        return cls('DP-HP',dimension,float(bandwidth),None,rho,int(order),int(product_order),coordinates)

    def _hermite(self,x,order):
        previous=(1-self.rho*self.rho)**.25*torch.exp(-self.rho/(1+self.rho)*x*x)
        values=[previous]
        current=math.sqrt(2*self.rho)*x*previous;values.append(current)
        for degree in range(2,order+1):
            k=degree-1
            following=math.sqrt(self.rho/(2*(k+1)))*2*x*current-self.rho*math.sqrt(k/(k+1))*previous
            values.append(following);previous,current=current,following
        return torch.stack(values,dim=-1)

    def transform_torch(self,x):
        if x.ndim!=2 or x.shape[1]!=self.dimension or not torch.isfinite(x).all():
            raise ValueError('Finite fitted common-feature rows required')
        if self.kind=='DP-MERF-RF':
            frequency=torch.as_tensor(self.frequencies,dtype=x.dtype,device=x.device)
            phase=x@frequency.T
            feature=torch.cat([torch.cos(phase),torch.sin(phase)],dim=1)/math.sqrt(len(frequency))
        elif self.kind=='DP-HP':
            summed=self._hermite(x,self.order).flatten(1)/math.sqrt(self.dimension)
            a,b=self.product_coordinates
            first=self._hermite(x[:,a],self.product_order)
            second=self._hermite(x[:,b],self.product_order)
            product=(first[:,:,None]*second[:,None,:]).flatten(1)
            feature=torch.cat([summed,product],dim=1)/math.sqrt(2.)
        else: raise ValueError('Unknown feature family')
        # Numerical safety is an explicit adaptation; in exact arithmetic both
        # constituent feature norms are <=1 by cosine or Mehler identity.
        return bounded_rows(feature)

    def transform(self,x,device='cpu'):
        with torch.no_grad():
            return self.transform_torch(torch.as_tensor(x,dtype=torch.float64,device=device)).cpu().numpy()


def source_release(source,y,k,feature_map,epsilon,seed=None,delta=1e-5,device='cpu'):
    """SOURCE ONLY. No target arrays or data-dependent kernel fitting accepted."""
    source=np.asarray(source,float)
    if source.ndim!=2 or not len(source) or np.max(np.linalg.norm(source,axis=1))>1+1e-9:
        raise ValueError('Frozen bounded common-source map required')
    feature=feature_map.transform(source,device)
    return release_joint(feature,y,k,epsilon,seed,delta)


class ConditionalGenerator(torch.nn.Module):
    def __init__(self,dimension,k,code=8,hidden=128):
        super().__init__();self.k=k;self.code=code
        self.layers=torch.nn.Sequential(torch.nn.Linear(code+k,hidden),torch.nn.ReLU(),
            torch.nn.Linear(hidden,hidden),torch.nn.ReLU(),torch.nn.Linear(hidden,dimension),torch.nn.Tanh())

    def forward(self,noise,labels):
        onehot=torch.nn.functional.one_hot(labels,self.k).to(noise.dtype)
        return bounded_rows(self.layers(torch.cat([noise,onehot],dim=1)))


@dataclass
class ReceiverGenerator:
    generator: ConditionalGenerator
    released_class_mass: np.ndarray
    feature_map: PrivateFeatureMap
    public_config: dict
    diagnostics: dict

    def sample(self,n_per_class=128,seed=75599,device=None):
        if n_per_class<1: raise ValueError('Positive public synthetic count required')
        device=next(self.generator.parameters()).device if device is None else device
        generator=torch.Generator(device=device);generator.manual_seed(seed)
        labels=torch.arange(self.generator.k,device=device).repeat_interleave(n_per_class)
        with torch.no_grad():
            noise=torch.randn(len(labels),self.generator.code,generator=generator,device=device)
            x=self.generator(noise,labels)
        return x.cpu().numpy(),labels.cpu().numpy()


def fit_receiver(message,feature_map,*,steps=500,learning_rate=.001,
                 samples_per_class=16,hidden=128,seed=75511,device='auto'):
    """RECEIVER ONLY. Matches the noisy unconditional labelled kernel mean.

    Uniform per-class Monte Carlo avoids rare minibatch omission; its generated
    means are weighted by noisy-count postprocessing, not true class counts.
    All model/loss outputs are postprocessing of the one private message.
    """
    if not isinstance(message,JointMessage): raise TypeError('Single JointMessage required')
    if steps<1 or samples_per_class<1 or learning_rate<=0 or not np.isfinite(learning_rate):
        raise ValueError('Fixed valid public optimization settings required')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    k=len(message.raw);probe=feature_map.transform(np.zeros((1,feature_map.dimension)))
    if message.raw.shape!=(k,probe.shape[1]+1) or not np.isfinite(message.raw).all():
        raise ValueError('Released mean must match the frozen feature map')
    counts=np.maximum(message.raw[:,-1],message.count_floor/message.n_public)
    mass=counts/counts.sum()
    devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model=ConditionalGenerator(feature_map.dimension,k,hidden=hidden).to(device)
    rng=torch.Generator(device=device);rng.manual_seed(seed+1)
    target=torch.as_tensor(message.raw[:,:-1],dtype=torch.float32,device=device)
    weights=torch.as_tensor(mass,dtype=torch.float32,device=device)[:,None]
    labels=torch.arange(k,device=device).repeat_interleave(samples_per_class)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_history=[]
    for step in range(steps):
        noise=torch.randn(len(labels),model.code,generator=rng,device=device)
        feature=feature_map.transform_torch(model(noise,labels)).reshape(k,samples_per_class,-1).mean(1)
        loss=torch.sum((weights*feature-target)**2)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite postprocessing objective')
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step==0 or step==steps-1:loss_history.append(float(loss.detach().cpu()))
    return ReceiverGenerator(model,mass,feature_map,
        {'steps':steps,'learning_rate':learning_rate,'samples_per_class':samples_per_class,'hidden':hidden,'seed':seed,'device':str(device)},
        {'postprocessing_loss_endpoints':loss_history,'query_bytes':int(message.raw.nbytes),
         'scope':'DP-summary author-objective adaptation; not native reproduction or utility certification'})
