"""DP-NTK author-feature/generator-objective adaptation to fixed sensor maps.

Exact author NTK class is extracted by AST from a pinned source. Vectorized
per-record gradients preserve the native normalized-gradient feature. Added
joint noisy counts and full labelled-record replace-one accounting are explicit
adaptations. No claim of reproducing the published image/tabular numbers.
"""
import ast
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

AUTHOR=Path(__file__).resolve().parent/'author_baselines_v7/DP-NTK/d3d93b9df28fa1f3129a499fd47e4a529613047c/IMAGE&TAB/models/ntk.py'


class NTKFeatureMap:
    kind='DP-NTK'

    def __init__(self,dimension,classes,hidden=32,seed=76201):
        self.dimension=dimension;self.classes=classes;self.hidden=hidden;self.seed=seed
        tree=ast.parse(AUTHOR.read_text(encoding='utf8'))
        selected=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='NTK']
        if len(selected)!=1:raise RuntimeError('Pinned author NTK class required')
        module=ast.Module(body=selected,type_ignores=[]);ast.fix_missing_locations(module)
        namespace={'torch':torch,'nn':nn}
        exec(compile(module,str(AUTHOR),'exec'),namespace)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.network=namespace['NTK'](dimension,hidden,classes).double()
        self.author_sha256=hashlib.sha256(AUTHOR.read_bytes()).hexdigest()

    def _network_parameters(self,x):
        # Fixed random public parameters; conversions do not train on x.
        return {name:value.detach().to(device=x.device,dtype=x.dtype) for name,value in self.network.named_parameters()}

    def transform_torch(self,x):
        if x.ndim!=2 or x.shape[1]!=self.dimension or not torch.isfinite(x).all():raise ValueError('Finite common records required')
        parameters=self._network_parameters(x)
        def scalar_output(params,row):return functional_call(self.network,params,(row[None,:],)).sum()
        gradients=vmap(grad(scalar_output),in_dims=(None,0))(parameters,x)
        feature=torch.cat([gradients[name].flatten(1) for name in parameters],dim=1)
        return feature/torch.clamp(torch.linalg.vector_norm(feature,dim=1,keepdim=True),min=1e-12)

    def transform(self,x,device='cpu',batch_size=256):
        x=np.asarray(x,float);chunks=[]
        # Parameter differentiation is required even for an inference feature;
        # generated input gradients remain available in transform_torch.
        with torch.enable_grad():
            for start in range(0,len(x),batch_size):
                batch=torch.as_tensor(x[start:start+batch_size],dtype=torch.float64,device=device)
                chunks.append(self.transform_torch(batch).detach().cpu().numpy())
        return np.concatenate(chunks)

    def native_loop_features(self,x):
        """Author formula used for equivalence QA, not the fast execution path."""
        x=torch.as_tensor(x,dtype=torch.float64)
        rows=[]
        for row in x:
            output=self.network(row[None,:])
            gradients=torch.autograd.grad(output,self.network.parameters(),grad_outputs=torch.ones_like(output))
            feature=torch.cat([part.flatten() for part in gradients])
            rows.append(feature/torch.linalg.vector_norm(feature))
        return torch.stack(rows).detach().numpy()
