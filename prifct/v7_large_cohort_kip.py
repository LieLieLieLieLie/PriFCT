"""Same native full-cohort clipped gradient, larger fixed CUDA microbatches.

No sampling, objective, step count, noise or privacy accounting change. Frozen
MF training code is untouched. This execution adapter is fixed before any
external WISDM utility and tested against the original128-record aggregation.
"""
import ast,inspect,math
import numpy as np
import torch
from prifct import v7_dp_kip as original
from prifct import v7_dp_kip_gdp as gdp

MICROBATCH=1024


def compiled(accountant):
    node=next(n for n in ast.parse(inspect.getsource(original)).body if isinstance(n,ast.FunctionDef) and n.name=='fit')
    node.args.kwonlyargs.append(ast.arg(arg='initialization_scale'))
    node.args.kw_defaults.append(ast.Constant(value=1.))
    class Execution(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==128:return ast.Constant(value=MICROBATCH)
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch' and node.func.attr=='randn' and isinstance(node.args[0],ast.Tuple):
                return ast.BinOp(left=node,op=ast.Mult(),right=ast.Name(id='initialization_scale',ctx=ast.Load()))
            return node
    node=Execution().visit(node);module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module)
    namespace=dict(original.__dict__);namespace['noise_scale']=gdp.noise_scale if accountant=='gdp' else original.noise_scale
    exec(compile(module,str(__file__),'exec'),namespace)
    return namespace['fit']


_gdp=compiled('gdp');_zcdp=compiled('zcdp')


def fit(*args,accountant,initialization_scale=1.,**kwargs):
    if accountant not in ['gdp','zcdp'] or not math.isfinite(initialization_scale) or initialization_scale<=0:raise ValueError('Fixed valid public execution/accountant settings required')
    model=(_gdp if accountant=='gdp' else _zcdp)(*args,initialization_scale=initialization_scale,**kwargs)
    d=model['diagnostics'];d['microbatch']=MICROBATCH;d['query']='Every source record at every step; exact per-record norm clipping, no sampling'
    if accountant=='gdp':
        d.update(accountant='Exact adaptive full-cohort Gaussian GDP composition; no amplification or CLT',
            gdp_mu_total=math.sqrt(2*d['rho_total']),initialization_scale=initialization_scale,
            exact_composition_source=gdp.SOURCE,initialization='Data-independent public-scale Gaussian')
    return model


def gradient_equivalence():
    from torch.func import grad,vmap
    rng=np.random.default_rng(77601)
    x=torch.as_tensor(rng.normal(size=(1157,8)),dtype=torch.float64,device='cuda')
    y=torch.nn.functional.one_hot(torch.as_tensor(rng.integers(0,4,len(x)),device='cuda'),4).double()
    s=torch.as_tensor(rng.normal(size=(8,8)),dtype=torch.float64,device='cuda')
    sy=torch.nn.functional.one_hot(torch.arange(4,device='cuda').repeat_interleave(2),4).double()
    gradient=vmap(grad(original.record_loss),in_dims=(None,0,0,None,None))
    values=[]
    for batch in [128,MICROBATCH]:
        total=torch.zeros_like(s)
        for start in range(0,len(x),batch):
            individual=gradient(s,x[start:start+batch],y[start:start+batch],sy,1e-6)
            individual/=torch.clamp(torch.linalg.vector_norm(individual.flatten(1),dim=1)/.01,min=1.)[:,None,None]
            total+=individual.sum(0)
        values.append((total/len(x)).cpu().numpy())
    error=float(np.max(abs(values[0]-values[1])))
    assert error<2e-14,error
    return dict(records=len(x),old_microbatch=128,new_microbatch=MICROBATCH,max_absolute_clipped_mean_difference=error,
        per_record_objective='Unchanged original FC infinite NTK trace-ridge MSE',sampling=False)


if __name__=='__main__':
    import json,hashlib
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):result=gradient_equivalence()
    result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=Path(__file__).resolve().parents[1]/'results/models/eswa_fedkapt/original_v7_research/manuscript_work'
    (out/'large_cohort_kip_execution_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
