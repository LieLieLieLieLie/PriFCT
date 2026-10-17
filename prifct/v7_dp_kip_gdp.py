"""Same native DP-KIP objective with exact full-cohort Gaussian composition.

Fixed sensitivity each step, fixed number of steps, independent noise and
adaptive gradients: Dong/Roth/Su2022 Thm4+Cor2 allow exact GDP composition.
This is NOT the asymptotic approximation for subsampled DP-SGD. No sampling
amplification is asserted. Old zCDP experiments remain immutable.
"""
import ast,inspect,math
import torch
from core.privacy import compute_sigma
from prifct import v7_dp_kip as original

SOURCE='https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12454'


def noise_scale(n,steps,clip,epsilon,delta):
    if n<1 or steps<1 or clip<=0:raise ValueError('Positive fixed public full-cohort parameters required')
    sensitivity=2*clip/n
    if epsilon==float('inf'):return 0.,float('inf')
    sigma=float(compute_sigma(sensitivity*math.sqrt(steps),epsilon,delta))
    mu_total=sensitivity*math.sqrt(steps)/sigma
    return sigma,.5*mu_total**2


tree=ast.parse(inspect.getsource(original))
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='fit')
function.args.kwonlyargs.append(ast.arg(arg='initialization_scale'))
function.args.kw_defaults.append(ast.Constant(value=1.))
class InitialScale(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch' and node.func.attr=='randn':
            # Modify only initial support draw, never the every-step noise draw.
            if isinstance(node.args[0],ast.Tuple):return ast.BinOp(left=node,op=ast.Mult(),right=ast.Name(id='initialization_scale',ctx=ast.Load()))
        return node
function=InitialScale().visit(function)
module=ast.Module(body=[function],type_ignores=[]);ast.fix_missing_locations(module)
namespace=dict(original.__dict__);namespace['noise_scale']=noise_scale
exec(compile(module,str(__file__),'exec'),namespace)
_fit=namespace['fit']


def fit(*args,initialization_scale=1.,**kwargs):
    if not math.isfinite(initialization_scale) or initialization_scale<=0:raise ValueError('Positive data-independent initialization scale required')
    model=_fit(*args,initialization_scale=initialization_scale,**kwargs)
    diagnostic=model['diagnostics']
    diagnostic.update(accountant='Exact adaptive full-cohort Gaussian GDP composition; converted analytically to total epsilon/delta; no CLT or amplification',
        gdp_mu_total=math.sqrt(2*diagnostic['rho_total']),initialization_scale=initialization_scale,
        exact_composition_source=SOURCE,initialization='Data-independent Gaussian; native scale or scale matching public unit-bound feature dimension')
    return model


predict=original.predict
