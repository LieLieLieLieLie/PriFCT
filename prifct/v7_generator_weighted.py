"""Same summary matching generator with declared sum/product objective weight."""
import ast,inspect
from prifct import v6_private_generation as original

source=inspect.getsource(original.fit_receiver)
source=source.replace("seed=0,device='auto'):","seed=0,device='auto',loss_weights=None):")
# Signature spelling is checked rather than silently ignored.
if 'loss_weights=None' not in source:
    signature_end=source.index('):\n')
    source=source[:signature_end]+',loss_weights=None'+source[signature_end:]
source=source.replace('loss=torch.sum((weights*feature-target)**2)',
    "objective_weights=torch.ones(target.shape[1],device=device) if loss_weights is None else torch.as_tensor(loss_weights,dtype=target.dtype,device=device)\n        loss=torch.sum((weights*feature-target)**2*objective_weights)")
assert 'objective_weights' in source
namespace=dict(original.__dict__)
exec(compile(ast.parse(source),__file__,'exec'),namespace)
fit_receiver=namespace['fit_receiver']
