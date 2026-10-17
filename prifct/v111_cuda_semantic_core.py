"""CUDA-only double-precision implementation of the frozen positive interface.

Existing CPU implementations/checkpoints remain immutable references. This
module does not accept labels, silently fall back to CPU, or alter the query.
Host work is confined to metadata, I/O and diagnostic scalar control flow.
"""
import torch

DTYPE = torch.float64
TEMPERATURES = [.01, .03, .1, .3]
OPTIONS = [dict(bandwidth_factor=b, ridge=r) for b in [.25, 1.] for r in [.01, .1, 1.]]


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory; CPU fallback is disabled')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda:0')


def tensor(value):
    device = require_cuda()
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=DTYPE)
    return torch.as_tensor(value, device=device, dtype=DTYPE)


def host(value):
    return value.detach().cpu().numpy()


def clip(x):
    return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1.)


def standardize(scaler, raw):
    x = tensor(raw)
    name = type(scaler).__name__
    if name == 'PublicLogScale':
        return (x.sign() * x.abs().log1p() - tensor(scaler.mean)) / tensor(scaler.std)
    if name == 'PublicStandardizer':
        z = x.sign() * x.abs().log1p() if scaler.logarithmic else x
        return clip((z - tensor(scaler.mean)) / tensor(scaler.scale))
    raise TypeError('Unsupported frozen standardizer: ' + name)


def squared_distance(a, b):
    a, b = tensor(a), tensor(b)
    return ((a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T).clamp_min(0.)


def point_transform(point, raw, source):
    name = type(point).__name__
    if name == 'PublicRegressionBridge':
        if source:
            return clip(standardize(point.source_scaler, raw) @ tensor(point.source_basis) / point.source_scale)
        z = standardize(point.target_scaler, raw)
        kernel = torch.exp(-squared_distance(z, point.target_landmarks) / point.target_bandwidth)
        return clip(kernel @ tensor(point.target_coefficients))
    if name == 'PairedRelationMap':
        scale = point.source_standardizer if source else point.target_standardizer
        landmarks = point.source_landmarks if source else point.target_landmarks
        bandwidth = point.source_bandwidth if source else point.target_bandwidth
        center = point.source_center if source else point.target_center
        basis = point.source_basis if source else point.target_basis
        kernel = torch.exp(-squared_distance(standardize(scale, raw), landmarks) / bandwidth) / len(landmarks)**.5
        return clip((kernel-tensor(center)) @ tensor(basis))
    raise TypeError('Unsupported frozen point map: ' + name)


def signature_transform(signature, x):
    name = type(signature).__name__
    if name == 'ProjectedSignature':
        return signature_transform(signature.original, x) @ tensor(signature.basis)
    if name == 'IdentitySignature':
        return x
    if name == 'MultiScaleSignature':
        pieces = [x]
        for frequency in signature.frequencies:
            phase = x @ tensor(frequency)
            pieces.append(torch.cat([torch.cos(phase), torch.sin(phase)], dim=1) / frequency.shape[1]**.5)
        return torch.cat(pieces, dim=1) / len(pieces)**.5
    raise TypeError('Unsupported frozen signature: ' + name)


def transform(mapping, raw, source=True):
    if type(mapping).__name__ != 'FunctionalBridge':
        return point_transform(mapping, raw, source)
    if source:
        return signature_transform(mapping.signature, point_transform(mapping.point_bridge, raw, True))
    point = mapping.point_bridge
    kernel = torch.exp(-squared_distance(standardize(point.target_scaler, raw), point.target_landmarks) / mapping.bandwidth)
    return clip(kernel @ tensor(mapping.coefficients))


def features(x, mean):
    x = tensor(x)
    return torch.cat([x-tensor(mean), torch.ones((len(x), 1), device=x.device, dtype=x.dtype)], dim=1)


def sufficient_statistic(x, raw, n_public):
    x, raw = tensor(x), tensor(raw)
    mean = x.mean(0)
    count = (raw[:, -1]*n_public).clamp_min(1.)
    q = count/count.sum()
    moment = raw[:, :-1].T - mean[:, None]*raw[:, -1][None, :]
    sufficient = torch.cat([moment, q[None, :]], dim=0)
    return mean, sufficient-sufficient.mean(1, keepdim=True), q


def teacher_objective(z, sufficient, weight, penalty):
    centered = weight-weight.mean(1, keepdim=True)
    logits = z @ centered
    p = torch.softmax(logits, dim=1)
    value = logits.logsumexp(1).mean() - (sufficient*centered).sum() + .5*penalty*(weight*weight).sum()
    gradient = z.T @ p/len(z)-sufficient
    gradient = gradient-gradient.mean(1, keepdim=True)+penalty*weight
    return value, gradient, p


def teacher_hessian(z, p, penalty):
    covariance = torch.diag_embed(p)-p[:, :, None]*p[:, None, :]
    d, k = z.shape[1], p.shape[1]
    result = torch.einsum('ni,nj,nkl->ikjl', z, z, covariance).reshape(d*k, d*k)/len(z)
    return result+penalty*torch.eye(d*k, device=z.device, dtype=z.dtype)


def fit(source_public, raw, n_public, ridge):
    x, raw = tensor(source_public), tensor(raw)
    if ridge <= 0 or raw.ndim != 2 or raw.shape[1] != x.shape[1]+1:
        raise ValueError('Positive ridge and matching released packet required')
    mean, sufficient, _ = sufficient_statistic(x, raw, n_public)
    z = features(x, mean)
    penalty = ridge/len(raw)
    weight = torch.zeros_like(sufficient, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight], lr=1., max_iter=2000, max_eval=3000,
                                 tolerance_grad=1e-10, tolerance_change=1e-15,
                                 history_size=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad(set_to_none=True)
        value, gradient, _ = teacher_objective(z, sufficient, weight, penalty)
        weight.grad = gradient.detach()
        return value
    optimizer.step(closure)
    refinements = 0
    # The objective is strongly convex. Refine stationarity on GPU; different
    # quasi-Newton histories must not be confused with an algorithm change.
    with torch.no_grad():
        for _ in range(8):
            value, gradient, p = teacher_objective(z, sufficient, weight, penalty)
            if gradient.abs().max().item() <= 2e-11:
                break
            hessian = teacher_hessian(z, p, penalty)
            direction = torch.linalg.solve(hessian, -gradient.flatten()).reshape_as(weight)
            scale = 1.
            for _ in range(30):
                candidate = weight+scale*direction
                new_value, new_gradient, _ = teacher_objective(z, sufficient, candidate, penalty)
                if new_value <= value+1e-13 and new_gradient.abs().max() < gradient.abs().max():
                    weight.copy_(candidate); refinements += 1; break
                scale *= .5
            else:
                break
        value, gradient, _ = teacher_objective(z, sufficient, weight, penalty)
        error = gradient.abs().max().item()
        if not torch.isfinite(weight).all() or error > 2e-9:
            raise RuntimeError('CUDA teacher stationarity failed: '+str(error))
        result = weight.detach()-weight.detach().mean(1, keepdim=True)
    iterations = optimizer.state[weight].get('n_iter', 0)
    return result, mean, dict(gradient_infinity=error, objective=value.item(), regularization=penalty,
        iterations=int(iterations), newton_refinements=refinements, device=str(x.device), dtype=str(x.dtype),
        private_source_rows_accessed=0, private_source_covariance_accessed=False)


def public_operators(point, target_fit, target_held, adaptation, validation):
    distances = [squared_distance(standardize(point.target_scaler, x), point.target_landmarks)
                 for x in [target_fit, target_held, adaptation, validation]]
    operators = []
    for option in OPTIONS:
        kernels = [torch.exp(-d/(point.target_bandwidth*option['bandwidth_factor'])) for d in distances]
        a = kernels[0]
        matrix = a.T@a + option['ridge']*torch.eye(a.shape[1], device=a.device, dtype=a.dtype)
        regression = torch.linalg.solve(matrix, a.T)
        residual = (matrix@regression-a.T).abs().max().item()
        if residual > 2e-9:
            raise RuntimeError('CUDA kernel regression residual '+str(residual))
        operators.append(dict(option=option, regression=regression, held=kernels[1],
                              adaptation=kernels[2], validation=kernels[3], solve_residual=residual))
    return operators


def calibrated(operators, source_fit, source_held, weight, temperature):
    fit_probability = torch.softmax(source_fit@weight/temperature, dim=1)
    held_probability = torch.softmax(source_held@weight/temperature, dim=1)
    choices = []
    for index, operator in enumerate(operators):
        coefficients = operator['regression']@fit_probability
        logits = operator['held']@coefficients
        for gamma in TEMPERATURES:
            p = torch.softmax(logits/gamma, dim=1)
            error = ((p-held_probability)**2).sum(1).mean()
            choices.append((error.item(), index, gamma, coefficients))
    choice = min(range(len(choices)), key=lambda j: choices[j][0])
    _, selected, gamma, coefficients = choices[choice]
    operator = operators[selected]
    al = torch.log_softmax(operator['adaptation']@coefficients/gamma, dim=1)
    vl = torch.log_softmax(operator['validation']@coefficients/gamma, dim=1)
    return dict(adaptation_log=al, validation_log=vl, adaptation=al.exp(), validation=vl.exp(),
                coefficients=coefficients, selected=selected, projection_temperature=gamma,
                public_choice=choice, public_errors=[v[0] for v in choices])


def offset(logscores, temperature, balance_strength, q, mass=None):
    logscores, q = tensor(logscores), tensor(q)
    if temperature <= 0 or balance_strength < 0:
        raise ValueError('Positive temperature and nonnegative coupling required')
    n, k = logscores.shape
    a = torch.full((n,), 1./n, device=logscores.device, dtype=DTYPE) if mass is None else tensor(mass)
    if (a <= 0).any() or (q <= 0).any():
        raise ValueError('Positive row mass and released prior required')
    a, q = a/a.sum(), q/q.sum()
    if balance_strength == 0:
        return torch.zeros_like(q), dict(dual_gradient_infinity=0., duality_gap=0., iterations=0, device='cuda:0')
    tau, beta = temperature, balance_strength*temperature
    def state(v, hessian=False):
        logits = (logscores-v[None, :])/tau
        p = torch.softmax(logits, dim=1)
        prior = q*torch.exp(v/beta)
        marginal = (a[:, None]*p).sum(0)
        value = beta*(q*torch.expm1(v/beta)).sum()+tau*(a*logits.logsumexp(1)).sum()
        gradient = prior-marginal
        if not hessian:
            return value, gradient
        matrix = torch.diag(prior/beta+marginal/tau)-(p.T*a)@p/tau
        return value, gradient, matrix, p
    v = torch.zeros_like(q)
    # Small K-dimensional convex dual: damped Newton, all matrices on CUDA.
    for iteration in range(100):
        value, gradient, matrix, p = state(v, True)
        if gradient.abs().max().item() <= 1e-11:
            break
        direction = torch.linalg.solve(matrix, -gradient)
        scale = 1.
        for _ in range(60):
            candidate = v+scale*direction
            if (candidate/beta).max().item() > 650:
                scale *= .5; continue
            new_value, new_gradient = state(candidate)
            if new_value <= value+1e-13 and (new_gradient.abs().max() < gradient.abs().max() or
                                           new_value < value+1e-4*scale*(gradient*direction).sum()):
                v = candidate; break
            scale *= .5
        else:
            raise RuntimeError('CUDA dual Newton line search stalled')
    value, gradient, _, p = state(v, True)
    plan = a[:, None]*p
    cost = -logscores+tau*q.log()[None, :]
    reference = a[:, None]*q[None, :]
    marginal = plan.sum(0)
    kl_row = (plan*(plan.clamp_min(1e-300).log()-reference.log())-plan+reference).sum()
    kl_col = (marginal*(marginal.clamp_min(1e-300).log()-q.log())-marginal+q).sum()
    primal = (plan*cost).sum()+tau*kl_row+beta*kl_col
    gap = (primal+value).item()
    error = gradient.abs().max().item()
    if not torch.isfinite(p).all() or error > 2e-9 or not -1e-8 <= gap <= 1e-10:
        raise RuntimeError('CUDA dual certificate failed '+str((error, gap)))
    correction = -v+v.mean()
    return correction, dict(dual_gradient_infinity=error, duality_gap=max(gap, 0.),
                             iterations=iteration+1, device=str(logscores.device))


def predict(logscores, correction, temperature):
    p = torch.softmax((tensor(logscores)+tensor(correction))/temperature, dim=1).clamp_min(1e-12)
    return p/p.sum(1, keepdim=True)
