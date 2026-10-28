"""Weighted, global actor gradient accounting and symmetric FB/style PCGrad."""

import torch
import torch.distributed as distributed


OBJECTIVES = ('fb', 'style', 'auxiliary')


def require_finite(values, world_size):
    finite = torch.stack([torch.isfinite(value.detach()).all() for value in values]).all().float()
    if world_size > 1:
        distributed.all_reduce(finite, op=distributed.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError('Nonfinite actor objective or gradient on one or more ranks')


def gradient_vector(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
    return torch.cat([gradient.detach().reshape(-1) for gradient in gradients])


def ratio_metric(metrics, name, numerator, denominator):
    valid = denominator != 0
    metrics[name + '_valid'] = valid.to(numerator.dtype)
    metrics[name + '_zero_denominator'] = (~valid).to(numerator.dtype)
    if valid.item():
        metrics[name] = numerator / denominator


def vector_metrics(vectors, prefix):
    metrics = {prefix + name + '_norm': vector.double().norm() for name, vector in vectors.items()}
    for left, right in (('fb', 'style'), ('fb', 'auxiliary'), ('style', 'auxiliary')):
        if left in vectors and right in vectors:
            first, second = vectors[left].double(), vectors[right].double()
            ratio_metric(metrics, prefix + left + '_' + right + '_cosine',
                         first.dot(second), first.norm() * second.norm())
    return metrics


def symmetric_projection(vectors):
    """Project both original weighted global vectors; auxiliary is unchanged."""
    forward, style, auxiliary = vectors
    forward64, style64 = forward.double(), style.double()
    inner = forward64.dot(style64)
    forward_square = forward64.square().sum()
    style_square = style64.square().sum()
    conflict = (inner < 0) & (forward_square > 0) & (style_square > 0)
    if conflict.item():
        projected_forward = (forward64 - inner / style_square * style64).to(forward.dtype)
        projected_style = (style64 - inner / forward_square * forward64).to(style.dtype)
    else:
        projected_forward, projected_style = forward.clone(), style.clone()
    pair = projected_forward + projected_style
    combined = pair + auxiliary
    metrics = {'pcgrad/conflict': conflict.float(), 'pcgrad/original_inner': inner,
               'pcgrad/fb_square': forward_square, 'pcgrad/style_square': style_square,
               'pcgrad/gradient_norm': combined.double().norm(),
               'pcgrad/original_pair_norm': (forward64 + style64).norm(),
               'pcgrad/projected_pair_norm': pair.double().norm()}
    ratio_metric(metrics, 'pcgrad/pair_norm_ratio', pair.double().norm(), (forward64 + style64).norm())
    ratio_metric(metrics, 'pcgrad/pair_final_cosine', pair.double().dot(combined.double()),
                 pair.double().norm() * combined.double().norm())
    if conflict.item():
        metrics['pcgrad/fb_projection_residual'] = projected_forward.double().dot(style64).abs()
        metrics['pcgrad/style_projection_residual'] = projected_style.double().dot(forward64).abs()
    return combined, metrics


class ActorGradientAccumulator:
    """Accumulate original sampled base/local/goal graphs without holding all three."""

    def __init__(self, parameters, world_size, probe):
        self.parameters = tuple(parameters)
        self.world_size = world_size
        self.probe = probe
        self.vectors = None
        self.metrics = {}
        self.objective_backward_calls = 0
        self.strata_backward_calls = 0
        self.before = torch.cat([parameter.detach().reshape(-1) for parameter in self.parameters]) if probe else None

    def add(self, q_values, coefficients, weight=1.0, label='base', strata=None):
        scale = q_values[0].detach().abs().mean()
        objectives = (q_values[0], scale * coefficients[0] * q_values[1],
                      scale * coefficients[1] * q_values[2])
        losses = [-weight * objective.mean() for objective in objectives]
        require_finite([*q_values, *losses], self.world_size)
        if self.probe:
            for name, mask in (strata or {}).items():
                count = mask.sum().float()
                if self.world_size > 1:
                    distributed.all_reduce(count)
                prefix = 'pcgrad_strata/' + label + '/' + name + '/'
                self.metrics[prefix + 'count'] = count
                if not count.item():
                    continue
                vectors = torch.stack([gradient_vector(-weight * (objective * mask).mean(), self.parameters)
                                       for objective in objectives])
                self.strata_backward_calls += len(objectives)
                if self.world_size > 1:
                    distributed.all_reduce(vectors)
                    vectors.div_(self.world_size)
                require_finite([vectors], self.world_size)
                self.metrics.update(vector_metrics(dict(zip(OBJECTIVES, vectors)), prefix))
        vectors = torch.stack([gradient_vector(loss, self.parameters, retain_graph=index < 2)
                               for index, loss in enumerate(losses)])
        self.objective_backward_calls += len(losses)
        require_finite([vectors], self.world_size)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors.add_(vectors)
        if self.probe:
            branch = vectors.clone()
            if self.world_size > 1:
                distributed.all_reduce(branch)
                branch.div_(self.world_size)
            require_finite([branch], self.world_size)
            self.metrics.update(vector_metrics(dict(zip(OBJECTIVES, branch)), 'pcgrad_branch/' + label + '/'))

    def assign(self):
        if self.world_size > 1:
            distributed.all_reduce(self.vectors)
            self.vectors.div_(self.world_size)
        require_finite([self.vectors], self.world_size)
        combined, metrics = symmetric_projection(self.vectors)
        require_finite([combined, *metrics.values()], self.world_size)
        self.metrics.update(metrics)
        offset = 0
        for parameter in self.parameters:
            size = parameter.numel()
            parameter.grad = combined[offset:offset + size].reshape_as(parameter).clone()
            offset += size
        if self.probe:
            self.metrics.update(vector_metrics(dict(zip(OBJECTIVES, self.vectors)), 'pcgrad_full/'))
            forward = self.vectors[0].double()
            ratio_metric(self.metrics, 'pcgrad_full/fb_final_cosine',
                         forward.dot(combined.double()), forward.norm() * combined.double().norm())

    def after_step(self, maximum_norm, actual_norm):
        denominator = actual_norm.double() + 1e-6  # Exact torch clip_grad_norm_ implementation.
        self.metrics['pcgrad/clip_scale'] = (maximum_norm / denominator).clamp(max=1)
        self.metrics['pcgrad/clipped'] = (self.metrics['pcgrad/clip_scale'] < 1).float()
        self.metrics['pcgrad/objective_backward_calls'] = actual_norm.new_tensor(self.objective_backward_calls)
        self.metrics['pcgrad/strata_backward_calls'] = actual_norm.new_tensor(self.strata_backward_calls)
        if self.probe:
            update = torch.cat([parameter.detach().reshape(-1) for parameter in self.parameters]) - self.before
            self.metrics['pcgrad_full/update_norm'] = update.double().norm()
            for name, vector in zip(OBJECTIVES, self.vectors):
                inner = vector.double().dot(update.double())
                self.metrics['pcgrad_full/' + name + '_update_dot'] = inner
                ratio_metric(self.metrics, 'pcgrad_full/' + name + '_descent_update_cosine',
                             -inner, vector.double().norm() * update.double().norm())
        require_finite(list(self.metrics.values()), self.world_size)
        return self.metrics
