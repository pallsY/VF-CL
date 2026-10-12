"""Online Curvature-Aware Replay with block K-FAC in split VFL."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import split_features
from .er import ERCL


def kfac_precondition(gradient, activation_rows, output_covariance, damping):
    """Apply damped G^-1 grad A^-1 with low-rank A via Woodbury."""
    if (gradient.ndim != 2 or activation_rows.ndim != 2
            or output_covariance.ndim != 2
            or gradient.size(1) != activation_rows.size(1)
            or output_covariance.shape != (gradient.size(0), gradient.size(0))
            or activation_rows.size(0) == 0 or damping <= 0):
        raise ValueError('invalid OCAR K-FAC factors')
    out_dim = gradient.size(0)
    left = torch.linalg.solve(
        output_covariance + damping * torch.eye(
            out_dim, dtype=gradient.dtype, device=gradient.device), gradient)
    rows = activation_rows.to(dtype=gradient.dtype, device=gradient.device)
    n = len(rows)
    small = torch.eye(n, dtype=gradient.dtype, device=gradient.device)
    small = small + rows @ rows.T / (n * damping)
    correction = (left @ rows.T) @ torch.linalg.solve(small, rows)
    return left / damping - correction / (n * damping * damping)


class OCARCL(ERCL):
    """ER gradient with model-Fisher K-FAC and replay-weighted curvature."""
    def __init__(self, trainer, args):
        super().__init__(trainer, args)
        self.name = 'OCAR_VFL'
        self.ema = float(getattr(args, 'ocar_ema', 0.01))
        self.delta_tau = float(getattr(args, 'ocar_delta_tau', 1e-5))
        self.max_rows = int(getattr(args, 'ocar_max_rows', 32))
        if not 0 < self.ema <= 1 or self.delta_tau < 0 or self.max_rows < 1:
            raise ValueError('invalid OCAR curvature parameters')
        self.tau = float(args.lr)
        self.steps = 0
        self.lambda_buffer = 1.0
        self.factors = {}
        self.current_batch = None
        self.replay_batch = None

    def _modules(self):
        named = {}
        for party, bottom in enumerate(self.trainer.bottoms):
            for name, module in bottom.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    named[f'bottom{party}.{name}'] = module
        for name, module in self.trainer.top_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                named[f'top.{name}'] = module
        return named

    def before_task(self, task_id, new_classes, seen_classes):
        old_top = self.trainer.top_model.classifier
        old_factor = self.factors.get(old_top)
        super().before_task(task_id, new_classes, seen_classes)
        self.lambda_buffer = len(seen_classes) / max(len(new_classes), 1)
        new_top = self.trainer.top_model.classifier
        if old_top is not new_top and old_factor is not None:
            self.factors[new_top] = (old_factor[0], None)
            self.factors.pop(old_top, None)

    def _forward(self, examples):
        pieces = split_features(examples, self.args)
        embeddings = [bottom(piece) for bottom, piece in
                      zip(self.trainer.bottoms, pieces)]
        self.trainer.bytes_transmitted += sum(
            value.numel() * value.element_size() * 2 for value in embeddings)
        self.trainer.comm_rounds += 1
        return self.trainer.top_model(self.trainer._aggregate(embeddings))

    def _ocar_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        self.current_batch = batch_x.detach()
        replay = self.buffer.sample(self.buffer_batch_size)
        if replay is None:
            self.replay_batch = None
            return loss_ce
        old_x, old_y = replay
        self.replay_batch = old_x.to(self.args.device)
        logits = self._forward(self.replay_batch)
        return loss_ce + F.cross_entropy(
            logits, old_y.to(self.args.device))

    @staticmethod
    def _rows(module, source, grad_output, max_rows):
        if isinstance(module, nn.Linear):
            z = source.reshape(-1, source.size(-1))
            g = grad_output.reshape(-1, grad_output.size(-1))
        else:
            patches = F.unfold(source, module.kernel_size,
                               dilation=module.dilation,
                               padding=module.padding, stride=module.stride)
            z = patches.transpose(1, 2).reshape(-1, patches.size(1))
            g = grad_output.permute(0, 2, 3, 1).reshape(-1, grad_output.size(1))
        if len(z) > max_rows:
            indices = torch.linspace(0, len(z) - 1,
                                     max_rows, device=z.device).long()
            z, g = z[indices], g[indices]
        if module.bias is not None:
            z = torch.cat((z, torch.ones_like(z[:, :1])), dim=1)
        return z.float(), g.float()

    def _update_factors(self):
        if self.current_batch is None:
            raise RuntimeError('OCAR requires the current training batch')
        modules = self._modules()
        inputs, outputs = {}, {}
        handles = []
        all_models = [*self.trainer.bottoms, self.trainer.top_model]
        modes = [model.training for model in all_models]
        parameters = [parameter for model in all_models
                      for parameter in model.parameters()]
        saved_grads = [None if parameter.grad is None else parameter.grad.clone()
                       for parameter in parameters]

        def save_input(module, values):
            inputs[module] = values[0].detach()

        def save_output(module, _values, value):
            if value.requires_grad:
                value.register_hook(
                    lambda gradient, key=module:
                    outputs.__setitem__(key, gradient.detach()))

        try:
            for model in all_models:
                model.eval()
            for parameter in parameters:
                parameter.grad = None
            for module in modules.values():
                handles.append(module.register_forward_pre_hook(save_input))
                handles.append(module.register_forward_hook(save_output))
            parts = [self.current_batch]
            if self.replay_batch is not None:
                parts.append(self.replay_batch)
            examples = torch.cat(parts, 0)
            logits = self._forward(examples)
            # Labels are drawn from the predictive distribution: true Fisher,
            # not empirical Fisher based on the observed class labels.
            pseudo = torch.multinomial(logits.detach().softmax(1), 1).squeeze(1)
            weights = torch.ones(len(examples), device=examples.device)
            if self.replay_batch is not None:
                weights[len(self.current_batch):] = math.sqrt(
                    1.0 + self.lambda_buffer)
            pseudo_loss = (F.cross_entropy(logits, pseudo, reduction='none')
                           * weights).sum()
            pseudo_loss.backward()
            for module in modules.values():
                if module not in outputs or module not in inputs:
                    continue
                z, g = self._rows(module, inputs[module],
                                  outputs[module], self.max_rows)
                new_g = g.T @ g / len(g)
                previous = self.factors.get(module)
                if previous is None:
                    self.factors[module] = (z, new_g)
                else:
                    old_z, old_g = previous
                    if old_g is None or old_g.shape != new_g.shape:
                        updated_g = new_g
                    else:
                        updated_g = (1 - self.ema) * old_g + self.ema * new_g
                    combined = torch.cat((
                        old_z * math.sqrt(1 - self.ema),
                        z * math.sqrt(self.ema)), 0)
                    if len(combined) > self.max_rows:
                        indices = torch.linspace(
                            0, len(combined) - 1, self.max_rows,
                            device=combined.device).long()
                        combined = combined[indices]
                    self.factors[module] = (combined, updated_g)
        finally:
            for handle in handles:
                handle.remove()
            for model, mode in zip(all_models, modes):
                model.train(mode)
            for parameter, gradient in zip(parameters, saved_grads):
                parameter.grad = gradient

    def _precondition(self, models):
        damping = math.sqrt(self.tau)
        for model in models:
            for module in model.modules():
                factor = self.factors.get(module)
                if factor is None or factor[1] is None or module.weight.grad is None:
                    continue
                activation_rows, output_covariance = factor
                weight = module.weight.grad
                gradient = weight.reshape(weight.size(0), -1)
                bias = (None if module.bias is None or module.bias.grad is None
                        else module.bias.grad)
                if bias is not None:
                    gradient = torch.cat((gradient, bias[:, None]), dim=1)
                adjusted = kfac_precondition(
                    gradient, activation_rows, output_covariance, damping)
                weight.copy_(adjusted[:, :weight[0].numel()].reshape_as(weight))
                if bias is not None:
                    bias.copy_(adjusted[:, -1])

    def _transform_top(self, models):
        self.tau += self.delta_tau
        self._update_factors()
        self._precondition(models)
        self.steps += 1

    def train_task(self, train_loader, task_id):
        self.trainer.gradient_transform_top = self._transform_top
        self.trainer.gradient_transform_bottom = self._precondition
        try:
            return self.trainer.train_task(
                train_loader, self.args.epochs_per_task,
                extra_loss_fn=self._ocar_loss)
        finally:
            del self.trainer.gradient_transform_top
            del self.trainer.gradient_transform_bottom

    def get_state(self):
        state = super().get_state()
        state.update(
            tau=self.tau, steps=self.steps,
            factors={name: (rows.detach().cpu(),
                            None if covariance is None else covariance.detach().cpu())
                     for name, module in self._modules().items()
                     if module in self.factors
                     for rows, covariance in [self.factors[module]]},
        )
        return state

    def load_state(self, state):
        super().load_state(state)
        self.tau, self.steps = float(state['tau']), int(state['steps'])
        modules = self._modules()
        unknown = set(state['factors']) - set(modules)
        if unknown:
            raise ValueError('OCAR checkpoint contains unknown layers')
        self.factors = {
            modules[name]: (rows.to(self.args.device),
                            None if covariance is None else
                            covariance.to(self.args.device))
            for name, (rows, covariance) in state['factors'].items()
        }
