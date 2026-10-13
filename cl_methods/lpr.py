"""Layerwise Proximal Replay preconditioner for split VFL networks."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import split_features
from .er import ERCL


def precondition_affine_gradient(activations, weight_gradient,
                                 bias_gradient, omega, normalizer=None):
    """Apply (I + omega/n * Z.T Z)^-1 to an affine layer gradient.

    Woodbury gives the same inverse using an n-by-n solve when a layer's input
    dimension is large (notably ResNet convolutional patches).
    """
    if (activations.ndim != 2 or weight_gradient.ndim != 2
            or activations.size(0) == 0
            or activations.size(1) != weight_gradient.size(1)
            or omega < 0):
        raise ValueError('invalid LPR layer activations or gradient')
    if bias_gradient is not None:
        if bias_gradient.shape != weight_gradient.shape[:1]:
            raise ValueError('invalid LPR bias gradient')
        z = torch.cat((activations, torch.ones_like(activations[:, :1])), dim=1)
        gradient = torch.cat((weight_gradient, bias_gradient[:, None]), dim=1)
    else:
        z, gradient = activations, weight_gradient
    coefficient = float(omega) / (z.size(0) if normalizer is None
                                  else normalizer)
    if coefficient == 0:
        result = gradient
    else:
        # (I + c Z.T Z)^-1 G.T = G.T - c Z.T(I+c ZZ.T)^-1 Z G.T
        system = torch.eye(z.size(0), device=z.device, dtype=z.dtype)
        system = system + coefficient * (z @ z.T)
        rhs = z @ gradient.T
        result = gradient - coefficient * torch.linalg.solve(system, rhs).T @ z
    if bias_gradient is None:
        return result, None
    return result[:, :-1], result[:, -1]


class LPRCL(ERCL):
    """ER loss with paper Eq. 8-9 applied to each VFL model layer."""
    def __init__(self, trainer, args):
        super().__init__(trainer, args)
        self.name = 'LPR_VFL'
        self.omega0 = float(getattr(args, 'lpr_omega0', 1.0))
        self.beta = float(getattr(args, 'lpr_beta', 1.0))
        self.interval = int(getattr(args, 'lpr_interval', 10))
        self.reference_size = int(getattr(args, 'lpr_reference_size', 8))
        self.max_rows = int(getattr(args, 'lpr_max_rows', 64))
        if (self.omega0 < 0 or self.beta < 0 or self.interval < 1
                or self.reference_size < 1 or self.max_rows < 1):
            raise ValueError('invalid LPR preconditioner settings')
        self.steps = 0
        self.references = {}
        self.reference_count = 0

    def before_task(self, task_id, new_classes, seen_classes):
        super().before_task(task_id, new_classes, seen_classes)
        self.references = {}
        self.steps = 0
        self.reference_count = 0

    def _record_inputs(self, module, inputs):
        value = inputs[0].detach()
        if isinstance(module, nn.Linear):
            rows, n_eff = value.reshape(-1, value.size(-1)), 1
        elif isinstance(module, nn.Conv2d):
            patches = F.unfold(
                value, module.kernel_size, dilation=module.dilation,
                padding=module.padding, stride=module.stride)
            n_eff = patches.size(-1)
            rows = patches.transpose(1, 2).reshape(-1, patches.size(1))
        elif isinstance(module, nn.BatchNorm2d):
            n_eff = value.size(2) * value.size(3)
            if module.running_mean is None or module.running_var is None:
                raise ValueError('LPR BatchNorm requires running statistics')
            mean = module.running_mean[None, :, None, None]
            scale = torch.sqrt(module.running_var[None, :, None, None]
                               + module.eps)
            normalized = (value - mean) / scale
            rows = normalized.permute(0, 2, 3, 1).reshape(-1, value.size(1))
        else:
            return
        total_rows = len(rows)
        if total_rows > self.max_rows:
            indices = torch.randperm(total_rows, device=rows.device)[:self.max_rows]
            rows = rows[indices]
        row_mass = total_rows / len(rows)
        omega = self.omega0 / n_eff ** self.beta
        self.references[module] = (rows.float().clone(), omega, row_mass)

    @torch.no_grad()
    def _refresh(self):
        stored = [example for examples in self.buffer.data.values()
                  for example in examples]
        if not stored:
            self.references = {}
            return
        picked = random.sample(stored, min(self.reference_size, len(stored)))
        self.reference_count = len(picked)
        examples = torch.stack(picked).to(self.args.device)
        modules = [*self.trainer.bottoms, self.trainer.top_model]
        training = [module.training for module in modules]
        handles = []
        self.references = {}
        try:
            for model in modules:
                model.eval()
                for submodule in model.modules():
                    if isinstance(submodule, (nn.Linear, nn.Conv2d,
                                              nn.BatchNorm2d)):
                        handles.append(submodule.register_forward_pre_hook(
                            self._record_inputs))
            pieces = split_features(examples, self.args)
            embeddings = [bottom(piece) for bottom, piece in
                          zip(self.trainer.bottoms, pieces)]
            self.trainer.top_model(self.trainer._aggregate(embeddings))
            self.trainer.bytes_transmitted += sum(
                value.numel() * value.element_size() * 2
                for value in embeddings)
            self.trainer.comm_rounds += 1
        finally:
            for handle in handles:
                handle.remove()
            for module, was_training in zip(modules, training):
                module.train(was_training)

    def _transform_modules(self, modules):
        for model in modules:
            for module in model.modules():
                reference = self.references.get(module)
                if reference is None or module.weight.grad is None:
                    continue
                rows, omega, row_mass = reference
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    gradient = module.weight.grad
                    flat = gradient.reshape(gradient.size(0), -1)
                    bias = (None if module.bias is None or module.bias.grad is None
                            else module.bias.grad)
                    adjusted, adjusted_bias = precondition_affine_gradient(
                        rows, flat, bias, omega,
                        normalizer=self.reference_count / row_mass)
                    gradient.copy_(adjusted.reshape_as(gradient))
                    if adjusted_bias is not None:
                        bias.copy_(adjusted_bias)
                elif isinstance(module, nn.BatchNorm2d):
                    coefficient = omega * row_mass / self.reference_count
                    a = 1 + coefficient * rows.square().sum(0)
                    b = coefficient * rows.sum(0)
                    d = 1 + coefficient * len(rows)
                    determinant = a * d - b.square()
                    gamma = module.weight.grad.clone()
                    beta = (module.bias.grad.clone()
                            if module.bias is not None and module.bias.grad is not None
                            else torch.zeros_like(gamma))
                    module.weight.grad.copy_((d * gamma - b * beta) / determinant)
                    if module.bias is not None and module.bias.grad is not None:
                        module.bias.grad.copy_((-b * gamma + a * beta) / determinant)

    def _transform_top(self, modules):
        if not self.references or self.steps % self.interval == 0:
            self._refresh()
        self._transform_modules(modules)
        self.steps += 1

    def train_task(self, train_loader, task_id):
        self.trainer.gradient_transform_top = self._transform_top
        self.trainer.gradient_transform_bottom = self._transform_modules
        try:
            return super().train_task(train_loader, task_id)
        finally:
            del self.trainer.gradient_transform_top
            del self.trainer.gradient_transform_bottom
