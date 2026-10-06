"""Bottom/Top models supporting ResNet18 and SmallCNN, N-party, sum/concat."""
import math
import numbers
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(nn.Conv2d(inp, out, 1, stride=stride, bias=False), nn.BatchNorm2d(out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet18Bottom(nn.Module):
    """ResNet18 bottom, output 512-dim."""
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(nn.Conv2d(3,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.layer1 = nn.Sequential(BasicBlock(64,64), BasicBlock(64,64))
        self.layer2 = nn.Sequential(BasicBlock(64,128,2), BasicBlock(128,128))
        self.layer3 = nn.Sequential(BasicBlock(128,256,2), BasicBlock(256,256))
        self.layer4 = nn.Sequential(BasicBlock(256,512,2), BasicBlock(512,512))

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x)))))
        return F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)


class SmallCNNBottom(nn.Module):
    """3conv + 3fc bottom (V-LETO paper), output 128-dim."""
    def __init__(self, input_width=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        h, w = 32//8, max(input_width//8, 1)
        self.fc = nn.Linear(128*h*w, 128)

    def forward(self, x):
        return F.relu(self.fc(self.features(x).view(x.size(0), -1)))


class MLPBottom(nn.Module):
    """Per-party MLP bottom for tabular / multi-view VFL (e.g. mfeat).

    Maps one view's feature dim -> embed_dim via Linear->ReLU->Linear. Each party
    has its OWN input dim (its view's feature count), so build_models sizes one
    MLPBottom per party from the column-range map. Under CONCAT aggregation the
    top head holds a separate weight block per party, so per-class ownership can
    concentrate on the view(s) that discriminate that class.
    """
    def __init__(self, in_dim, embed_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        # x is (B, in_dim); flatten defensively in case of trailing singleton dims
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.net(x)


class TopModel(nn.Module):
    """Single linear classifier (for sum aggregation or concat).

    cosine=True turns it into a LUCIR-style cosine head: both the feature x and
    each class weight row are L2-normalised before the dot product, then scaled
    by a learnable `scale`. This removes per-class weight-norm magnitude from the
    logits, so argmax reflects direction (cosine similarity) only — fixing the
    class-recency bias where the largest-norm class dominates regardless of input.
    `self.classifier` (an nn.Linear) is kept either way so weight introspection
    and expand_classes() keep working.
    """
    def __init__(self, input_dim, num_classes, cosine=False, scale=16.0):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        self.cosine = cosine
        if cosine:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.register_buffer(
            '_logit_calibration_alpha', torch.ones(num_classes)
        )
        self.register_buffer(
            '_logit_calibration_bias', torch.zeros(num_classes)
        )
        self.register_buffer(
            '_logit_calibration_task',
            torch.full((num_classes,), -1, dtype=torch.long),
        )
        self.register_buffer(
            '_logit_calibration_task_weight', torch.tensor(1.0)
        )
        self.register_buffer(
            '_logit_calibration_enabled', torch.tensor(False)
        )
        self.register_buffer(
            '_adaptive_full_weight', torch.empty(0, input_dim)
        )
        self.register_buffer('_adaptive_full_bias', torch.empty(0))
        self.register_buffer(
            '_adaptive_gate', torch.tensor(0.0, dtype=torch.float64)
        )
        self.register_buffer('_adaptive_enabled', torch.tensor(False))
        self.register_buffer(
            '_adaptive_version', torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            '_adaptive_class_order', torch.empty(0, dtype=torch.long)
        )

    def _apply(self, fn, recurse=True):
        gate = self._adaptive_gate
        super()._apply(fn, recurse=recurse)
        if gate.is_meta:
            self._adaptive_gate = torch.empty_like(
                self._adaptive_gate, dtype=torch.float64
            )
        else:
            self._adaptive_gate = gate.to(device=self._adaptive_gate.device)
        return self

    def _ordered_adaptive_classes(self, class_order):
        integer_dtypes = {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }
        if isinstance(class_order, torch.Tensor):
            if class_order.ndim != 1 or class_order.dtype not in integer_dtypes:
                raise ValueError(
                    'adaptive class order must be a one-dimensional integer tensor'
                )
            raw_classes = class_order.detach().cpu().tolist()
        else:
            try:
                raw_classes = list(class_order)
            except TypeError as error:
                raise TypeError(
                    'adaptive class order must be an ordered integer sequence'
                ) from error
        if any(isinstance(class_id, bool) for class_id in raw_classes):
            raise TypeError('adaptive class IDs must be integers')
        try:
            classes = [operator.index(class_id) for class_id in raw_classes]
        except TypeError as error:
            raise TypeError('adaptive class IDs must be integers') from error
        if not classes:
            raise ValueError('adaptive class order must not be empty')
        if any(left >= right for left, right in zip(classes, classes[1:])):
            raise ValueError(
                'adaptive class IDs must be unique and strictly ordered'
            )
        if classes[0] < 0 or classes[-1] >= self.classifier.out_features:
            raise ValueError('adaptive class is outside classifier range')
        return torch.tensor(
            classes, dtype=torch.long, device=self.classifier.weight.device
        )

    def _validated_adaptive_components(self, full_weight, full_bias, gate,
                                       class_order, version):
        if isinstance(version, bool):
            raise TypeError('adaptive version must be an integer')
        try:
            version = operator.index(version)
        except TypeError as error:
            raise TypeError('adaptive version must be an integer') from error
        if version != 1:
            raise ValueError('unsupported adaptive state version')
        if isinstance(gate, bool) or not isinstance(gate, numbers.Real):
            raise TypeError('adaptive gate must be a real scalar')
        gate = float(gate)
        if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
            raise ValueError('adaptive gate must be finite and in [0, 1]')
        if not isinstance(full_weight, torch.Tensor) or not isinstance(
                full_bias, torch.Tensor):
            raise TypeError('adaptive classifier state must use tensors')
        if not full_weight.is_floating_point() or not full_bias.is_floating_point():
            raise TypeError('adaptive classifier state must be floating point')
        classes = self._ordered_adaptive_classes(class_order)
        if full_weight.ndim != 2 or full_weight.shape != (
                classes.numel(), self.classifier.in_features):
            raise ValueError('adaptive full classifier weight shape mismatch')
        if full_bias.ndim != 1 or full_bias.shape != classes.shape:
            raise ValueError('adaptive full classifier bias shape mismatch')
        if not torch.isfinite(full_weight).all() or not torch.isfinite(full_bias).all():
            raise ValueError('adaptive full classifier state must be finite')
        device = self.classifier.weight.device
        dtype = self.classifier.weight.dtype
        weight = full_weight.detach().to(device=device, dtype=dtype).clone()
        bias = full_bias.detach().to(device=device, dtype=dtype).clone()
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError(
                'adaptive full classifier state is not finite after conversion'
            )
        return (
            weight,
            bias,
            torch.tensor(gate, device=device, dtype=torch.float64),
            classes,
            torch.tensor(version, device=device, dtype=torch.long),
        )

    def _validate_adaptive_bias_state(self, state_dict, prefix, class_order):
        calibration_names = (
            '_logit_calibration_alpha', '_logit_calibration_bias',
            '_logit_calibration_task', '_logit_calibration_task_weight',
            '_logit_calibration_enabled',
        )
        required = [
            prefix + 'classifier.weight', prefix + 'classifier.bias',
            *(prefix + name for name in calibration_names),
        ]
        if self.cosine:
            required.append(prefix + 'scale')
        elif prefix + 'scale' in state_dict:
            raise ValueError('adaptive classifier head mode mismatch')
        if any(key not in state_dict for key in required):
            raise ValueError('adaptive Bias branch state is incomplete')
        values = [state_dict[key] for key in required]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise ValueError('adaptive Bias branch fields must be tensors')

        weight = state_dict[prefix + 'classifier.weight']
        bias = state_dict[prefix + 'classifier.bias']
        dtype = self.classifier.weight.dtype
        if (weight.shape != self.classifier.weight.shape or weight.dtype != dtype
                or not weight.is_floating_point()
                or not torch.isfinite(weight).all()):
            raise ValueError('adaptive Bias classifier weight is invalid')
        if (bias.shape != self.classifier.bias.shape or bias.dtype != dtype
                or not bias.is_floating_point()
                or not torch.isfinite(bias).all()):
            raise ValueError('adaptive Bias classifier bias is invalid')
        if self.cosine:
            scale = state_dict[prefix + 'scale']
            if (scale.shape != torch.Size([]) or scale.dtype != self.scale.dtype
                    or not scale.is_floating_point()
                    or not torch.isfinite(scale)):
                raise ValueError('adaptive cosine scale is invalid')

        alpha = state_dict[prefix + '_logit_calibration_alpha']
        calibration_bias = state_dict[prefix + '_logit_calibration_bias']
        task = state_dict[prefix + '_logit_calibration_task']
        task_weight = state_dict[
            prefix + '_logit_calibration_task_weight'
        ]
        enabled = state_dict[prefix + '_logit_calibration_enabled']
        output_shape = torch.Size([self.classifier.out_features])
        for name, value in (
                ('alpha', alpha), ('bias', calibration_bias)):
            if (value.shape != output_shape or value.dtype != dtype
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()):
                raise ValueError(
                    'adaptive Bias calibration ' + name + ' is invalid'
                )
        if task.shape != output_shape or task.dtype != torch.long:
            raise ValueError('adaptive Bias calibration task map is invalid')
        if (task_weight.shape != torch.Size([]) or task_weight.dtype != dtype
                or not task_weight.is_floating_point()
                or not torch.isfinite(task_weight)
                or float(task_weight) <= 0.0):
            raise ValueError('adaptive Bias calibration task weight is invalid')
        if enabled.shape != torch.Size([]) or enabled.dtype != torch.bool:
            raise ValueError('adaptive Bias calibration flag is invalid')

        if bool(enabled):
            expected = torch.zeros_like(task, dtype=torch.bool)
            expected[class_order.to(task.device)] = True
            if (not torch.equal(task >= 0, expected)
                    or not torch.all(task[~expected] == -1)):
                raise ValueError(
                    'adaptive Bias class order and task map do not match'
                )
        elif (not torch.equal(alpha, torch.ones_like(alpha))
              or not torch.equal(calibration_bias, torch.zeros_like(calibration_bias))
              or not torch.all(task == -1)
              or float(task_weight) != 1.0):
            raise ValueError('disabled adaptive Bias calibration is not empty')

    def set_adaptive_mixture(self, full_weight, full_bias, gate, class_order,
                             version=1):
        """Atomically install a validated Full/Bias probability mixture."""
        weight, bias, gate, classes, version = self._validated_adaptive_components(
            full_weight, full_bias, gate, class_order, version
        )
        self._validate_adaptive_bias_state(self.state_dict(), '', classes)
        self._adaptive_full_weight = weight
        self._adaptive_full_bias = bias
        self._adaptive_gate = gate
        self._adaptive_class_order = classes
        self._adaptive_version = version
        self._adaptive_enabled.fill_(True)

    def set_adaptive_endpoint(self, gate, class_order, version=1):
        """Mark the live classifier as an exact Full or Bias endpoint."""
        if gate not in (0.0, 1.0):
            raise ValueError('adaptive endpoint gate must be exactly 0 or 1')
        if isinstance(version, bool):
            raise TypeError('adaptive version must be an integer')
        try:
            version = operator.index(version)
        except TypeError as error:
            raise TypeError('adaptive version must be an integer') from error
        if version != 1:
            raise ValueError('unsupported adaptive state version')
        classes = self._ordered_adaptive_classes(class_order)
        self._validate_adaptive_bias_state(self.state_dict(), '', classes)
        device = self.classifier.weight.device
        dtype = self.classifier.weight.dtype
        self._adaptive_full_weight = torch.empty(
            0, self.classifier.in_features, device=device, dtype=dtype
        )
        self._adaptive_full_bias = torch.empty(0, device=device, dtype=dtype)
        self._adaptive_gate = torch.tensor(
            gate, device=device, dtype=torch.float64
        )
        self._adaptive_class_order = classes
        self._adaptive_version = torch.tensor(
            version, device=device, dtype=torch.long
        )
        self._adaptive_enabled.fill_(True)

    def clear_adaptive_mixture(self):
        device = self.classifier.weight.device
        dtype = self.classifier.weight.dtype
        self._adaptive_full_weight = torch.empty(
            0, self.classifier.in_features, device=device, dtype=dtype
        )
        self._adaptive_full_bias = torch.empty(0, device=device, dtype=dtype)
        self._adaptive_class_order = torch.empty(
            0, device=device, dtype=torch.long
        )
        self._adaptive_gate = torch.tensor(
            0.0, device=device, dtype=torch.float64
        )
        self._adaptive_version = torch.tensor(0, device=device, dtype=torch.long)
        self._adaptive_enabled.fill_(False)

    def clear_logit_calibration(self):
        self._logit_calibration_alpha.fill_(1.0)
        self._logit_calibration_bias.zero_()
        self._logit_calibration_task.fill_(-1)
        self._logit_calibration_task_weight.fill_(1.0)
        self._logit_calibration_enabled.fill_(False)

    def set_logit_calibration(self, classes, alpha, bias, task_for_class,
                              task_weight):
        classes = torch.as_tensor(
            classes, dtype=torch.long, device=self.classifier.weight.device
        )
        alpha = torch.as_tensor(
            alpha, dtype=self.classifier.weight.dtype, device=classes.device
        )
        bias = torch.as_tensor(
            bias, dtype=self.classifier.weight.dtype, device=classes.device
        )
        task_for_class = torch.as_tensor(
            task_for_class, dtype=torch.long, device=classes.device
        )
        if classes.ndim != 1 or classes.numel() == 0:
            raise ValueError('logit calibration requires at least one class')
        if alpha.shape != classes.shape or bias.shape != classes.shape:
            raise ValueError('logit calibration alpha/bias shape mismatch')
        if task_for_class.shape != classes.shape:
            raise ValueError('logit calibration task mapping shape mismatch')
        if int(classes.min()) < 0 or int(classes.max()) >= self.classifier.out_features:
            raise ValueError('logit calibration class is outside classifier range')
        if int(task_for_class.min()) < 0:
            raise ValueError('logit calibration task ids must be non-negative')
        if not torch.isfinite(alpha).all() or not torch.isfinite(bias).all():
            raise ValueError('logit calibration parameters must be finite')
        task_weight = float(task_weight)
        if not torch.isfinite(torch.tensor(task_weight)) or task_weight <= 0:
            raise ValueError('logit calibration task weight must be positive')
        self.clear_logit_calibration()
        self._logit_calibration_alpha[classes] = alpha
        self._logit_calibration_bias[classes] = bias
        self._logit_calibration_task[classes] = task_for_class
        self._logit_calibration_task_weight.fill_(task_weight)
        self._logit_calibration_enabled.fill_(True)

    def _apply_logit_calibration(self, logits):
        if not bool(self._logit_calibration_enabled):
            return logits
        calibrated = (
            logits * self._logit_calibration_alpha
            + self._logit_calibration_bias
        )
        output = logits.clone()
        task_ids = torch.unique(
            self._logit_calibration_task[
                self._logit_calibration_task >= 0
            ],
            sorted=True,
        )
        evidence = torch.stack([
            torch.logsumexp(
                calibrated[:, self._logit_calibration_task == task_id], dim=1
            )
            for task_id in task_ids
        ], dim=1)
        task_log_probability = F.log_softmax(evidence, dim=1)
        for position, task_id in enumerate(task_ids):
            mask = self._logit_calibration_task == task_id
            within = F.log_softmax(calibrated[:, mask], dim=1)
            output[:, mask] = (
                within
                + self._logit_calibration_task_weight
                * task_log_probability[:, position, None]
            )
        return output

    def _classifier_logits(self, x, weight, bias):
        if self.cosine:
            xn = F.normalize(x, dim=1)
            wn = F.normalize(weight, dim=1)
            logits = self.scale * F.linear(xn, wn)   # bias intentionally unused
        else:
            logits = F.linear(x, weight, bias)
        return logits

    def _bias_logits(self, x):
        if self.cosine:
            return self._classifier_logits(
                x, self.classifier.weight, self.classifier.bias
            )
        return self.classifier(x)

    def branch_log_probabilities(self, x):
        if not bool(self._adaptive_enabled):
            raise RuntimeError('adaptive mixture is not enabled')
        if self._adaptive_full_weight.numel() == 0:
            logits = self._bias_logits(x)
            if float(self._adaptive_gate) == 0.0:
                logits = self._apply_logit_calibration(logits)
            logits = logits.index_select(1, self._adaptive_class_order)
            selected = F.log_softmax(logits.to(torch.float64), dim=1)
            if not torch.isfinite(selected).all():
                raise ValueError(
                    'adaptive endpoint log-probabilities must be finite'
                )
            return selected.clone(), selected.clone()
        full_logits = self._classifier_logits(
            x, self._adaptive_full_weight, self._adaptive_full_bias
        )
        bias_logits = self._apply_logit_calibration(self._bias_logits(x))
        bias_logits = bias_logits.index_select(1, self._adaptive_class_order)
        log_p_full = F.log_softmax(full_logits.to(torch.float64), dim=1)
        log_p_bias = F.log_softmax(bias_logits.to(torch.float64), dim=1)
        if not torch.isfinite(log_p_full).all() or not torch.isfinite(log_p_bias).all():
            raise ValueError('adaptive branch log-probabilities must be finite')
        return log_p_full, log_p_bias

    def forward(self, x):
        if not bool(self._adaptive_enabled):
            return self._apply_logit_calibration(self._bias_logits(x))
        from adaptive_head_consolidation import mix_log_probabilities
        log_p_full, log_p_bias = self.branch_log_probabilities(x)
        return mix_log_probabilities(
            log_p_full, log_p_bias, float(self._adaptive_gate)
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        adaptive_names = (
            '_adaptive_full_weight', '_adaptive_full_bias', '_adaptive_gate',
            '_adaptive_enabled', '_adaptive_version', '_adaptive_class_order',
        )
        adaptive_keys = [prefix + name for name in adaptive_names]
        present = [key in state_dict for key in adaptive_keys]
        adaptive_payload_enabled = False
        if not any(present):
            self.clear_adaptive_mixture()
            for name, key in zip(adaptive_names, adaptive_keys):
                state_dict[key] = getattr(self, name).detach().clone()
        elif not all(present):
            error_msgs.append(prefix + 'adaptive state is incomplete')
            for name, key in zip(adaptive_names, adaptive_keys):
                state_dict[key] = getattr(self, name).detach().clone()
        else:
            adaptive = {
                name: state_dict[key]
                for name, key in zip(adaptive_names, adaptive_keys)
            }
            try:
                if any(not isinstance(value, torch.Tensor)
                       for value in adaptive.values()):
                    raise ValueError('adaptive state fields must be tensors')
                full_weight = adaptive['_adaptive_full_weight']
                full_bias = adaptive['_adaptive_full_bias']
                gate = adaptive['_adaptive_gate']
                enabled = adaptive['_adaptive_enabled']
                version = adaptive['_adaptive_version']
                class_order = adaptive['_adaptive_class_order']
                if (full_weight.ndim != 2
                        or full_weight.shape[1] != self.classifier.in_features
                        or not full_weight.is_floating_point()
                        or full_weight.dtype != self.classifier.weight.dtype):
                    raise ValueError('adaptive Full weight has invalid schema')
                if (full_bias.ndim != 1 or not full_bias.is_floating_point()
                        or full_bias.dtype != self.classifier.weight.dtype):
                    raise ValueError('adaptive Full bias has invalid schema')
                if (gate.shape != torch.Size([])
                        or gate.dtype != torch.float64):
                    raise ValueError('adaptive gate must be a float64 scalar')
                if (not isinstance(enabled, torch.Tensor)
                        or enabled.shape != torch.Size([])
                        or enabled.dtype != torch.bool):
                    raise ValueError('adaptive enabled flag must be a bool scalar')
                if (version.shape != torch.Size([])
                        or version.dtype != torch.long):
                    raise ValueError('adaptive version must be a long scalar')
                if class_order.ndim != 1 or class_order.dtype != torch.long:
                    raise ValueError('adaptive class order must be a long vector')
                if bool(enabled):
                    adaptive_payload_enabled = True
                    if full_weight.numel() == 0 and full_bias.numel() == 0:
                        if (full_weight.shape != torch.Size([
                                0, self.classifier.in_features])
                                or float(gate) not in (0.0, 1.0)
                                or int(version) != 1):
                            raise ValueError('adaptive endpoint state is invalid')
                        self._ordered_adaptive_classes(class_order)
                    else:
                        self._validated_adaptive_components(
                            full_weight, full_bias, float(gate), class_order,
                            int(version),
                        )
                    self._validate_adaptive_bias_state(
                        state_dict, prefix, class_order
                    )
                    self._adaptive_full_weight = torch.empty(
                        adaptive['_adaptive_full_weight'].shape,
                        device=self.classifier.weight.device,
                        dtype=self.classifier.weight.dtype,
                    )
                    self._adaptive_full_bias = torch.empty(
                        adaptive['_adaptive_full_bias'].shape,
                        device=self.classifier.weight.device,
                        dtype=self.classifier.weight.dtype,
                    )
                    self._adaptive_class_order = torch.empty(
                        adaptive['_adaptive_class_order'].shape,
                        device=self.classifier.weight.device,
                        dtype=torch.long,
                    )
                else:
                    if (int(adaptive['_adaptive_version']) != 0
                            or float(adaptive['_adaptive_gate']) != 0.0
                            or adaptive['_adaptive_full_weight'].shape
                            != torch.Size([0, self.classifier.in_features])
                            or adaptive['_adaptive_full_bias'].numel() != 0
                            or adaptive['_adaptive_class_order'].numel() != 0):
                        raise ValueError('disabled adaptive state is not empty')
                    self.clear_adaptive_mixture()
            except (TypeError, ValueError, RuntimeError) as error:
                error_msgs.append(prefix + 'invalid adaptive state: ' + str(error))
                for name, key in zip(adaptive_names, adaptive_keys):
                    state_dict[key] = getattr(self, name).detach().clone()
        calibration_defaults = {
            '_logit_calibration_alpha': torch.ones_like(
                self._logit_calibration_alpha
            ),
            '_logit_calibration_bias': torch.zeros_like(
                self._logit_calibration_bias
            ),
            '_logit_calibration_task': torch.full_like(
                self._logit_calibration_task, -1
            ),
            '_logit_calibration_task_weight': torch.ones_like(
                self._logit_calibration_task_weight
            ),
            '_logit_calibration_enabled': torch.zeros_like(
                self._logit_calibration_enabled
            ),
        }
        calibration_names = tuple(calibration_defaults)
        if adaptive_payload_enabled and any(
                prefix + name not in state_dict for name in calibration_names):
            error_msgs.append(prefix + 'adaptive Bias branch state is incomplete')
        # Checkpoints created before V2 do not contain calibration buffers.
        for name in calibration_names:
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = calibration_defaults[name]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def expand_classes(self, new_num, device):
        old_num = self.classifier.out_features
        if new_num <= old_num:
            return
        if bool(self._adaptive_enabled):
            raise RuntimeError(
                'cannot expand classes while adaptive mixture is enabled'
            )
        old_w, old_b = self.classifier.weight.data, self.classifier.bias.data
        self.classifier = nn.Linear(self.classifier.in_features, new_num).to(device)
        self.classifier.weight.data[:old_num] = old_w
        self.classifier.bias.data[:old_num] = old_b
        alpha = torch.ones(new_num, device=device)
        bias = torch.zeros(new_num, device=device)
        task = torch.full((new_num,), -1, dtype=torch.long, device=device)
        alpha[:old_num] = self._logit_calibration_alpha
        bias[:old_num] = self._logit_calibration_bias
        task[:old_num] = self._logit_calibration_task
        self._logit_calibration_alpha = alpha
        self._logit_calibration_bias = bias
        self._logit_calibration_task = task


def build_models(args):
    """Build N bottom models + 1 top model.

    ResNet18 bottom is width-agnostic (global avg-pool), so feature-quantity
    heterogeneity (unequal party widths) needs no special handling there.
    SmallCNN bottoms are sized per-party from get_party_widths.
    """
    from data_utils import get_party_widths
    widths = get_party_widths(args)
    if args.model_type == 'mlp':
        # widths[i] = view i's feature count (per-party input dim). embed_dim is a
        # modest 128 for tabular (set in config for model_type==mlp).
        embed = getattr(args, 'embed_dim', 128)
        bottoms = [MLPBottom(in_dim=widths[i], embed_dim=embed).to(args.device)
                   for i in range(args.num_parties)]
    elif args.model_type == 'small_cnn':
        bottoms = [SmallCNNBottom(input_width=widths[i]).to(args.device) for i in range(args.num_parties)]
        embed = 128
    else:
        bottoms = [ResNet18Bottom().to(args.device) for _ in range(args.num_parties)]
        embed = 512

    if args.aggregation == 'sum':
        top_dim = embed
    else:
        top_dim = embed * args.num_parties

    # Fixed full-width head (num_classes) from the start, matching canonical CIL
    # implementations (Mammoth / PyCIL). expand_classes() then becomes a no-op
    # (its `new_num <= old_num` guard returns early), removing per-task head growth
    # and the logit-width mismatches it caused.
    top = TopModel(top_dim, args.num_classes,
                   cosine=getattr(args, 'cosine_head', False)).to(args.device)
    return bottoms, top
