"""Core VFL trainer supporting N parties with sum/concat aggregation."""
import json, math, os, torch, torch.nn as nn, torch.nn.functional as F, time
from copy import deepcopy
from data_utils import split_features
from determinism import tensor_sha256


class VFLTrainer:
    def __init__(self, bottoms, top_model, args):
        self.bottoms = bottoms        # list of N bottom models
        self.top_model = top_model
        self.args = args
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.comm_rounds = 0
        self.bytes_transmitted = 0
        # Optional CE class-slice (opt-in, set by a CL method in before_task).
        # ce_hi=None => full-head CE over global labels (default; all methods unchanged).
        # Set ce_lo/ce_hi to restrict CE to logit columns [ce_lo:ce_hi] with labels
        # relabeled (batch_y - ce_lo) -- canonical class-IL LwF (PyCIL/FACIL) computes
        # CE only on the NEW-class slice so old-class columns aren't suppressed each batch.
        self.ce_lo = 0
        self.ce_hi = None
        # Exact ordered columns, opt-in for noncontiguous current-task CE.
        # None preserves the existing full-head / ce_lo:ce_hi behavior.
        self.ce_classes = None

    def _aggregate(self, embeddings):
        """Aggregate party embeddings: sum or concat."""
        if self.args.aggregation == 'sum':
            return sum(embeddings)
        else:
            return torch.cat(embeddings, dim=1)

    def _create_optimizers(self, lr=None):
        lr = self.args.lr if lr is None else lr
        bottom_lr = lr * getattr(self.args, 'bottom_lr_scale', 1.0)
        optimizer = getattr(self.args, 'optimizer', 'sgd')
        if bottom_lr < 0:
            raise ValueError('bottom_lr_scale must produce a non-negative learning rate')
        if optimizer == 'sgd':
            def build(parameters, rate):
                return torch.optim.SGD(
                    parameters, lr=rate, momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay,
                )
        elif optimizer == 'adamw':
            def build(parameters, rate):
                return torch.optim.AdamW(
                    parameters, lr=rate, weight_decay=self.args.weight_decay,
                )
        else:
            raise ValueError(f'unsupported optimizer: {optimizer}')
        opts_b = [build(bottom.parameters(), bottom_lr) for bottom in self.bottoms]
        opt_t = build(self.top_model.parameters(), lr)
        return opts_b, opt_t

    def train_epoch(self, train_loader, optimizers, extra_loss_fn=None,
                    grad_clip_norm=None):
        opts_b, opt_t = optimizers
        for b in self.bottoms: b.train()
        self.top_model.train()
        total_loss, correct, total = 0., 0, 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            self._record_data_flow(train_loader, batch_idx, batch_x)
            batch_x = batch_x.to(self.args.device)
            batch_y = batch_y.to(self.args.device)
            parts = split_features(batch_x, self.args)

            # Forward: bottom models
            embeddings = [self.bottoms[i](parts[i]) for i in range(len(self.bottoms))]
            agg = self._aggregate(embeddings)

            # Detach for split learning protocol
            agg_detached = agg.detach().requires_grad_(True)

            # Forward: top model
            opt_t.zero_grad()
            output = self.top_model(agg_detached)
            if self.ce_classes is not None:
                classes = self.ce_classes
                if (type(classes) not in (list, tuple) or not classes
                        or any(type(c) is not int or not 0 <= c < output.size(1)
                               for c in classes)
                        or len(set(classes)) != len(classes)):
                    raise ValueError('CE classes must be unique in-range integers')
                if (batch_y.ndim != 1 or batch_y.dtype != torch.long
                        or batch_y.numel() != output.size(0) or not batch_y.numel()):
                    raise ValueError('CE labels must be nonempty one-dimensional int64 targets')
                indices = torch.tensor(classes, dtype=torch.long, device=output.device)
                matches = batch_y[:, None] == indices
                if not matches.any(dim=1).all().item():
                    raise ValueError('CE labels must belong to the selected CE classes')
                loss_ce = F.cross_entropy(output.index_select(1, indices),
                                          matches.long().argmax(dim=1))
            elif self.ce_hi is not None:
                # CE only on the [ce_lo:ce_hi] logit slice with relabeled targets.
                # output stays connected to agg_detached, so the split-learning
                # grad_agg path that drives the bottom update is preserved.
                loss_ce = F.cross_entropy(output[:, self.ce_lo:self.ce_hi], batch_y - self.ce_lo)
            else:
                loss_ce = self.criterion(output, batch_y)

            # CL method combines CE + extra loss
            if extra_loss_fn is not None:
                loss = extra_loss_fn(self.bottoms, self.top_model, batch_x, batch_y, loss_ce)
            else:
                loss = loss_ce

            # Unlearning-aware CL: concentrate per-class ownership across parties so
            # that ANY future forget request has a small minimal owning set S*(f)
            # (-> cheaper, cleaner certified unlearning). Penalize the entropy of the
            # per-party score mass {|<W_c^(k), e_k>|}_k for each class in the batch.
            # The penalty flows into BOTH the bottoms (via embeddings) and the head
            # (via W) — exactly the parties the decomposition reads.
            ocw = getattr(self.args, 'own_concentrate_weight', 0.0)
            if ocw > 0:
                loss = loss + ocw * self._ownership_concentration_penalty(embeddings, batch_y)

            if grad_clip_norm is not None and not torch.isfinite(loss).item():
                raise FloatingPointError('non-finite clipped training loss')

            # retain_graph when the concentration penalty is on: it backprops through
            # `embeddings`, which the split-protocol `agg.backward(grad_agg)` below also
            # traverses; without retaining, that shared subgraph is freed first.
            loss.backward(retain_graph=(ocw > 0))
            if grad_clip_norm is None:
                opt_t.step()

            # Backward to bottom models via split protocol
            # Check if any bottom param requires grad (FIM may freeze all)
            any_trainable = any(p.requires_grad for b in self.bottoms for p in b.parameters())

            if any_trainable:
                grad_agg = agg_detached.grad
                if grad_agg is not None:
                    # Don't zero_grad: preserve EWC/regularization gradients
                    # from loss.backward(), then accumulate CE gradient
                    agg.backward(grad_agg)
                # Replacement losses may backprop directly through bottoms
                # without touching agg_detached. Apply those gradients too.
                if grad_clip_norm is not None:
                    parameters = [
                        parameter
                        for module in (*self.bottoms, self.top_model)
                        for parameter in module.parameters()
                        if parameter.grad is not None
                    ]
                    torch.nn.utils.clip_grad_norm_(
                        parameters, grad_clip_norm, error_if_nonfinite=True,
                    )
                    opt_t.step()
                for opt in opts_b: opt.step()
                for opt in opts_b: opt.zero_grad()  # clean up after step
            elif grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.top_model.parameters()
                     if parameter.grad is not None],
                    grad_clip_norm, error_if_nonfinite=True,
                )
                opt_t.step()

            total_loss += loss.item() * batch_y.size(0)
            correct += (output.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
            self.comm_rounds += 1
            self.bytes_transmitted += sum(e.nelement()*4*2 for e in embeddings)

        return total_loss/max(total,1), correct/max(total,1)

    def _record_data_flow(self, loader, batch_idx, batch_x):
        if not getattr(self.args, 'data_flow_audit', 0):
            return
        sampler = getattr(loader, 'audit_sampler', None)
        if sampler is None or not sampler.epoch_orders:
            raise RuntimeError('data-flow audit requires a recording deterministic sampler')
        if batch_x.device.type != 'cpu' or batch_x.dtype != torch.float32:
            raise RuntimeError('data-flow audit expects post-augmentation CPU float32 tensors')
        order = sampler.epoch_orders[-1]
        start = batch_idx * loader.batch_size
        record = {
            'loader_key': loader.audit_key,
            'loader_iteration': len(sampler.epoch_orders) - 1,
            'batch': batch_idx,
            'indices': order[start:start + batch_x.size(0)],
            'dtype': str(batch_x.dtype),
            'shape': list(batch_x.shape),
            'batch_sha256': tensor_sha256(batch_x),
        }
        path = os.path.join(self.args.output_dir, 'data_flow_audit.jsonl')
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')

    def train_task(self, train_loader, epochs, extra_loss_fn=None, lr=None,
                   grad_clip_norm=None):
        if (grad_clip_norm is not None
                and (type(grad_clip_norm) not in (int, float)
                     or isinstance(grad_clip_norm, bool)
                     or not math.isfinite(grad_clip_norm)
                     or grad_clip_norm <= 0)):
            raise ValueError(
                'gradient clip norm must be a positive finite number')
        optimizers = self._create_optimizers(lr)
        start = time.time()
        history = []
        for ep in range(epochs):
            loss, acc = self.train_epoch(
                train_loader, optimizers, extra_loss_fn, grad_clip_norm)
            history.append({'epoch':ep, 'loss':loss, 'acc':acc})
        return history, time.time()-start

    def _ownership_concentration_penalty(self, embeddings, batch_y):
        """Mean over in-batch classes of the entropy of the per-party ownership
        distribution p_{c,k} ∝ E_{x∈c}|<W_c^(k), e_k(x)>|. Minimizing entropy makes
        each class concentrate on few parties (small future S*). Differentiable in
        both the head W (classifier.weight) and the party embeddings e_k."""
        W = self.top_model.classifier.weight                # (C, D)
        P = len(embeddings)
        concat = self.args.aggregation == 'concat'
        d = W.size(1) // (P if concat else 1)
        ents = []
        for c in batch_y.unique():
            mask = (batch_y == c)
            if mask.sum() < 2:
                continue
            ci = int(c.item())
            scores = []
            for k in range(P):
                wk = W[ci, k*d:(k+1)*d] if concat else W[ci]
                scores.append((embeddings[k][mask] * wk).sum(1).abs().mean())
            s = torch.stack(scores)                          # (P,) per-party mass
            p = s / (s.sum() + 1e-8)
            ents.append(-(p * (p + 1e-8).log()).sum())       # entropy (minimize)
        if not ents:
            return torch.zeros((), device=W.device)
        return torch.stack(ents).mean()

    @torch.no_grad()
    def evaluate(self, test_loader):
        for b in self.bottoms: b.eval()
        self.top_model.eval()
        correct, total = 0, 0
        all_probs, all_labels = [], []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(self.args.device)
            batch_y = batch_y.to(self.args.device)
            parts = split_features(batch_x, self.args)
            embs = [self.bottoms[i](parts[i]) for i in range(len(self.bottoms))]
            agg = self._aggregate(embs)
            output = self.top_model(agg)
            correct += (output.argmax(1)==batch_y).sum().item()
            total += batch_y.size(0)
            all_probs.append(torch.softmax(output,1).cpu())
            all_labels.append(batch_y.cpu())
        acc = correct/max(total,1)
        return acc, torch.cat(all_probs) if all_probs else torch.tensor([]), torch.cat(all_labels) if all_labels else torch.tensor([])

    @torch.no_grad()
    def collect_logits(self, data_loader):
        for bottom in self.bottoms:
            bottom.eval()
        self.top_model.eval()
        rows, labels = [], []
        for batch_x, batch_y in data_loader:
            parts = split_features(batch_x.to(self.args.device), self.args)
            embeddings = [self.bottoms[i](parts[i]) for i in range(len(self.bottoms))]
            rows.append(self.top_model(self._aggregate(embeddings)).cpu())
            labels.append(batch_y.cpu())
        if not rows:
            return torch.empty((0, 0)), torch.empty((0,), dtype=torch.long)
        return torch.cat(rows), torch.cat(labels)

    @torch.no_grad()
    def compute_embeddings(self, data_loader):
        """Compute aggregated global embeddings for all samples."""
        for b in self.bottoms: b.eval()
        all_emb, all_labels = [], []
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.args.device)
            parts = split_features(batch_x, self.args)
            embs = [self.bottoms[i](parts[i]) for i in range(len(self.bottoms))]
            agg = self._aggregate(embs)
            all_emb.append(agg.cpu())
            all_labels.append(batch_y)
        return torch.cat(all_emb), torch.cat(all_labels)

    def get_state(self):
        return {
            'bottoms': [deepcopy(b.state_dict()) for b in self.bottoms],
            'top_model': deepcopy(self.top_model.state_dict()),
        }

    def load_state(self, state):
        for i, sd in enumerate(state['bottoms']):
            self.bottoms[i].load_state_dict(sd)
        self.top_model.load_state_dict(state['top_model'])

    def get_comm_stats(self):
        return {'comm_rounds':self.comm_rounds, 'megabytes_transmitted':round(self.bytes_transmitted/(1024*1024),2)}

    def reset_comm_stats(self):
        self.comm_rounds = 0
        self.bytes_transmitted = 0
