"""Maximally interfered retrieval over aligned VFL exemplar sets."""
import random

import torch
import torch.nn.functional as F
from torch.func import functional_call

from data_utils import split_features
from .er import ERCL


class MIRCL(ERCL):
    def __init__(self, trainer, args):
        super().__init__(trainer, args)
        self.name = 'MIR_VFL'
        self.candidates = int(getattr(args, 'mir_candidates', 128))
        if self.candidates <= self.buffer_batch_size:
            raise ValueError('MIR candidate pool must exceed replay batch size')

    def _forward_functional(self, x, parameters):
        pieces = split_features(x, self.args)
        embeddings = []
        for index, bottom in enumerate(self.trainer.bottoms):
            buffers = {name: value.detach().clone()
                       for name, value in bottom.named_buffers()}
            embeddings.append(functional_call(
                bottom, (parameters[index], buffers), (pieces[index],)))
        top = self.trainer.top_model
        buffers = {name: value.detach().clone()
                   for name, value in top.named_buffers()}
        logits = functional_call(
            top, (parameters[-1], buffers),
            (self.trainer._aggregate(embeddings),))
        self.trainer.bytes_transmitted += sum(
            embedding.numel() * embedding.element_size() * 2
            for embedding in embeddings)
        self.trainer.comm_rounds += 1
        return logits

    def _interference_scores(self, incoming_x, incoming_y,
                             candidate_x, candidate_y):
        """MIR score: candidate CE after one virtual current step minus before."""
        device = self.args.device
        incoming_x, incoming_y = incoming_x.to(device), incoming_y.to(device)
        candidate_x, candidate_y = candidate_x.to(device), candidate_y.to(device)
        modules = [*self.trainer.bottoms, self.trainer.top_model]
        current = [{name: parameter for name, parameter in module.named_parameters()}
                   for module in modules]
        loss = F.cross_entropy(
            self._forward_functional(incoming_x, current), incoming_y)
        named = [(index, name, parameter)
                 for index, params in enumerate(current)
                 for name, parameter in params.items() if parameter.requires_grad]
        gradients = torch.autograd.grad(
            loss, [parameter for _, _, parameter in named],
            allow_unused=True)
        virtual = [dict(params) for params in current]
        for (index, name, parameter), gradient in zip(named, gradients):
            if gradient is not None:
                rate = float(self.args.lr) * (
                    float(getattr(self.args, 'bottom_lr_scale', 1.0))
                    if index < len(self.trainer.bottoms) else 1.0)
                virtual[index][name] = (parameter - rate * gradient).detach()
        with torch.no_grad():
            before = F.cross_entropy(
                self._forward_functional(candidate_x, current), candidate_y,
                reduction='none')
            after = F.cross_entropy(
                self._forward_functional(candidate_x, virtual), candidate_y,
                reduction='none')
        return (after - before).detach()

    def _replay_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        all_examples = [(example, label)
                        for label, examples in self.buffer.data.items()
                        for example in examples]
        if not all_examples:
            return loss_ce
        picked = random.sample(all_examples,
                               min(self.candidates, len(all_examples)))
        candidates_x = torch.stack([example for example, _ in picked])
        candidates_y = torch.tensor([label for _, label in picked],
                                    dtype=torch.long)
        scores = self._interference_scores(
            batch_x, batch_y, candidates_x, candidates_y)
        count = min(self.buffer_batch_size, len(picked))
        indices = scores.topk(count).indices.cpu().tolist()
        replay_x = candidates_x[indices].to(self.args.device)
        replay_y = candidates_y[indices].to(self.args.device)
        pieces = split_features(replay_x, self.args)
        embeddings = [bottom(piece) for bottom, piece in zip(bottoms, pieces)]
        replay_loss = F.cross_entropy(
            top_model(self.trainer._aggregate(embeddings)), replay_y)
        self.trainer.bytes_transmitted += sum(
            embedding.numel() * embedding.element_size() * 2
            for embedding in embeddings)
        self.trainer.comm_rounds += 1
        fresh_count, replay_count = batch_y.numel(), replay_y.numel()
        return ((fresh_count * loss_ce + replay_count * replay_loss)
                / (fresh_count + replay_count))
