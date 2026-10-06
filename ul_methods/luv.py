"""LUV unlearning: manifold mixup + GA + recovery, top model only."""
import torch, torch.nn as nn, time
from data_utils import split_features


class LUVUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'LUV'

    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        self.trainer.top_model.train()
        # Collect trainable params (respects FIM freeze)
        params = [p for b in self.trainer.bottoms for p in b.parameters() if p.requires_grad]
        params += list(self.trainer.top_model.parameters())
        # Only optimize top model
        opt = torch.optim.SGD(params,
                              lr=self.args.ul_lr, momentum=self.args.momentum)
        ce = nn.CrossEntropyLoss()
        start = time.time()
        history = []

        for ep in range(self.args.ul_epochs):
            # Phase 1: forget via manifold mixup + GA on forget data
            for bx, by in forget_train_loader:
                bx, by = bx.to(self.args.device), by.to(self.args.device)
                opt.zero_grad()
                parts = split_features(bx, self.args)
                embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                agg = self.trainer._aggregate(embs)

                n = agg.size(0)
                if n > 1:
                    perm = torch.randperm(n)
                    lam = torch.distributions.Beta(1.0, 1.0).sample((n, 1)).to(agg.device)
                    agg_mixed = lam * agg + (1 - lam) * agg[perm]
                else:
                    agg_mixed = agg

                loss = -ce(self.trainer.top_model(agg_mixed), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            # Phase 2: recover on retain data
            for bx, by in retain_train_loader:
                bx, by = bx.to(self.args.device), by.to(self.args.device)
                opt.zero_grad()
                parts = split_features(bx, self.args)
                embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                agg = self.trainer._aggregate(embs)
                loss = ce(self.trainer.top_model(agg), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            history.append({'epoch': ep})

        return {'history': history, 'time': time.time() - start, 'method': 'luv'}
