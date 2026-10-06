"""GA unlearning: only operates on top model, respects FIM freeze."""
import torch, torch.nn as nn, time
from data_utils import split_features


class GradientAscentUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'GA'

    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        self.trainer.top_model.train()
        params = [p for b in self.trainer.bottoms for p in b.parameters() if p.requires_grad]
        params += list(self.trainer.top_model.parameters())
        # Only optimize top model (bottom frozen by FIM)
        opt = torch.optim.SGD(params,
                              lr=self.args.ul_lr, momentum=self.args.momentum)
        ce = nn.CrossEntropyLoss()
        start = time.time()
        history = []

        for ep in range(self.args.ul_epochs):
            tl, nb = 0, 0
            # Phase 1: gradient ascent on forget data (top model only)
            for bx, by in forget_train_loader:
                bx, by = bx.to(self.args.device), by.to(self.args.device)
                opt.zero_grad()
                parts = split_features(bx, self.args)
                with torch.no_grad():
                    embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                    agg = self.trainer._aggregate(embs)
                loss = -ce(self.trainer.top_model(agg), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainer.top_model.parameters(), 1.0)
                opt.step()
                tl += loss.item(); nb += 1

            # Phase 2: recovery on retain data (top model only)
            for bx, by in retain_train_loader:
                bx, by = bx.to(self.args.device), by.to(self.args.device)
                opt.zero_grad()
                parts = split_features(bx, self.args)
                with torch.no_grad():
                    embs = [self.trainer.bottoms[i](parts[i]) for i in range(len(self.trainer.bottoms))]
                    agg = self.trainer._aggregate(embs)
                loss = ce(self.trainer.top_model(agg), by)
                loss.backward()
                opt.step()

            history.append({'epoch': ep, 'loss': tl / max(nb, 1)})

        return {'history': history, 'time': time.time() - start, 'method': 'ga'}
