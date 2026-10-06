"""Full retrain oracle."""
from models import build_models, TopModel
class RetrainUL:
    def __init__(self, trainer, args): self.trainer = trainer; self.args = args; self.name = 'Retrain'
    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        eff = kw.get('effective_classes', [])
        bottoms, top = build_models(self.args)
        self.trainer.bottoms = bottoms
        num_c = max(eff)+1 if eff else self.args.num_classes
        if self.args.aggregation == 'sum': top_dim = self.args.embed_dim
        else: top_dim = self.args.embed_dim * self.args.num_parties
        self.trainer.top_model = TopModel(top_dim, num_c,
                                          cosine=getattr(self.args, 'cosine_head', False)
                                          ).to(self.args.device)
        h, e = self.trainer.train_task(retain_train_loader, self.args.epochs_per_task)
        return {'history':h,'time':e,'method':'retrain'}
