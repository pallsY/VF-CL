"""FineTune: no CL protection."""
class FineTuneCL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'FineTune'

    def before_task(self, task_id, new_classes, seen_classes):
        req = max(seen_classes)+1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)

    def train_task(self, train_loader, task_id):
        return self.trainer.train_task(train_loader, self.args.epochs_per_task)

    def after_task(self, train_loader, task_id):
        pass

    def get_state(self): return {}
    def load_state(self, s): pass
