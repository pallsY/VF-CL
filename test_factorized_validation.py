import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from factorized_validation import diagnose_checkpoint
from models import TopModel


class FactorizedValidationTests(unittest.TestCase):
    def test_reloads_existing_adaptive_state_and_reads_validation_only(self):
        model = TopModel(4, 4)
        with torch.no_grad():
            model.classifier.weight.copy_(8 * torch.eye(4))
            model.classifier.bias.zero_()
        model.set_logit_calibration([0, 1, 2, 3], [1.0] * 4,
                                    [0.0] * 4, [0, 0, 1, 1], 1.0)
        full_weight = model.classifier.weight.detach().clone()
        full_weight[[0, 1]] = full_weight[[1, 0]].clone()
        model.set_adaptive_mixture(full_weight, torch.zeros(4), 1.0,
                                   [0, 1, 2, 3])
        state = copy.deepcopy(model.state_dict())
        checkpoint = {
            'trainer_state': {'top_model': copy.deepcopy(state)},
            'cl_state': {'adaptive_audit_bundle': {
                'installed_state': copy.deepcopy(state),
                'validation_embeddings': torch.eye(4),
                'validation_labels': torch.arange(4),
                'task_classes': {0: [0, 1], 1: [2, 3]},
                'result': {
                    'ordered_classes': [0, 1, 2, 3],
                    'validation_manifest': {'dataset': 'toy'},
                },
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save(checkpoint, path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            report = diagnose_checkpoint(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertEqual(report['dataset'], 'toy')
        self.assertEqual(report['split'], 'frozen_training_validation')
        self.assertEqual(report['count'], 4)
        self.assertEqual(report['mixed']['task_id_accuracy'], 1.0)
        self.assertEqual(report['mixed']['within_task_class_accuracy'], 0.5)
        self.assertEqual(report['factorized']['task_id_accuracy'], 1.0)
        self.assertEqual(report['factorized']['within_task_class_accuracy'], 1.0)
        self.assertEqual(report['factorized']['class_il_accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
