import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from adaptive_branch_diagnostics import diagnose_checkpoint
from head_consolidation import hash_top_state
from models import TopModel


class BranchDiagnosticsTests(unittest.TestCase):
    def test_existing_checkpoint_reports_four_heads_without_modification(self):
        pre = TopModel(4, 4)
        with torch.no_grad():
            pre.classifier.weight.copy_(8 * torch.eye(4))
            pre.classifier.bias.zero_()
        full = copy.deepcopy(pre)
        bias = copy.deepcopy(pre)
        with torch.no_grad():
            full.classifier.weight[[0, 1]] = full.classifier.weight[[1, 0]].clone()
            bias.classifier.weight[[0, 2]] = bias.classifier.weight[[2, 0]].clone()
        hashes = {
            'pre': hash_top_state(pre),
            'full': hash_top_state(full),
            'bias': hash_top_state(bias),
        }
        checkpoint = {
            'cl_state': {'adaptive_audit_bundle': {
                'pre_state': pre.state_dict(),
                'full_state': full.state_dict(),
                'bias_state': bias.state_dict(),
                'validation_embeddings': torch.eye(4),
                'validation_labels': torch.arange(4),
                'task_classes': {0: [0, 1], 1: [2, 3]},
                'result': {
                    'pre_head_sha256': hashes['pre'],
                    'candidate_hashes': hashes,
                    'ordered_classes': [0, 1, 2, 3],
                    'gate': {'g': 1.0},
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
        self.assertEqual(set(report['heads']), {'pre', 'full', 'bias', 'mixed'})
        self.assertEqual(report['heads']['pre']['task_id_accuracy'], 1.0)
        self.assertEqual(report['heads']['pre']['within_task_class_accuracy'], 1.0)
        self.assertEqual(report['heads']['full']['task_id_accuracy'], 1.0)
        self.assertEqual(report['heads']['full']['within_task_class_accuracy'], 0.5)
        self.assertEqual(report['heads']['bias']['task_id_accuracy'], 0.5)
        self.assertEqual(report['heads']['bias']['within_task_class_accuracy'], 1.0)
        self.assertEqual(report['heads']['mixed'], report['heads']['full'])


if __name__ == '__main__':
    unittest.main()
