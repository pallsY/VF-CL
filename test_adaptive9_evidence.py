"""Integration gate for the proposed method's frozen final state."""

import json
import os
import unittest
from pathlib import Path

from replay11_evidence import (
    _verify_checkpoint_method_state, _verify_training_flow_strict)


class AdaptiveFinalStateTests(unittest.TestCase):
    def test_sample_only_training_batches_preserve_identity(self):
        run = Path(os.environ.get(
            'ADAPTIVE9_PILOT_RUN',
            '/brf/users/yangxx/VFCL-results/sampleonly-upmc_food101-'
            'preflight-20261017/upmc_food101/adaptive/42/run'))
        if not run.is_dir():
            self.skipTest('isolated Linux pilot is not mounted')
        config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        flow = _verify_training_flow_strict(
            run, config, '/brf/users/yangxx/VF-CL/data', 10)
        self.assertGreater(flow['batches'], 0)
        self.assertEqual(len(flow['party_col_ranges']), 4)

    def test_four_party_adaptive_checkpoint_strictly_restores(self):
        run = Path(os.environ.get(
            'ADAPTIVE9_PILOT_RUN',
            '/brf/users/yangxx/VFCL-results/sampleonly-upmc_food101-'
            'preflight-20261017/upmc_food101/adaptive/42/run'))
        if not run.is_dir():
            self.skipTest('isolated Linux pilot is not mounted')
        config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        frozen = json.loads((run / 'FORMAL_STATE_FROZEN.json').read_text(
            encoding='utf-8'))
        complete = json.loads((run / 'FORMAL_EVALUATION_COMPLETE.json').read_text(
            encoding='utf-8'))
        checked = _verify_checkpoint_method_state(
            run / 'checkpoints/formal_final.pt', config, 10,
            frozen_final=frozen['identity']['final_checkpoint'],
            evaluation=complete['evaluation'],
            party_col_ranges=[(0, 256), (256, 512),
                              (512, 1024), (1024, 1536)],
            producer_root='/tmp/vfcl-sampleonly-producer-preflight-v1')
        self.assertEqual(checked['method'], 'proto_evolve')
        self.assertTrue(checked['strict_reloaded'])


if __name__ == '__main__':
    unittest.main()
