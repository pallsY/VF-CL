import json
import tempfile
import unittest
from pathlib import Path

from frozen_ablation_run import gpu_eligible, verify_formal_seal


class AblationRunTests(unittest.TestCase):
    def test_gpu_admission_requires_correct_physical_card_and_headroom(self):
        good = {'index': 0,
                'uuid': 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
                'name': 'NVIDIA GeForce RTX 3080', 'free_mib': 3072}
        self.assertTrue(gpu_eligible(good, 0))
        self.assertFalse(gpu_eligible({**good, 'free_mib': 3071}, 0))
        self.assertFalse(gpu_eligible({**good, 'uuid': 'wrong'}, 0))

    def test_formal_seal_requires_single_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'formal_access').mkdir()
            (run / 'checkpoints').mkdir()
            (run / 'results.json').write_text('{}')
            (run / 'checkpoints/formal_final.pt').write_bytes(b'model')
            (run / 'FORMAL_EVALUATION_COMPLETE.json').write_text(
                json.dumps({'status': 'complete'}))
            (run / 'FORMAL_EVALUATION_SEALED.json').write_text(
                json.dumps({'status': 'sealed'}))
            (run / 'FORMAL_EVALUATION_PUBLISHED.json').write_text(
                json.dumps({'status': 'published'}))
            access = run / 'formal_access/test.consumed.json'
            access.write_text(json.dumps({'status': 'consumed',
                'access': {'split': 'test', 'event': 'first_iteration'}}))
            result = verify_formal_seal(run)
            self.assertEqual(len(result['results_sha256']), 64)
            access.unlink()
            with self.assertRaises(ValueError):
                verify_formal_seal(run)


if __name__ == '__main__':
    unittest.main()
