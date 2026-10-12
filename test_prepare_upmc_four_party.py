import json
from pathlib import Path
import tempfile
import unittest
import hashlib

import numpy as np

from prepare_upmc_four_party import build_four_party_cache


class FourPartyCacheTest(unittest.TestCase):
    def test_preserves_samples_and_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / 'source'
            source_dir.mkdir()
            x = np.arange(3 * 1536, dtype=np.float32).reshape(3, 1536)
            y = np.array([0, 1, 0], dtype=np.int64)
            source_ids = np.array([10, 20, 30], dtype=np.int64)
            np.savez_compressed(
                source_dir / 'upmc_food101_vfl.npz',
                X=x, y=y, source_row_id=source_ids,
                train_idx=np.array([0, 1], dtype=np.int64),
                test_idx=np.array([2], dtype=np.int64),
                view_names=np.array(['image', 'text']),
                range_lo=np.array([0, 512], dtype=np.int64),
                range_hi=np.array([512, 1536], dtype=np.int64),
            )
            source_cache = source_dir / 'upmc_food101_vfl.npz'
            source_hash = hashlib.sha256(source_cache.read_bytes()).hexdigest()
            (source_dir / 'upmc_food101_vfl.metadata.json').write_text(
                json.dumps({'output_sha256': source_hash,
                            'selection_manifest_sha256': 'abc'}),
                encoding='utf-8',
            )

            output = build_four_party_cache(source_dir, root / 'four')
            with np.load(output, allow_pickle=False) as converted:
                for name, expected in (
                        ('X', x), ('y', y), ('source_row_id', source_ids),
                        ('train_idx', [0, 1]), ('test_idx', [2])):
                    np.testing.assert_array_equal(converted[name], expected)
                np.testing.assert_array_equal(converted['range_lo'], [0, 256, 512, 1024])
                np.testing.assert_array_equal(converted['range_hi'], [256, 512, 1024, 1536])
                self.assertEqual(converted['view_names'].tolist(), [
                    'image_a', 'image_b', 'text_a', 'text_b'])
            metadata = json.loads((output.parent / 'upmc_food101_vfl.metadata.json').read_text())
            self.assertEqual([view['hi'] - view['lo'] for view in metadata['views']],
                             [256, 256, 512, 512])
            self.assertEqual(metadata['source_selection_manifest_sha256'], 'abc')
