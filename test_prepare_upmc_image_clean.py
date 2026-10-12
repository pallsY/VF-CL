"""Source-row alignment tests for the versioned UPMC feature cache."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prepare_upmc_image_clean import _existing_cache, load_selected_rows
from upmc_image_clean_split import build_selection_manifest


class PrepareUpmcImageCleanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'UPMC-Food-101'
        self._write_split('train', [
            ('a0.jpg', 'alpha', 0, b'A'),
            ('b0.jpg', 'beta', 0, b'B'),
            ('a1.jpg', 'alpha', 0, b'A'),
        ])
        self._write_split('test', [
            ('a2.jpg', 'other', 0, b'A'),
            ('c0.jpg', 'gamma', 1, b'C'),
        ])

    def _write_split(self, split, rows):
        image_dir = self.source / 'images' / split / 'food'
        image_dir.mkdir(parents=True)
        with (self.source / f'{split}.csv').open(
                'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=('id', 'text', 'annotation', 'label'))
            writer.writeheader()
            for name, text, label, image in rows:
                (image_dir / name).write_bytes(image)
                writer.writerow({
                    'id': name, 'text': text, 'annotation': 'food',
                    'label': label,
                })

    def test_preserves_original_row_id_and_csv_order(self):
        manifest = build_selection_manifest(self.source)
        train, test, source_ids = load_selected_rows(self.source, manifest)
        self.assertEqual(['a0.jpg', 'b0.jpg'], [row['id'] for row in train])
        self.assertEqual(['c0.jpg'], [row['id'] for row in test])
        self.assertEqual([0, 1, 4], source_ids.tolist())

    def test_rejects_csv_changed_after_manifest(self):
        manifest = build_selection_manifest(self.source)
        with (self.source / 'train.csv').open('a', encoding='utf-8') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'source CSV'):
            load_selected_rows(self.source, manifest)

    def test_existing_cache_symlink_is_rejected(self):
        other = Path(self.temp.name) / 'other.npz'
        other.write_bytes(b'not a UPMC cache')
        output = Path(self.temp.name) / 'upmc_food101_vfl.npz'
        output.symlink_to(other)
        metadata = Path(self.temp.name) / 'upmc_food101_vfl.metadata.json'
        metadata.write_text(json.dumps({
            'selection_manifest_sha256': 'a' * 64,
            'source_sha256': {},
            'output_sha256': hashlib.sha256(other.read_bytes()).hexdigest(),
        }), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            _existing_cache(output, metadata, {
                'manifest_sha256': 'a' * 64,
                'source_sha256': {},
            })


if __name__ == '__main__':
    unittest.main()
