"""Tests for the immutable image-level UPMC selection rule."""
import csv
import tempfile
import unittest
from pathlib import Path

from upmc_image_clean_split import build_selection_manifest, select_rows


class UpmcImageCleanSplitTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'UPMC-Food-101'
        self._write_split('train', [
            ('a0.jpg', 'apple', 'food', 0, b'A'),
            ('b0.jpg', 'bad', 'food', 0, b'B'),
            ('a1.jpg', 'apple', 'food', 0, b'A'),
            ('c0.jpg', 'caption one', 'food', 0, b'C'),
            ('c1.jpg', 'caption two', 'food', 0, b'C'),
            ('c2.jpg', 'caption one', 'food', 0, b'C'),
        ])
        self._write_split('test', [
            ('a2.jpg', 'other', 'food', 0, b'A'),
            ('b1.jpg', 'bad', 'food', 1, b'B'),
            ('d0.jpg', 'novel', 'food', 1, b'D'),
            ('d1.jpg', 'novel', 'food', 1, b'D'),
        ])

    def _write_split(self, split, rows):
        folder = self.source / 'images' / split / 'food'
        folder.mkdir(parents=True)
        with (self.source / f'{split}.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=('id', 'text', 'annotation', 'label'))
            writer.writeheader()
            for name, text, annotation, label, image in rows:
                (folder / name).write_bytes(image)
                writer.writerow({
                    'id': name, 'text': text,
                    'annotation': annotation, 'label': label,
                })

    def test_conflicts_train_priority_and_exact_duplicates(self):
        manifest = build_selection_manifest(self.source)
        self.assertEqual('upmc_image_clean_v1', manifest['rules_version'])
        self.assertEqual([0, 3, 4], select_rows(manifest, 'train'))
        self.assertEqual([2], select_rows(manifest, 'test'))
        self.assertEqual(3, manifest['summary']['train_min_per_class'])
        self.assertEqual(1, manifest['summary']['test_min_per_class'])
        reasons = {
            (row['split'], row['source_index']): row['reason']
            for row in manifest['rows']
        }
        self.assertEqual('label_conflict', reasons['train', 1])
        self.assertEqual('label_conflict', reasons['test', 1])
        self.assertEqual('train_priority', reasons['test', 0])
        self.assertEqual('exact_duplicate', reasons['train', 2])
        self.assertEqual('exact_duplicate', reasons['train', 5])
        self.assertEqual('exact_duplicate', reasons['test', 3])
        self.assertEqual(manifest, build_selection_manifest(self.source))

    def test_missing_image_fails_closed(self):
        (self.source / 'images' / 'test' / 'food' / 'd0.jpg').unlink()
        with self.assertRaises(FileNotFoundError):
            build_selection_manifest(self.source)

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            select_rows({'rows': []}, 'validation')


if __name__ == '__main__':
    unittest.main()
