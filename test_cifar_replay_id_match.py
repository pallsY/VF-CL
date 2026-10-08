import unittest

import numpy as np
import torch

from cifar_replay_id_match import CifarReplayMatcher


MEAN = np.array([.507, .487, .441], dtype=np.float32)
STD = np.array([.267, .256, .276], dtype=np.float32)


def augmented(image, top, left, flip):
    padded = np.pad(image, ((4, 4), (4, 4), (0, 0)))
    crop = padded[top:top + 32, left:left + 32]
    if flip:
        crop = crop[:, ::-1]
    tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 255
    return (tensor - torch.from_numpy(MEAN)[:, None, None]) / torch.from_numpy(STD)[:, None, None]


class CifarReplayMatcherTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.images = rng.integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8)

    def test_recovers_exact_source_across_crop_and_flip(self):
        matcher = CifarReplayMatcher(self.images, [0, 1, 2])
        for top, left, flip in ((0, 8, False), (8, 0, True), (3, 5, True)):
            with self.subTest(top=top, left=left, flip=flip):
                self.assertEqual(matcher.recover_id(augmented(self.images[1], top, left, flip)), 1)

    def test_rejects_missing_source(self):
        matcher = CifarReplayMatcher(self.images, [0, 2])
        with self.assertRaisesRegex(ValueError, 'no exact source'):
            matcher.recover_id(augmented(self.images[1], 2, 6, False))

    def test_uses_canonical_id_for_identical_duplicate_source(self):
        self.images[2] = self.images[1]
        matcher = CifarReplayMatcher(self.images, [1, 2])
        self.assertEqual(matcher.recover_id(augmented(self.images[1], 4, 4, True)), 1)
        self.assertEqual(matcher.identical_duplicate_matches, 1)

    def test_rejects_distinct_sources_with_same_cropped_view(self):
        self.images[2] = self.images[1]
        self.images[2, 31, 31, 0] ^= 1
        matcher = CifarReplayMatcher(self.images, [1, 2])
        with self.assertRaisesRegex(ValueError, 'ambiguous distinct'):
            matcher.recover_id(augmented(self.images[1], 0, 0, False))


if __name__ == '__main__':
    unittest.main()
