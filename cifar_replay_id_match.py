"""Recover CIFAR-100 training IDs from stored crop/flip replay tensors."""

import numpy as np
import torch


_MEAN = np.array([.507, .487, .441], dtype=np.float32)
_STD = np.array([.267, .256, .276], dtype=np.float32)


class CifarReplayMatcher:
    """Exact matcher for the repository's 4-pixel pad, crop, and flip view."""

    def __init__(self, cifar_data, candidate_ids):
        self.data = np.asarray(cifar_data)
        self.candidate_ids = [int(index) for index in candidate_ids]
        if (self.data.ndim != 4 or self.data.shape[1:] != (32, 32, 3)
                or self.data.dtype != np.uint8
                or not self.candidate_ids
                or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or min(self.candidate_ids) < 0
                or max(self.candidate_ids) >= len(self.data)):
            raise ValueError('malformed CIFAR candidate pool')
        self.identical_duplicate_matches = 0
        self.patch_index = {}
        for image_id in self.candidate_ids:
            image = self.data[image_id]
            for top in range(9):
                for left in range(9):
                    key = image[10 + top:14 + top, 10 + left:14 + left].tobytes()
                    self.patch_index.setdefault(key, []).append((image_id, top, left))

    @staticmethod
    def _pixels(augmented):
        if (not isinstance(augmented, torch.Tensor)
                or augmented.shape != (3, 32, 32)
                or not bool(torch.isfinite(augmented).all())):
            raise ValueError('malformed replay tensor')
        values = augmented.detach().cpu().permute(1, 2, 0).numpy().astype(np.float64)
        pixels = (values * _STD + _MEAN) * 255
        rounded = np.rint(pixels)
        if (np.any(rounded < 0) or np.any(rounded > 255)
                or np.max(np.abs(pixels - rounded)) > 1e-3):
            raise ValueError('replay tensor is not an exact CIFAR pixel view')
        return rounded.astype(np.uint8)

    def recover_id(self, augmented):
        observed = self._pixels(augmented)
        patch = observed[14:18, 14:18]
        matches = set()
        for flip, query in ((False, patch), (True, patch[:, ::-1])):
            for image_id, top, left in self.patch_index.get(query.tobytes(), ()):
                padded = np.pad(self.data[image_id], ((4, 4), (4, 4), (0, 0)))
                crop = padded[top:top + 32, left:left + 32]
                if flip:
                    crop = crop[:, ::-1]
                if np.array_equal(crop, observed):
                    matches.add(image_id)
        if not matches:
            raise ValueError('no exact source image for replay tensor')
        if len(matches) != 1:
            canonical = min(matches)
            if not all(np.array_equal(self.data[canonical], self.data[other])
                       for other in matches):
                raise ValueError('ambiguous distinct source images for replay tensor')
            self.identical_duplicate_matches += 1
            return canonical
        return next(iter(matches))
