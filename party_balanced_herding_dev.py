"""Temporary equal-party features for original-example herding studies."""

import math

import torch
import torch.nn.functional as F


def party_balanced_features(embeddings):
    if (len(embeddings) < 2 or any(
            not isinstance(part, torch.Tensor) or part.ndim != 2
            or not torch.isfinite(part).all() for part in embeddings)):
        raise ValueError('party embeddings must be finite matrices')
    count = embeddings[0].size(0)
    if count == 0 or any(part.size(0) != count for part in embeddings):
        raise ValueError('party embeddings must have matching nonzero rows')
    return torch.cat([
        F.normalize(part.float(), dim=1) / math.sqrt(len(embeddings))
        for part in embeddings
    ], dim=1)
