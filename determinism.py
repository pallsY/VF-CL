"""Small deterministic-experiment primitives shared by runner and loaders."""
import hashlib
import os
import random

import numpy as np
import torch


REQUIRED_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
}


def derive_seed(seed, *parts):
    payload = '|'.join(map(str, (seed,) + parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def configure_determinism(seed):
    expected = {**REQUIRED_ENV, 'PYTHONHASHSEED': str(seed)}
    mismatches = [
        f'{key}={os.environ.get(key)!r} (expected {value!r})'
        for key, value in expected.items()
        if os.environ.get(key) != value
    ]
    if mismatches:
        raise RuntimeError('deterministic environment contract failed: ' + '; '.join(mismatches))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_sha256(tensor):
    if tensor.device.type != 'cpu':
        raise ValueError('tensor_sha256 requires a CPU tensor')
    value = tensor.detach().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()
