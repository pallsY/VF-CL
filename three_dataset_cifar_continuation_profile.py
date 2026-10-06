"""Project the verified CIFAR origin bundle into a 42-cell census."""

from dataclasses import asdict

import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
from three_dataset_cifar_continuation_reconcile import (
    origin_keys, validate_bundle,
)


def reuse_census(bundle):
    validate_bundle(bundle)
    if registry.experiment_profile() != registry.CONTINUATION_PROFILE:
        raise ValueError('CIFAR continuation profile is required')
    census = driver.build_census({})
    admitted = {row['spec_key']: row for row in bundle['admitted']}
    if set(admitted) != set(origin_keys()):
        raise ValueError('CIFAR continuation reuse membership differs')
    for index, spec in enumerate(registry.formal_specs()):
        key = driver.spec_key(spec)
        if key not in admitted:
            continue
        row = admitted[key]
        raw = {
            'status': 'REUSABLE', 'reason': 'admitted',
            'spec': asdict(spec),
            'protocol_sha256': driver._protocol_sha256(spec),
            'source_sha256': driver._digest(row['source_sha256']),
            'artifact_sha256': row['artifact_sha256'],
            'metrics': row['metrics'],
            'metric_formula_version': driver.FORMULA_VERSION,
            'trajectory_sha256': row['trajectory_sha256'],
        }
        census['records'][index] = {
            **raw, 'spec_key': key,
            'admission_record_sha256': driver._digest(raw),
        }
    driver._validate_census(census)
    return census
