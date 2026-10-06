"""Conservative, owner-bound disk admission for the full public matrix."""
import fcntl
import os
from pathlib import Path
from urllib.parse import quote

import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
from three_dataset_seed42_reconcile import Evidence, pruning


GIB = 1024 ** 3
SAFETY_BYTES = 30 * GIB
RECEIPT = 'disk-reservation.json'
_RESULTS = '/home/c3080/YangXiaoXiang/VF-CL/results/'
_PILOT = _RESULTS + 'three_dataset_seed42_pilot_20260907_065121'
_FORMAL = _RESULTS + 'three_dataset_formal_20260906_021954'
_CLEANUP = _RESULTS + 'cleanup_evidence_20260911TwAWC7Q51/'

# Frozen storage-only observations, 2026-09-11. Historical roots are never read
# by admission. All three pinned authorities were inspected; these are the
# witnesses for the maxima. Peak = verified removed bytes + retained run bytes.
# GPM has no admitted record: its storage observation is bound to the approved
# predelete manifest/receipt, not scientific admission. Never infer accuracy.
OBSERVATIONS = [
    {'spec_key': 'cifar100:er:42', 'model_family': 'resnet18',
     'source_path': _PILOT + '/runs/cifar100%3Aer%3A42',
     'peak_bytes': 8761583073, 'retained_bytes': 573686343,
     'evidence_sha256': {
         _PILOT + '/records/cifar100%3Aer%3A42.json': 'cbfaaead4d88f32f76f624bbcf46db62e92165eeebeb1d55dae81b34158999d6',
         _PILOT + '/runs/cifar100%3Aer%3A42/PRUNE_PLAN.json': '7529168eaea676e5860eaedda4fddecd6ad0de38a1fb889de1fff35f1744e2b3',
         _PILOT + '/runs/cifar100%3Aer%3A42/PRUNED_EVIDENCE.json': 'cc05d2f56fc4dac2e21af578c0014c828eafc578c0f60810639c8f13b9310b8d'}},
    {'spec_key': 'isolet:er:42', 'model_family': 'mlp',
     'source_path': _PILOT + '/runs/isolet%3Aer%3A42',
     'peak_bytes': 242635025, 'retained_bytes': 16544177,
     'evidence_sha256': {
         _PILOT + '/records/isolet%3Aer%3A42.json': '88f06f71036013c5dcfa045c4137c6905b033650bcc67cd334cc5fa49b3d9636',
         _PILOT + '/runs/isolet%3Aer%3A42/PRUNE_PLAN.json': '38c64ab0d06bac41b3283f0bc897946678fd9bbf4e1346111e72057d018a9c45',
         _PILOT + '/runs/isolet%3Aer%3A42/PRUNED_EVIDENCE.json': 'cec4efca76ba51b2c35c8a1ec12eb66aa42415715eea83b057e71a3b2ed40cf7'}},
    {'spec_key': 'upmc_food101:er:42', 'model_family': 'mlp',
     'source_path': _PILOT + '/runs/upmc_food101%3Aer%3A42',
     'peak_bytes': 2491411245, 'retained_bytes': 199164899,
     'evidence_sha256': {
         _PILOT + '/records/upmc_food101%3Aer%3A42.json': '3f4d445df7457eae2e62579d75d7a014ca4d65572ab2356a0ba9a6b6ab4c6ff2',
         _PILOT + '/runs/upmc_food101%3Aer%3A42/PRUNE_PLAN.json': '7aeb27f0d1dbc371fc33de41110e04ee72e9752a21ff1c3d968dc9d0c8151e79',
         _PILOT + '/runs/upmc_food101%3Aer%3A42/PRUNED_EVIDENCE.json': '4fa381a35b7d8561e63954e71d8cfc40dc39344c9894c28dbaa002e54eaa2da5'}},
    *[{'spec_key': f'cifar100:gpm:{seed}', 'model_family': 'resnet18',
       'source_path': _FORMAL + f'/runs/cifar100%3Agpm%3A{seed}',
       'peak_bytes': peak, 'retained_bytes': retained,
       'evidence_sha256': {
           _CLEANUP + 'TARGETS_PREDELETE.tsv': '444d680e0baa87fdb2084a55cc3cf13d77d8a6528825b9e856e2ae22416cac24',
           _CLEANUP + 'CLEANUP_SUMMARY.txt': '4aa8c60d6f41f4e8e1e425256d925b238e793dc86b9a93fd754d88d64ad11a2a',
           _CLEANUP + 'DELETE_RECEIPT.txt': 'fb56f519efb52d37e62bb27861ceded9b6357601f35e3bb18fa2a76ca23d3f47'}}
      for seed, peak, retained in ((42, 16675212588, 969779196),
                                   (43, 16786183522, 966856178))],
]


def _size(value):
    if type(value) is not int or value < 0:
        raise ValueError('disk size must be a nonnegative exact integer')
    return value


def scoped_disk_status(root: Path, requested_slots: int = 1) -> dict:
    """Conservative capacity floor for a dataset-scoped matrix."""
    profile = registry.experiment_profile()
    if profile not in (registry.SINGLE_DATASET_PROFILE,
                       registry.CONTINUATION_PROFILE,
                       registry.METHOD_SHARD_PROFILE):
        raise ValueError('dataset-scoped profile is required')
    if type(requested_slots) is not int or requested_slots not in (1, 2):
        raise ValueError('single-dataset GPU count must be 1 or 2')
    if requested_slots != driver.pipeline_inflight_limit():
        raise ValueError('single-dataset GPU count differs from requested slots')
    root = driver._validate_formal_root_path(root, create=False)
    plan, _ = driver._load_installed_plan(root)
    dataset = registry.selected_formal_dataset()
    prefix = dataset + ':'
    if profile == registry.METHOD_SHARD_PROFILE:
        prefix += registry.selected_formal_method() + ':'
    observations = [row for row in OBSERVATIONS
                    if row['spec_key'].startswith(prefix)]
    if profile == registry.METHOD_SHARD_PROFILE and not observations:
        observations = [row for row in OBSERVATIONS
                        if row['spec_key'].startswith(dataset + ':')]
    if not observations:
        raise ValueError('dataset disk observation is missing')
    peak, retained = 0, 0
    for row in observations:
        spec = (registry.formal_specs()[0]
                if profile == registry.METHOD_SHARD_PROFILE
                else driver.spec_for_key(row['spec_key'], 'formal'))
        if (set(row) != {'spec_key', 'model_family', 'source_path',
                         'peak_bytes', 'retained_bytes', 'evidence_sha256'}
                or row['model_family'] != registry.protocol_for(spec)['base_options']['model_type']
                or Path(row['source_path']).name != (
                    quote(row['spec_key'], safe='')
                    if profile == registry.METHOD_SHARD_PROFILE
                    else registry.safe_spec_name(spec))):
            raise ValueError('dataset disk observation identity differs')
        observed_peak = _size(row['peak_bytes'])
        observed_retained = _size(row['retained_bytes'])
        if observed_peak < observed_retained or not observed_retained:
            raise ValueError('dataset disk observation size differs')
        peak = max(peak, observed_peak)
        retained = max(retained, observed_retained)
    peak = ((peak + GIB - 1) // GIB) * GIB
    specs = registry.formal_specs()
    if profile == registry.CONTINUATION_PROFILE:
        missing = set(plan['missing_jobs'])
        specs = [spec for spec in specs if driver.spec_key(spec) in missing]
    remaining = sum(driver._installed_completed_record(root, spec, plan) is None
                    for spec in specs)
    filesystem = os.statvfs(root)
    available = _size(filesystem.f_bavail) * _size(filesystem.f_frsize)
    active_peaks = (requested_slots + 1) * peak
    safety = 20 * GIB if dataset == 'cifar100' and requested_slots == 2 else SAFETY_BYTES
    reserved = active_peaks + remaining * retained + safety
    return {'kind': 'single_dataset_disk_status_v1', 'dataset': dataset,
            'remaining_jobs': remaining, 'available_bytes': available,
            'requested_slots': requested_slots,
            'active_peaks_bytes': active_peaks,
            'predicted_retained_remainder_bytes': remaining * retained,
            'safety_bytes': safety, 'required_bytes': reserved,
            'plan_sha256': driver._digest(plan), 'safe': available >= reserved}


def dataset_estimates():
    if registry.experiment_profile() != registry.FULL_MATRIX_PROFILE:
        raise ValueError('disk reservation requires full matrix profile')
    estimates = {}
    seen = set()
    for row in OBSERVATIONS:
        if type(row) is not dict or set(row) != {
                'spec_key', 'model_family', 'source_path', 'peak_bytes',
                'retained_bytes', 'evidence_sha256'}:
            raise ValueError('disk observation schema differs')
        spec = driver.spec_for_key(row['spec_key'], 'formal')
        path = driver._path(row['source_path'], 'disk observation source')
        if (row['model_family'] != registry.protocol_for(spec)['base_options']['model_type']
                or path.name != registry.safe_spec_name(spec) or path.parent.name != 'runs'
                or str(path) in seen):
            raise ValueError('disk observation identity differs')
        seen.add(str(path))
        if type(row['evidence_sha256']) is not dict or not row['evidence_sha256']:
            raise ValueError('disk observation evidence is missing')
        for name, digest in row['evidence_sha256'].items():
            driver._path(name, 'disk observation evidence')
            driver._hash_string(digest, 'disk observation evidence hash')
        peak, retained = _size(row['peak_bytes']), _size(row['retained_bytes'])
        if peak < retained or not retained:
            raise ValueError('disk observation sizes differ')
        previous = estimates.get(spec.dataset, (0, 0))
        estimates[spec.dataset] = (max(previous[0], peak), max(previous[1], retained))
    if set(estimates) != set(registry.DATASETS):
        raise ValueError('disk estimate unavailable')
    return {dataset: (((peak + GIB - 1) // GIB) * GIB, retained)
            for dataset, (peak, retained) in estimates.items()}


def reservation_receipt(owner, plan):
    driver._validate_owner(owner)
    peak, _ = dataset_estimates()[driver.spec_for_key(owner['job']).dataset]
    return {'kind': 'full_matrix_disk_reservation_v1', 'spec_key': owner['job'],
            'owner_sha256': driver._file_digest(owner), 'root_identity': owner['root_identity'],
            'source_commit': owner['source_commit'], 'plan_sha256': driver._digest(plan),
            'observations_sha256': driver._digest(OBSERVATIONS), 'reservation_bytes': peak}


def validate_receipt(receipt, owner, plan=None):
    # Receipt and started evidence bind the same immutable owner, not a PID alone.
    if type(receipt) is not dict:
        raise ValueError('disk reservation receipt is invalid')
    driver._hash_string(receipt.get('plan_sha256'), 'disk reservation plan hash')
    expected = reservation_receipt(owner, {} if plan is None else plan)
    if plan is None:
        expected['plan_sha256'] = receipt['plan_sha256']
    if not driver._exact_equal(receipt, expected):
        raise ValueError('disk reservation receipt identity differs')
    _size(receipt['reservation_bytes'])


def _settled(root, plan, key, queued, active):
    record = driver._installed_completed_record(root, driver.spec_for_key(key), plan)
    if record is None:
        return False
    # This is a pure evidence reader; it does not acquire or use GPU 0.
    driver._audit_handoff(root, plan, key, 0)
    run = 'runs/' + driver._claim_name(key)
    with Evidence(root) as evidence:
        resource = evidence.json(run + '/RESOURCE_EVIDENCE.json')
        if (resource.get('spec_key') != key or resource.get('plan_sha256') != driver._digest(plan)
                or any(not driver._exact_equal(resource.get(k), record[k]) for k in
                       ('claim_sha256', 'launch_sha256', 'command_sha256', 'resource'))
                or resource['artifact_sha256'].get('job.log') != record['log_sha256']):
            raise ValueError('disk settlement resource identity differs')
        _, proof = pruning(evidence, run, record, resource['artifact_sha256'])
        evidence.verify()
    return (proof is not None and key not in queued
            and (active is None or active['spec_key'] != key))


def _disk_status_locked(root, requested_slots, requesting_owner=None):
    """Caller holds the existing claims lock through admission and installation."""
    if type(requested_slots) is not int or requested_slots not in (1, 2):
        raise ValueError('requested slots must be 1 or 2')
    estimates = dataset_estimates()
    plan, _ = driver._load_installed_plan(root)
    identity, commit = driver._root_identity(root), driver._source_commit()
    queued, active = {}, None
    if os.path.lexists(root / 'audit_queue'):
        queue = driver._PinnedRoot(root / 'audit_queue')
        try:
            queued, active = driver._audit_state(root, plan, queue)
        finally:
            queue.close()
    specs = {driver.spec_key(spec): spec for spec in registry.formal_specs()}
    jobs = {registry.safe_spec_name(specs[key]): key for key in plan['missing_jobs']}
    claims = driver._PinnedRoot(root / 'claims')
    reserved, completed, workers = {}, set(), set()
    try:
        names = set(os.listdir(claims.fd))
        if not names <= set(jobs):
            raise ValueError('disk claims contain unknown or duplicate reservations')
        for name in sorted(names):
            key = jobs[name]
            owner, bundle_names, started = driver._read_claim_bundle(
                root / 'claims' / name, key, identity, commit)
            if RECEIPT not in bundle_names:
                raise ValueError('disk reservation receipt is missing')
            claim = driver._PinnedRoot(root / 'claims' / name)
            try:
                receipt = driver._read_json_from_pinned(claim, RECEIPT, canonical=True)
                validate_receipt(receipt, owner, plan)
            finally:
                claim.close()
            if _settled(root, plan, key, queued, active):
                completed.add(key)
                continue
            driver._validate_owner(owner, require_live=True)
            worker = driver._digest({k: v for k, v in owner.items() if k != 'job'})
            # A queued producer may start another job while its audited disk
            # reservation remains charged. An unhanded-off worker may not.
            if key not in queued:
                if worker in workers:
                    raise ValueError('duplicate worker disk reservation')
                workers.add(worker)
            reserved[key] = receipt['reservation_bytes']
        claims.verify()
    finally:
        claims.close()
    if requesting_owner is not None:
        worker = driver._digest({k: v for k, v in requesting_owner.items() if k != 'job'})
        if worker in workers:
            raise ValueError('duplicate worker disk reservation')
    remaining = []
    for key in plan['missing_jobs']:
        if key in completed or key in reserved:
            continue
        # An unclaimed completed record cannot bypass ownership accounting.
        if driver._installed_completed_record(root, specs[key], plan) is not None:
            raise ValueError('completed disk record has no reservation claim')
        remaining.append(estimates[specs[key].dataset])
    # ponytail: at most 126 cells; sorting chooses the conservative slots without
    # a scheduler abstraction. Retained bytes exclude slots already reserved.
    remaining.sort(reverse=True)
    requested = sum(peak for peak, _ in remaining[:requested_slots])
    retained = sum(size for _, size in remaining[requested_slots:])
    filesystem = os.statvfs(root)
    available = _size(filesystem.f_bavail) * _size(filesystem.f_frsize)
    active_bytes = sum(reserved.values())
    return {'kind': 'full_matrix_disk_status_v1', 'available_bytes': available,
            'safety_bytes': SAFETY_BYTES, 'active_reservation_bytes': active_bytes,
            'requested_reservation_bytes': requested,
            'predicted_retained_remainder_bytes': retained,
            'requested_slots': requested_slots,
            'safe': available - active_bytes - requested - retained >= SAFETY_BYTES,
            'plan_sha256': driver._digest(plan)}


def disk_status(root: Path, requested_slots: int) -> dict:
    root = driver._validate_formal_root_path(root, create=False)
    claims = driver._PinnedRoot(root / 'claims')
    try:
        fcntl.flock(claims.fd, fcntl.LOCK_EX)
        result = _disk_status_locked(root, requested_slots)
        claims.verify()
        return result
    finally:
        try:
            fcntl.flock(claims.fd, fcntl.LOCK_UN)
        finally:
            claims.close()
