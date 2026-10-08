"""Fit three replay arms through the frozen Adaptive final-head procedure."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

import analyze_cifar_selection_view_pilot as selection_audit
import analyze_online_offline_herding_gap as feature_audit
import calibration_split
import cifar_replay_id_match
import cl_methods.proto_evolve as proto_evolve
import data_utils
import determinism
import launch_cifar_head_replay_intervention as launcher
import models
import vfl_trainer
from adaptive_consolidation_audit import _safe_torch_load
from adaptive_head_consolidation import (
    adaptive_candidate_log_probabilities, fit_adaptive_candidates,
    install_and_reload_verify, solve_global_mixture_weight,
)
from cifar_replay_id_match import CifarReplayMatcher
from cl_methods.proto_evolve import herding_indices
from data_utils import VFLDataset, split_features
from determinism import derive_seed
from head_consolidation import hash_top_state


SOURCE_COMMIT = launcher.SOURCE_COMMIT
SELECTION_AUDIT_SHA256 = '14ec4af1a53fed22171bc2333413a5075626c4c06e41db4d940337ee02805485'
FEATURE_AUDIT_SHA256 = 'c980e3347becc75984f91009aeccfe50fac8ffa87a022d43502690e194a6f5c3'
DATA_KEYS = ('data:cifar-100-python/train',
             'data:cifar-100-python/test',
             'data:cifar-100-python/meta')
EXPECTED_BIC_MANIFEST = '065ddbc79356de78717ebab88ca8761242609c6293402e4ce838242660165b88'
EXPECTED_LAMBDA_MANIFEST = '30ac89a61450e8693023dc0d3e71712e6be5081078c504c9b531db838753f427'


def readout_metrics(log_probabilities, labels):
    if (not isinstance(log_probabilities, torch.Tensor)
            or log_probabilities.ndim != 2
            or log_probabilities.shape[1] != 100
            or not isinstance(labels, torch.Tensor)
            or labels.ndim != 1
            or labels.shape[0] != log_probabilities.shape[0]
            or labels.numel() == 0):
        raise ValueError('readout shape must be [N, 100] with N labels')
    if (labels.dtype != torch.long
            or not bool(((labels >= 0) & (labels < 100)).all())):
        raise ValueError('readout label is outside CIFAR-100 classes')
    if not bool(torch.isfinite(log_probabilities).all()):
        raise ValueError('readout log probabilities must be finite')
    log_probabilities = log_probabilities.detach().cpu().double()
    labels = labels.detach().cpu()
    if not bool(torch.allclose(
            torch.logsumexp(log_probabilities, dim=1),
            torch.zeros(labels.numel(), dtype=torch.float64),
            atol=1e-6, rtol=0)):
        raise ValueError('readout rows are not normalized log probabilities')
    rows = torch.arange(labels.numel())
    cil = log_probabilities.argmax(dim=1).eq(labels)
    starts = (labels // 10) * 10
    task_columns = starts[:, None] + torch.arange(10)[None, :]
    til = (log_probabilities.gather(1, task_columns).argmax(dim=1)
           + starts).eq(labels)
    losses = -log_probabilities[rows, labels]

    def group(mask):
        total = int(mask.sum())
        correct = int((cil & mask).sum())
        return {
            'correct': correct,
            'total': total,
            'accuracy': correct / total if total else None,
        }

    per_class = {
        str(class_id): {
            **group(labels == class_id),
            'til_correct': int((til & (labels == class_id)).sum()),
            'nll_sum': float(losses[labels == class_id].sum()),
        }
        for class_id in range(100)
    }
    per_task = {
        str(task_id): {
            **group(labels // 10 == task_id),
            'til_correct': int((til & (labels // 10 == task_id)).sum()),
            'nll_sum': float(losses[labels // 10 == task_id].sum()),
        }
        for task_id in range(10)
    }
    til_correct = int(til.sum())
    return {
        'cil': group(torch.ones_like(labels, dtype=torch.bool)),
        'til': {
            'correct': til_correct,
            'total': labels.numel(),
            'accuracy': til_correct / labels.numel(),
        },
        'old_cil': group(labels < 90),
        'new_cil': group(labels >= 90),
        'nll': {
            'sum': float(losses.sum()),
            'total': labels.numel(),
            'mean': float(losses.mean()),
        },
        'per_class': per_class,
        'per_task': per_task,
    }


def _verify_inputs(run, source_config, source_record):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            != SOURCE_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], text=True)):
        raise ValueError('intervention requires exact clean producer checkout')
    if (launcher.file_sha256(source_record) != launcher.SOURCE_RECORD_SHA256
            or launcher.file_sha256(source_config) != launcher.SOURCE_CONFIG_SHA256
            or launcher.file_sha256(selection_audit.__file__)
            != SELECTION_AUDIT_SHA256
            or launcher.file_sha256(feature_audit.__file__)
            != FEATURE_AUDIT_SHA256):
        raise ValueError('formal source or selection helper differs from locked input')
    record = json.loads(source_record.read_text(encoding='utf-8'))
    if (record.get('kind') != 'formal_completed_run'
            or record.get('method') != 'adaptive'
            or record.get('seed') != 42
            or record.get('source_commit') != SOURCE_COMMIT):
        raise ValueError('formal source record identity mismatch')
    source_modules = {
        name: launcher.file_sha256(Path(module.__file__))
        for name, module in (
            ('adaptive_consolidation_audit.py', __import__('adaptive_consolidation_audit')),
            ('adaptive_head_consolidation.py', __import__('adaptive_head_consolidation')),
            ('calibration_split.py', calibration_split),
            ('cl_methods/proto_evolve.py', proto_evolve),
            ('data_utils.py', data_utils),
            ('determinism.py', determinism),
            ('head_consolidation.py', __import__('head_consolidation')),
            ('models.py', models),
            ('vfl_trainer.py', vfl_trainer),
        )
    }
    if any(digest != record['artifact_sha256']['source:' + name]
           for name, digest in source_modules.items()):
        raise ValueError('producer fitting/model module differs from audited source')
    protocol = json.loads((run / 'PILOT_PROTOCOL.json').read_text(encoding='utf-8'))
    complete = json.loads((run / 'PILOT_TRAINING_ONLY_COMPLETE.json').read_text(
        encoding='utf-8',
    ))
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    expected, overrides = launcher.derive_config(
        json.loads(source_config.read_text(encoding='utf-8')), run.parent,
    )
    if (config != expected or Path(config['output_dir']).resolve() != run
            or protocol['source_commit'] != SOURCE_COMMIT
            or protocol['overrides'] != overrides
            or protocol['source_record_sha256'] != launcher.SOURCE_RECORD_SHA256
            or protocol['source_config_sha256'] != launcher.SOURCE_CONFIG_SHA256
            or protocol['launcher_sha256'] != launcher.file_sha256(launcher.__file__)
            or protocol['planned_stop'] != 'before_deferred_test_evaluation'
            or complete['status'] != 'training_only_before_deferred_test_evaluation'
            or complete['source_commit'] != SOURCE_COMMIT
            or (run / 'results.json').exists()):
        raise ValueError('training-only protocol/completion mismatch')
    data_hashes = {
        key: launcher.file_sha256(Path(config['data_path']) / key.split(':', 1)[1])
        for key in DATA_KEYS
    }
    if (data_hashes != protocol['source_data_sha256']
            or any(digest != record['artifact_sha256'][key]
                   for key, digest in data_hashes.items())):
        raise ValueError('CIFAR data differ from audited source')
    artifacts = {
        'config_sha256': config_path,
        'checkpoint_sha256': run / 'adaptive_final.pt',
        'adaptive_freeze_sha256': run / 'ADAPTIVE_STATE_FROZEN.json',
        'bic_manifest_sha256': run / 'bic' / 'calibration_manifest.json',
        'validation_manifest_sha256': run / 'validation' / 'validation_manifest.json',
        'data_flow_audit_sha256': run / 'data_flow_audit.jsonl',
    }
    if any(launcher.file_sha256(path) != complete[key]
           for key, path in artifacts.items()):
        raise ValueError('training-only artifact hash mismatch')
    events = [run / 'checkpoints' / f'event_{task}_CIL.pt'
              for task in range(10)]
    if ([launcher.file_sha256(path) for path in events]
            != complete['event_checkpoint_sha256']):
        raise ValueError('task checkpoint hash mismatch')
    accesses = [json.loads(line) for line in (run / 'data_flow_audit.jsonl').read_text(
        encoding='utf-8',
    ).splitlines()]
    if any(entry.get('split') in {'calibration', 'test'}
           or str(entry.get('loader_key', '')).startswith((
               "('bic_calibration'", "('test'",
           )) for entry in accesses):
        raise ValueError('training accessed calibration or test loader')
    return config, complete, data_hashes, source_modules


def _make_dataset(config, run, scratch):
    args = SimpleNamespace(**config)
    args.output_dir = scratch
    args.num_workers = 0
    args.data_flow_audit = 0
    dataset = VFLDataset(args)
    if (dataset.calibration_manifest != json.loads((
            run / 'bic' / 'calibration_manifest.json').read_text(encoding='utf-8'))
            or dataset.validation_manifest != json.loads((
                run / 'validation' / 'validation_manifest.json').read_text(encoding='utf-8'))
            or dataset.calibration_manifest['sha256'] != EXPECTED_BIC_MANIFEST
            or dataset.validation_manifest['sha256'] != EXPECTED_LAMBDA_MANIFEST):
        raise ValueError('rebuilt fresh holdouts differ from locked training run')
    targets = dataset.validationset.targets
    bic = dataset.calibration_indices
    validation = dataset.validation_indices
    if (len(bic) != 2500 or len(validation) != 2500 or bic & validation
            or Counter(int(targets[index]) for index in bic)
            != {class_id: 25 for class_id in range(100)}
            or Counter(int(targets[index]) for index in validation)
            != {class_id: 25 for class_id in range(100)}):
        raise ValueError('holdout split is not disjoint and class balanced')
    eligible = [index for index in range(len(targets))
                if index not in bic and index not in validation]
    if (len(eligible) != 45000
            or Counter(int(targets[index]) for index in eligible)
            != {class_id: 450 for class_id in range(100)}):
        raise ValueError('eligible training pool is not 450/class')
    candidates = [
        [index for index in eligible if int(targets[index]) == class_id]
        for class_id in range(100)
    ]
    return args, dataset, candidates


def _hash_indices(indices):
    return hashlib.sha256(json.dumps(
        indices, separators=(',', ':'),
    ).encode()).hexdigest()


def _reconstruct_selections(run, checkpoint, args, dataset, candidates):
    saved = {int(class_id): value for class_id, value in
             checkpoint['cl_state']['head_raw_replay'].items()}
    if (set(saved) != set(range(100))
            or any(not isinstance(value, torch.Tensor)
                   or value.shape != (20, 3, 32, 32)
                   or not bool(torch.isfinite(value).all())
                   for value in saved.values())):
        raise ValueError('saved replay is not exactly 20 raw tensors/class')
    stochastic_ids, duplicate_matches = [], 0
    for class_id in range(100):
        matcher = CifarReplayMatcher(dataset.validationset.data,
                                     candidates[class_id])
        selected = [matcher.recover_id(raw) for raw in saved[class_id]]
        if len(selected) != 20:
            raise ValueError('stochastic replay ID count differs from budget')
        stochastic_ids.append(selected)
        duplicate_matches += matcher.identical_duplicate_matches

    deterministic_ids = [None] * 100
    event_bottom_hashes = []
    for task_id in range(10):
        event = _safe_torch_load(run / 'checkpoints' / f'event_{task_id}_CIL.pt')
        classes = list(range(task_id * 10, task_id * 10 + 10))
        if (event.get('schema_version') != 4
                or event.get('event_idx') != task_id
                or event.get('task_id') != task_id
                or event.get('step') != f'event_{task_id}_CIL'
                or event.get('new_classes') != classes):
            raise ValueError('task checkpoint identity mismatch')
        event_replay = {int(class_id): value for class_id, value in
                        event['cl_state']['head_raw_replay'].items()}
        if (set(event_replay) != set(range((task_id + 1) * 10))
                or any(not torch.equal(event_replay[class_id], saved[class_id])
                       for class_id in classes)):
            raise ValueError('task-boundary replay differs from final memory')
        task_indices = [index for class_id in classes
                        for index in candidates[class_id]]
        aggregate, _parties, labels, bottom_hash = selection_audit._embed_state(
            event['trainer_state'], dataset, task_indices, args,
        )
        event_bottom_hashes.append(bottom_hash)
        if any(not bool((labels[local * 450:(local + 1) * 450]
                         == class_id).all())
               for local, class_id in enumerate(classes)):
            raise ValueError('task candidate feature labels are misaligned')
        for local, class_id in enumerate(classes):
            full = aggregate[local * 450:(local + 1) * 450]
            chosen = herding_indices(full, 20)
            if chosen.numel() != 20 or chosen.unique().numel() != 20:
                raise ValueError('deterministic task-time selection is malformed')
            deterministic_ids[class_id] = [
                candidates[class_id][int(index)] for index in chosen
            ]
        del event
    flat_s = [index for class_ids in stochastic_ids for index in class_ids]
    flat_d = [index for class_ids in deterministic_ids for index in class_ids]
    if (len(flat_s) != 2000 or len(flat_d) != 2000
            or len(set(flat_d)) != 2000):
        raise ValueError('replay selections violate 20/class budget')
    return saved, stochastic_ids, deterministic_ids, duplicate_matches, event_bottom_hashes


def _deterministic_raw(dataset, selected_ids):
    output = {}
    for class_id, indices in enumerate(selected_ids):
        rows = [dataset.validationset[index] for index in indices]
        if (len(rows) != 20
                or any(int(label) != class_id for _raw, label in rows)):
            raise ValueError('deterministic replay class/count mismatch')
        output[class_id] = torch.stack([raw for raw, _label in rows])
    return output


@torch.no_grad()
def _embed_replay_raw(bottoms, raw_by_class, args):
    if set(raw_by_class) != set(range(100)):
        raise ValueError('replay classes are incomplete')
    for bottom in bottoms:
        bottom.eval()
    before = feature_audit.bottom_state_sha256(bottoms)
    embeddings = {}
    for class_id in range(100):
        raw = raw_by_class[class_id]
        if raw.shape != (20, 3, 32, 32):
            raise ValueError('replay raw tensor count/shape mismatch')
        parts = split_features(raw.to(args.device), args)
        party = [bottom(part) for bottom, part in zip(bottoms, parts)]
        aggregate = sum(party) if args.aggregation == 'sum' else torch.cat(
            party, dim=1,
        )
        if not bool(torch.isfinite(aggregate).all()):
            raise ValueError('non-finite replay embedding')
        embeddings[class_id] = aggregate.detach().cpu()
    if feature_audit.bottom_state_sha256(bottoms) != before:
        raise RuntimeError('bottom state changed while embedding replay')
    return embeddings


def _fit_arm(pre_top, replay_embeddings, prototypes, task_classes,
             validation_embeddings, validation_labels, args):
    before = hash_top_state(pre_top)
    fitted = fit_adaptive_candidates(
        pre_top, replay_embeddings, prototypes, task_classes,
        2000, derive_seed(args.seed, 'head_consolidation', 9), args.device,
    )
    canonical_pre = copy.deepcopy(pre_top).cpu().eval()
    full_probs, bias_probs = adaptive_candidate_log_probabilities(
        canonical_pre, fitted, validation_embeddings,
    )
    gate = solve_global_mixture_weight(
        full_probs, bias_probs, validation_labels, fitted.ordered_classes,
    )
    installed = install_and_reload_verify(canonical_pre, fitted, gate)
    if hash_top_state(pre_top) != before:
        raise RuntimeError('pre-consolidation top changed between arms')
    return installed, {
        'pre_sha256': fitted.pre_head_sha256,
        'full_sha256': fitted.full_head_sha256,
        'bias_sha256': fitted.bias_head_sha256,
        'installed_sha256': hash_top_state(installed),
        'gate': gate,
    }


def _embed_bic_holdout(dataset, bottoms, args):
    indices = [index for class_id in range(100)
               for index in sorted(dataset.calibration_indices)
               if int(dataset.calibrationset.targets[index]) == class_id]
    if len(indices) != 2500:
        raise ValueError('BiC readout cohort is not 25/class')
    before = feature_audit.bottom_state_sha256(bottoms)
    embeddings, _parties, labels = feature_audit.embed_batches(
        bottoms,
        DataLoader(Subset(dataset.calibrationset, indices),
                   batch_size=64, shuffle=False, num_workers=0),
        args,
    )
    if (feature_audit.bottom_state_sha256(bottoms) != before
            or any(not bool((labels[class_id * 25:(class_id + 1) * 25]
                             == class_id).all()) for class_id in range(100))):
        raise RuntimeError('BiC embeddings changed bottoms or label order')
    return embeddings, labels


def _analyze(run, config, device):
    if (device != 'cuda:0' or os.environ.get('CUDA_VISIBLE_DEVICES') != '1'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError('intervention requires masked physical GPU 1 as cuda:0')
    determinism.configure_determinism(int(config['seed']))
    checkpoint = _safe_torch_load(run / 'adaptive_final.pt')
    if (checkpoint.get('schema_version') != 1
            or checkpoint.get('kind') != 'adaptive_final_checkpoint'
            or checkpoint['provenance']['source_commit'] != SOURCE_COMMIT):
        raise ValueError('unexpected final Adaptive checkpoint')
    cl_state = checkpoint['cl_state']
    bundle = cl_state['adaptive_audit_bundle']
    if (not isinstance(bundle, dict)
            or bundle['result']['task_id'] != 9
            or bundle['result']['validation_manifest']['sha256']
            != EXPECTED_LAMBDA_MANIFEST
            or bundle['event_order'] != [
                'candidates_frozen', 'validation_iterated', 'gate_solved',
                'state_installed', 'diagnostics_computed',
            ]):
        raise ValueError('final adaptive audit bundle identity mismatch')
    with tempfile.TemporaryDirectory(prefix='vfcl-head-replay-') as scratch:
        args, dataset, candidates = _make_dataset(config, run, scratch)
        args.device = device
        saved, stochastic_ids, deterministic_ids, duplicates, event_hashes = (
            _reconstruct_selections(run, checkpoint, args, dataset, candidates)
        )
        if (set(bundle['replay_raw']) != set(range(100))
                or any(not torch.equal(bundle['replay_raw'][class_id],
                                       saved[class_id])
                       for class_id in range(100))):
            raise ValueError('frozen replay audit differs from retained memory')
        bottoms, installed_top = models.build_models(args)
        trainer = vfl_trainer.VFLTrainer(bottoms, installed_top, args)
        trainer.load_state(checkpoint['trainer_state'])
        for bottom in trainer.bottoms:
            bottom.eval()
            for parameter in bottom.parameters():
                parameter.requires_grad_(False)
        bottom_hash = feature_audit.bottom_state_sha256(trainer.bottoms)
        if bottom_hash != event_hashes[-1]:
            raise ValueError('final bottoms differ from task-9 boundary')

        pre_top = models.TopModel(
            trainer.top_model.classifier.in_features, 100,
            cosine=bool(args.cosine_head),
        ).to(args.device)
        pre_top.load_state_dict(bundle['pre_state'], strict=True)
        pre_top.eval()
        if (hash_top_state(pre_top)
                != bundle['result']['pre_head_sha256']
                or hash_top_state(trainer.top_model)
                != hash_top_state(bundle['installed_state'])):
            raise ValueError('pre/installed top differs from frozen audit')
        replay_a = _embed_replay_raw(trainer.bottoms, saved, args)
        if (set(bundle['replay_embeddings']) != set(range(100))
                or any(not torch.equal(replay_a[class_id],
                                       bundle['replay_embeddings'][class_id])
                       for class_id in range(100))):
            raise ValueError('re-embedded original raw replay differs from audit')
        raw_b = _deterministic_raw(dataset, stochastic_ids)
        raw_c = _deterministic_raw(dataset, deterministic_ids)
        replay_b = _embed_replay_raw(trainer.bottoms, raw_b, args)
        replay_c = _embed_replay_raw(trainer.bottoms, raw_c, args)
        if feature_audit.bottom_state_sha256(trainer.bottoms) != bottom_hash:
            raise RuntimeError('replay interventions changed bottom state')

        validation_embeddings = bundle['validation_embeddings']
        validation_labels = bundle['validation_labels']
        if (validation_embeddings.shape[0] != 2500
                or validation_labels.shape != (2500,)
                or Counter(int(label) for label in validation_labels)
                != {class_id: 25 for class_id in range(100)}):
            raise ValueError('frozen lambda-validation cache is malformed')
        prototypes = cl_state['global_protos']
        task_classes = bundle['task_classes']
        if (set(prototypes) != set(range(100))
                or task_classes != {
                    task_id: list(range(task_id * 10, task_id * 10 + 10))
                    for task_id in range(10)
                }):
            raise ValueError('final prototypes/task classes are incomplete')
        replay_sets = {
            'A_original_stochastic': bundle['replay_embeddings'],
            'B_same_ids_deterministic_storage': replay_b,
            'C_deterministic_selection_storage': replay_c,
        }
        heads, arm_evidence = {}, {}
        for name, replay in replay_sets.items():
            heads[name], arm_evidence[name] = _fit_arm(
                pre_top, replay, prototypes, task_classes,
                validation_embeddings, validation_labels, args,
            )
        original = arm_evidence['A_original_stochastic']
        expected_hashes = bundle['result']['candidate_hashes']
        if (original['pre_sha256'] != expected_hashes['pre']
                or original['full_sha256'] != expected_hashes['full']
                or original['bias_sha256'] != expected_hashes['bias']
                or original['gate'] != bundle['result']['gate']
                or original['installed_sha256']
                != hash_top_state(bundle['installed_state'])):
            raise RuntimeError('A failed to reproduce frozen production head')
        pre_hash = hash_top_state(pre_top)
        if (feature_audit.bottom_state_sha256(trainer.bottoms) != bottom_hash
                or pre_hash != bundle['result']['pre_head_sha256']):
            raise RuntimeError('fitting altered shared bottom or pre-head state')

        # All fitting and gate selection end before the BiC readout is accessed.
        bic_embeddings, bic_labels = _embed_bic_holdout(
            dataset, trainer.bottoms, args,
        )
        arm_metrics = {}
        for name, head in heads.items():
            before = hash_top_state(head)
            head.eval()
            with torch.no_grad():
                log_probabilities = head(bic_embeddings)
            arm_metrics[name] = readout_metrics(log_probabilities, bic_labels)
            if hash_top_state(head) != before:
                raise RuntimeError('BiC readout changed a fitted head')
        if (feature_audit.bottom_state_sha256(trainer.bottoms) != bottom_hash
                or getattr(dataset, '_first_accessed_splits', set())):
            raise RuntimeError('readout changed bottoms or used a protected loader')

    flat_s = [index for class_ids in stochastic_ids for index in class_ids]
    flat_d = [index for class_ids in deterministic_ids for index in class_ids]
    if len(flat_s) != 2000 or len(flat_d) != 2000:
        raise ValueError('intervention memory count differs from 20/class')
    return {
        'bottom_state_sha256': bottom_hash,
        'pre_head_sha256': pre_hash,
        'event_bottom_state_sha256': event_hashes,
        'eligible_indices_sha256': _hash_indices(
            [index for class_ids in candidates for index in class_ids]),
        'stochastic_ids_sha256': _hash_indices(flat_s),
        'deterministic_ids_sha256': _hash_indices(flat_d),
        'identical_duplicate_matches': duplicates,
        'bic_indices_sha256': _hash_indices(
            [index for class_id in range(100)
             for index in sorted(dataset.calibration_indices)
             if int(dataset.calibrationset.targets[index]) == class_id]),
        'bic_count': 2500,
        'arms': {
            name: {
                'head': arm_evidence[name],
                'readout': arm_metrics[name],
            }
            for name in replay_sets
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--source-config', required=True, type=Path)
    parser.add_argument('--source-record', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', required=True, choices=('cuda:0',))
    cli = parser.parse_args()
    run = cli.run.resolve(strict=True)
    source_config = cli.source_config.resolve(strict=True)
    source_record = cli.source_record.resolve(strict=True)
    output = cli.output_dir.resolve()
    if output.exists() or output == run or run in output.parents:
        raise ValueError('intervention output must be a new root outside training run')
    config, complete, data_hashes, modules = _verify_inputs(
        run, source_config, source_record,
    )
    result = _analyze(run, config, cli.device)
    payload = {
        'schema_version': 1,
        'status': 'training_only_head_replay_intervention',
        'seed': 50,
        'run': str(run),
        'source_commit': SOURCE_COMMIT,
        'source_record_sha256': launcher.SOURCE_RECORD_SHA256,
        'source_config_sha256': launcher.SOURCE_CONFIG_SHA256,
        'pilot_protocol_sha256': launcher.file_sha256(run / 'PILOT_PROTOCOL.json'),
        'pilot_completion_sha256': launcher.file_sha256(
            run / 'PILOT_TRAINING_ONLY_COMPLETE.json'),
        'checkpoint_sha256': complete['checkpoint_sha256'],
        'bic_manifest_sha256': complete['bic_manifest_sha256'],
        'validation_manifest_sha256': complete['validation_manifest_sha256'],
        'data_sha256': data_hashes,
        'source_modules_sha256': modules,
        'script_sha256': launcher.file_sha256(__file__),
        'matcher_sha256': launcher.file_sha256(cifar_replay_id_match.__file__),
        'selection_audit_sha256': SELECTION_AUDIT_SHA256,
        'feature_audit_sha256': FEATURE_AUDIT_SHA256,
        **result,
    }
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / 'head_replay_intervention.json.tmp'
    with open(temporary, 'x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / 'head_replay_intervention.json')
    print(json.dumps({
        name: {
            'gate': value['head']['gate']['g'],
            'cil': value['readout']['cil']['accuracy'],
            'til': value['readout']['til']['accuracy'],
            'old_cil': value['readout']['old_cil']['accuracy'],
            'new_cil': value['readout']['new_cil']['accuracy'],
            'nll': value['readout']['nll']['mean'],
        }
        for name, value in result['arms'].items()
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
