"""Refit only the frozen sample-only head with equal-party exemplars."""

import argparse
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from diagnose_sampleonly_head import validate_run_inputs
from reconstruct_exemplar_selection_dev import reconstruct_run
from run_resource_lite_pilot import PilotDataset, score_branch, sha256


@torch.no_grad()
def _embed_raw(raw_by_class, args, trainer):
    from data_utils import split_features

    for bottom in trainer.bottoms:
        bottom.eval()
    output = {}
    for class_id in sorted(raw_by_class):
        chunks = []
        raw = raw_by_class[class_id]
        for start in range(0, raw.size(0), args.batch_size):
            parts = split_features(raw[start:start + args.batch_size], args)
            embeddings = [bottom(part) for bottom, part in zip(
                trainer.bottoms, parts)]
            chunks.append(trainer._aggregate(embeddings).cpu())
        output[class_id] = torch.cat(chunks)
    return output


@torch.no_grad()
def _heldout_metrics(dataset, args, trainer, top, tasks):
    from data_utils import split_features

    top.eval()
    logits, labels = [], []
    for batch_x, batch_y in dataset.holdout_loader(
            list(range(args.num_classes))):
        parts = split_features(batch_x.to('cpu'), args)
        embeddings = [bottom(part) for bottom, part in zip(
            trainer.bottoms, parts)]
        logits.append(top(trainer._aggregate(embeddings)).cpu())
        labels.append(batch_y.long().cpu())
    return score_branch(F.log_softmax(torch.cat(logits), dim=1),
                        torch.cat(labels), tasks)


def evaluate_run(run):
    import runner
    from adaptive_head_consolidation import (
        adaptive_candidate_log_probabilities, fit_adaptive_candidates,
        hash_top_state, install_and_reload_verify,
        solve_global_mixture_weight,
    )
    from models import build_models
    from determinism import derive_seed
    from vfl_trainer import VFLTrainer

    run = Path(run)
    config, readout, _, final_path = validate_run_inputs(run)
    reconstruction = reconstruct_run(run)
    if not reconstruction['summary']['baseline_exact_reproduction']:
        raise ValueError('original task-boundary exemplars did not reproduce')
    payload = runner._decode_checkpoint_value(torch.load(
        final_path, map_location='cpu', weights_only=True))
    bundle = payload['cl_state']['adaptive_audit_bundle']
    with tempfile.TemporaryDirectory(prefix='vfcl-party-herding-head-') as tmp:
        args = SimpleNamespace(**config)
        args.output_dir = tmp
        args.device = 'cpu'
        dataset = PilotDataset(args)
        if sha256(Path(tmp) / 'pilot_holdout' / 'manifest.json') \
                != readout['holdout_manifest_sha256']:
            raise ValueError('reconstructed holdout manifest differs')
        bottoms, top = build_models(args)
        trainer = VFLTrainer(bottoms, top, args)
        trainer.load_state(payload['trainer_state'])
        for bottom in trainer.bottoms:
            bottom.eval()
        pre_top = build_models(args)[1].cpu().eval()
        pre_top.load_state_dict(bundle['pre_state'], strict=True)
        if hash_top_state(pre_top) != bundle['result']['pre_head_sha256']:
            raise ValueError('pre-head hash differs from frozen evidence')
        task_classes = payload['seen_task_classes']
        seed = derive_seed(args.seed, 'head_consolidation',
                           args.num_tasks - 1)
        original_embeddings = bundle['replay_embeddings']
        raw_count = sum(value.size(0) for value in
                        reconstruction['baseline_raw'].values())
        fit_pre_top = copy.deepcopy(pre_top).to(config['device']).eval()
        original = fit_adaptive_candidates(
            fit_pre_top, original_embeddings, {}, task_classes,
            raw_count, seed, config['device'])
        expected = bundle['result']['candidate_hashes']
        if (original.full_head_sha256 != expected['full']
                or original.bias_head_sha256 != expected['bias']):
            raise ValueError(
                'original final head candidates did not reproduce: '
                f'full={original.full_head_sha256} expected={expected["full"]}; '
                f'bias={original.bias_head_sha256} expected={expected["bias"]}'
            )
        val_x = bundle['validation_embeddings']
        val_y = bundle['validation_labels']
        full, bias = adaptive_candidate_log_probabilities(
            pre_top, original, val_x)
        original_gate = solve_global_mixture_weight(
            full, bias, val_y, original.ordered_classes)
        if abs(original_gate['g'] - bundle['result']['gate']['g']) > 1e-10:
            raise ValueError('original validation gate did not reproduce')
        original_top = install_and_reload_verify(
            pre_top, original, original_gate)
        baseline = _heldout_metrics(dataset, args, trainer, original_top,
                                    task_classes)
        if abs(baseline['cil'] - readout['metrics']['cil']) > 1e-12:
            raise ValueError('original head holdout score did not reproduce')
        selected_embeddings = _embed_raw(
            reconstruction['balanced_raw'], args, trainer)
        candidate = fit_adaptive_candidates(
            fit_pre_top, selected_embeddings, {}, task_classes,
            raw_count, seed, config['device'])
        full, bias = adaptive_candidate_log_probabilities(
            pre_top, candidate, val_x)
        gate = solve_global_mixture_weight(
            full, bias, val_y, candidate.ordered_classes)
        candidate_top = install_and_reload_verify(
            pre_top, candidate, gate)
        balanced = _heldout_metrics(dataset, args, trainer, candidate_top,
                                    task_classes)
    return {
        'status': 'training_holdout_equal_party_herding_control',
        'dataset': config['pilot_dataset'], 'seed': config['seed'],
        'source': str(run), 'final_checkpoint_sha256': sha256(final_path),
        'holdout_manifest_sha256': readout['holdout_manifest_sha256'],
        'exemplar_reconstruction': reconstruction['summary'],
        'raw_per_class': 20, 'baseline_gate': original_gate['g'],
        'balanced_gate': gate['g'],
        'baseline': baseline, 'balanced': balanced,
        'holdout_delta_cil': balanced['cil'] - baseline['cil'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    result = evaluate_run(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)
                           + '\n')
    print(json.dumps({
        'dataset': result['dataset'], 'seed': result['seed'],
        'baseline_cil': result['baseline']['cil'],
        'balanced_cil': result['balanced']['cil'],
        'delta_cil': result['holdout_delta_cil'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
