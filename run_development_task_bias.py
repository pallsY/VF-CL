"""Read-only generic task-bias control for sample-only development runs."""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from development_task_bias import apply_task_bias, fit_and_score
from diagnose_sampleonly_head import validate_run_inputs
from run_resource_lite_pilot import (
    PilotDataset, score_branch, sha256, validate_training_only_artifacts,
)


@torch.no_grad()
def frozen_logits(args, dataset, checkpoint):
    import runner
    from data_utils import split_features
    from models import build_models
    from vfl_trainer import VFLTrainer

    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.load_state(payload['trainer_state'])
    for model in (*trainer.bottoms, trainer.top_model):
        model.eval()
    bundle = payload['cl_state']['adaptive_audit_bundle']
    validation_x = bundle['validation_embeddings'].to('cpu')
    validation_y = bundle['validation_labels'].to(torch.long)
    validation_logits = trainer.top_model(validation_x).to(torch.float64)
    holdout_logits, holdout_labels = [], []
    for batch_x, batch_y in dataset.holdout_loader(
            list(range(args.num_classes))):
        parts = split_features(batch_x.to('cpu'), args)
        embedded = [bottom(part) for bottom, part in zip(trainer.bottoms,
                                                          parts)]
        holdout_logits.append(trainer.top_model(
            trainer._aggregate(embedded)).to(torch.float64).cpu())
        holdout_labels.append(batch_y.to(torch.long).cpu())
    return (validation_logits, validation_y,
            torch.cat(holdout_logits), torch.cat(holdout_labels),
            payload['seen_task_classes'])


def evaluate_run(run):
    run = Path(run)
    config, readout, _, checkpoint = validate_run_inputs(run)
    validate_training_only_artifacts(run)
    with tempfile.TemporaryDirectory(prefix='vfcl-task-bias-control-') as tmp:
        args = SimpleNamespace(**config)
        args.output_dir = tmp
        args.device = 'cpu'
        dataset = PilotDataset(args)
        if sha256(Path(tmp) / 'pilot_holdout' / 'manifest.json') \
                != readout['holdout_manifest_sha256']:
            raise ValueError('reconstructed holdout manifest differs')
        val_x, val_y, hold_x, hold_y, tasks = frozen_logits(
            args, dataset, checkpoint)
    baseline = score_branch(F.log_softmax(hold_x, dim=1), hold_y, tasks)
    if abs(baseline['cil'] - readout['metrics']['cil']) > 1e-12:
        raise ValueError('control baseline differs from frozen readout')
    fitted = fit_and_score(val_x, val_y, hold_x, hold_y, tasks)
    calibrated = score_branch(fitted['holdout_log_probs'], hold_y, tasks)
    before_val = float(F.cross_entropy(val_x, val_y))
    after_val = float(F.cross_entropy(
        apply_task_bias(val_x, fitted['offsets'], tasks), val_y))
    return {
        'status': 'training_holdout_generic_task_bias_control',
        'dataset': config['pilot_dataset'], 'seed': config['seed'],
        'source': str(run), 'checkpoint_sha256': sha256(checkpoint),
        'holdout_manifest_sha256': readout['holdout_manifest_sha256'],
        'validation_count': int(val_y.numel()),
        'validation_nll_before': before_val,
        'validation_nll_after': after_val,
        'task_offsets': fitted['offsets'].tolist(),
        'baseline': baseline, 'calibrated': calibrated,
        'holdout_delta_cil': calibrated['cil'] - baseline['cil'],
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
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'dataset': result['dataset'], 'seed': result['seed'],
        'baseline_cil': result['baseline']['cil'],
        'calibrated_cil': result['calibrated']['cil'],
        'delta_cil': result['holdout_delta_cil'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
