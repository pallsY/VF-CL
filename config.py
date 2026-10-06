"""Experiment configuration."""
import argparse, os, json
from datetime import datetime

from adaptive_head_consolidation import (
    BIAS_BRANCH_CONFIG,
    FULL_BRANCH_CONFIG,
    SOLVER_MAX_ITERATIONS,
    SOLVER_TOLERANCE,
)


PARTY_KD_VARIANTS = {
    'base': {
        'dep_tracking_enabled': 0,
        'party_kd_enabled': 0,
        'party_kd_mode': 'uniform',
    },
    'uniform': {
        'dep_tracking_enabled': 1,
        'party_kd_enabled': 1,
        'party_kd_mode': 'uniform',
    },
    'static': {
        'dep_tracking_enabled': 1,
        'party_kd_enabled': 1,
        'party_kd_mode': 'static',
    },
    'task_marginal': {
        'dep_tracking_enabled': 1,
        'party_kd_enabled': 1,
        'party_kd_mode': 'task_marginal',
    },
    'inverse': {
        'dep_tracking_enabled': 1,
        'party_kd_enabled': 1,
        'party_kd_mode': 'inverse',
    },
    'shuffled': {
        'dep_tracking_enabled': 1,
        'party_kd_enabled': 1,
        'party_kd_mode': 'shuffled',
    },
}


def validate_resume_config(args, recorded):
    """Reject resume attempts that change the original training protocol."""
    runtime_keys = {'output_dir', 'resume_run_dir'}
    current = {
        key: value for key, value in vars(args).items()
        if key not in runtime_keys
    }
    saved = {
        key: value for key, value in recorded.items()
        if key not in runtime_keys
    }
    mismatches = {
        key: (saved.get(key, '<missing>'), current.get(key, '<missing>'))
        for key in sorted(set(saved) | set(current))
        if saved.get(key, '<missing>') != current.get(key, '<missing>')
    }
    if mismatches:
        raise ValueError(f'Resume config mismatch: {mismatches}')


def validate_party_kd_variant(config, expected):
    """Reject mislabeled experiment runs before training starts."""
    if not expected:
        return
    required = PARTY_KD_VARIANTS[expected]
    mismatches = [
        f"{key}={config.get(key)!r} (expected {value!r})"
        for key, value in required.items()
        if config.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"party KD variant contract {expected!r} failed: " + '; '.join(mismatches)
        )
    if expected == 'shuffled':
        if not config.get('party_weight_manifest'):
            raise ValueError("party KD variant contract 'shuffled' failed: manifest is required")
        if int(config.get('party_shuffle_seed', -1)) < 0:
            raise ValueError("party KD variant contract 'shuffled' failed: permutation seed is required")


def adaptive_validation_split_seed(args):
    if args.data == 'tinyimagenet':
        return 20260813
    vector_name = os.path.basename(args.vector_npz)
    if args.data == 'tabvfl' and vector_name in {
            'isolet_vfl.npz', 'upmc_food101_vfl.npz'}:
        return 20260809
    return 20260729


def validate_adaptive_head_consolidation(args):
    if (args.head_gate_rule != 'class_balanced'
            and os.environ.get('VFCL_REVIEWED_ADAPTIVE_ABLATION') != '1'):
        raise ValueError(
            'adaptive ablation rules require '
            'VFCL_REVIEWED_ADAPTIVE_ABLATION=1'
        )
    if args.head_consolidation_mode != 'adaptive_dual_branch':
        return
    tinyimagenet = args.data == 'tinyimagenet'
    required = {
        'head_consolidation_schedule': 'final',
        'head_consolidation_samples_per_class': 20,
        'head_consolidation_regularization': 0.01,
        'head_consolidation_class_regularization': 0.01,
        'head_consolidation_task_regularization': 0.01,
        'head_consolidation_task_weight': 1.3,
        'lambda_validation_enabled': 1,
        'lambda_validation_split_seed': adaptive_validation_split_seed(args),
        'head_full_lr': FULL_BRANCH_CONFIG['lr'],
        'head_full_steps': FULL_BRANCH_CONFIG['steps'],
        'head_bias_lr': BIAS_BRANCH_CONFIG['lr'],
        'head_bias_steps': BIAS_BRANCH_CONFIG['steps'],
        'head_gate_solver_tolerance': SOLVER_TOLERANCE,
        'head_gate_solver_max_iterations': SOLVER_MAX_ITERATIONS,
    }
    if tinyimagenet:
        required['lambda_validation_per_class'] = 50
    mismatches = [
        f"{key}={getattr(args, key)!r} (expected {value!r})"
        for key, value in required.items()
        if getattr(args, key) != value
    ]
    if mismatches:
        raise ValueError(
            "adaptive dual-branch contract failed: " + '; '.join(mismatches)
        )

def get_config():
    p = argparse.ArgumentParser(description="VFL-CLU Benchmark")
    # Dataset
    p.add_argument('--data', default='cifar100', choices=['cifar100','cifar10','tinyimagenet','mfeat','synthvfl','tabvfl'])
    p.add_argument('--data_path', default='./data')
    p.add_argument('--mfeat_path', default='', type=str,
                   help='Path to the prebuilt mfeat 6-view npz (default: '
                        '{data_path}/mfeat/mfeat_6view.npz). Built by prep_mfeat.py.')
    p.add_argument('--vector_npz', default='', type=str,
                   help='Explicit path to a vector-VFL npz (overrides per-dataset '
                        'default). Used by the synthvfl rho sweep to pick one npz '
                        'per specialization level. Built by prep_synth.py.')
    p.add_argument('--num_classes', default=100, type=int)
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--num_workers', default=2, type=int)
    p.add_argument('--deterministic', default=0, type=int,
                   help='Enable the strict reproducibility protocol (requires launcher env and 2 workers)')
    p.add_argument('--data_flow_audit', default=0, type=int,
                   help='Record sampler indices and post-augmentation CPU batch hashes')
    p.add_argument('--bic_enabled', default=0, type=int,
                   help='Enable formal output-only BiC calibration')
    p.add_argument('--bic_per_class', default=25, type=int,
                   help='Held-out CIFAR-100 training examples per class')
    p.add_argument('--bic_split_seed', default=20260722, type=int,
                   help='Method-independent calibration split seed')
    p.add_argument('--lambda_validation_enabled', default=0, type=int,
                   help='Evaluate hyperparameters only on a held-out training split')
    p.add_argument('--lambda_validation_per_class', default=25, type=int,
                   help='Held-out training examples per class for lambda selection')
    p.add_argument('--lambda_validation_split_seed', default=20260729, type=int,
                   help='Method-independent training-validation split seed')
    p.add_argument('--bic_lr', default=0.05, type=float)
    p.add_argument('--bic_steps', default=200, type=int)
    p.add_argument('--bic_fit_mode', default='sequential',
                   choices=['sequential', 'joint_final', 'joint_each_stage'],
                   help='Fit task-by-task, once at stream end, or jointly after each stage')
    # Task sequence
    p.add_argument('--num_tasks', default=5, type=int)
    p.add_argument('--custom_tasks', default='', type=str, help='e.g. 0,1,2|3,4,5|6,7|8,9')
    p.add_argument('--classes_per_task', default=20, type=int)
    p.add_argument('--unlearn_after_tasks', default='2,3', type=str)
    p.add_argument('--unlearn_classes', default='0,1;20', type=str)
    # VFL architecture
    p.add_argument('--num_parties', default=2, type=int)
    p.add_argument('--model_type', default='resnet18', choices=['resnet18','small_cnn','mlp'])
    p.add_argument('--cosine_head', action='store_true',
                   help='LUCIR-style cosine classifier head (removes class-norm bias)')
    p.add_argument('--aggregation', default='concat', choices=['concat','sum'])
    p.add_argument('--embed_dim', default=512, type=int)
    p.add_argument('--party_widths', default='', type=str,
                   help='Feature-quantity heterogeneity: per-party column widths summing to 32, '
                        'e.g. "20,12" for 2 unequal parties. Empty = homogeneous equal split.')
    # Training
    p.add_argument('--epochs_per_task', default=50, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--optimizer', default='sgd', choices=['sgd', 'adamw'],
                   help='Shared optimizer used by every party bottom and the top model')
    p.add_argument('--bottom_lr_scale', default=1.0, type=float,
                   help='Bottom learning rate multiplier relative to --lr')
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--weight_decay', default=5e-4, type=float)
    p.add_argument('--task_ce_mode', default='method',
                   choices=['method', 'current', 'seen', 'full'],
                   help='Shared CE logit scope: method policy, current-task classes, '
                        'all seen classes, or the fixed full classifier')
    p.add_argument('--device', default='cuda:0')
    # CL
    p.add_argument('--cl_method', default='proto_evolve',
                   choices=['finetune','proto_aug','proto_evolve','proto_fedspace','der_pp','er_ace','er','ewc','lwf','target','gpm','fedprotip_vfl','prl','afc','lwf_fim','lwf_wa','adagauss','proto_evolve_radapt'])
    p.add_argument('--formal_deferred_evaluation', type=int,
                   choices=(0, 1), default=0)
    # Redundancy-Adaptive (radapt) knobs ??used by proto_evolve_radapt CL +
    # radapt_router UL. easy_thresh = single-party-acc above which a class is
    # routed to LIGHT (LUV) instead of HEAVY (FedOSD).
    p.add_argument('--radapt_easy_thresh', default=0.50, type=float)
    p.add_argument('--radapt_frac_boost',  default=0.05, type=float,
                   help='Per easy-dominated class, extra FIM-freeze frac added to that party')
    p.add_argument('--radapt_frac_cap',    default=0.15, type=float,
                   help='Maximum total frac boost on any single party')
    p.add_argument('--proto_aug_weight', default=1.0, type=float)
    p.add_argument('--distill_weight', default=0.5, type=float)
    p.add_argument('--distill_weight_schedule', default='constant',
                   choices=['constant', 'stage_f_decay_010', 'stage_f_decay_005'],
                   help='Task-local schedule for proto_evolve logit distillation')
    p.add_argument('--proto_lambda_a', default=0.1, type=float,
                   help='proto_evolve: weight of prototype-replay head-protection loss_A')
    p.add_argument('--proto_replay_loss_norm',
                   default='legacy_class_normalized',
                   choices=['legacy_class_normalized', 'sample_mean'],
                   help='proto_evolve: reduction for balanced prototype replay; '
                        'sample_mean removes task-dependent loss scaling')
    p.add_argument('--proto_replay_ratio', default=1.0, type=float,
                   help='proto_evolve: synthetic prototype samples per current sample')
    p.add_argument('--head_consolidation_enabled', default=0, type=int,
                   help='proto_evolve: consolidate the classifier from stored prototypes at every task boundary')
    p.add_argument('--head_consolidation_mode', default='full_classifier',
                   choices=['full_classifier', 'task_class_bias', 'adaptive_dual_branch'],
                   help='Full classifier refit or lightweight task/class logit calibration')
    p.add_argument('--head_consolidation_regularization', default=0.01, type=float,
                   help='L2 anchor strength for task-boundary head consolidation')
    p.add_argument('--head_consolidation_lr', default=0.01, type=float)
    p.add_argument('--head_consolidation_steps', default=500, type=int)
    p.add_argument('--head_consolidation_class_regularization', default=0.01,
                   type=float,
                   help='Class-bias L2 strength for task/class calibration')
    p.add_argument('--head_consolidation_task_regularization', default=0.01,
                   type=float,
                   help='Task affine identity-anchor strength for task/class calibration')
    p.add_argument('--head_consolidation_task_weight', default=1.3, type=float,
                   help='Hierarchical task-evidence weight for task/class calibration')
    p.add_argument('--head_consolidation_samples_per_class', default=20, type=int,
                   help='Balanced raw replay examples per seen class for head consolidation')
    p.add_argument('--head_consolidation_schedule', default='final',
                   choices=['final', 'every'],
                   help='Apply head consolidation only after the stream or after every task')
    p.add_argument('--head_full_lr', default=FULL_BRANCH_CONFIG['lr'], type=float)
    p.add_argument('--head_full_steps', default=FULL_BRANCH_CONFIG['steps'], type=int)
    p.add_argument('--head_bias_lr', default=BIAS_BRANCH_CONFIG['lr'], type=float)
    p.add_argument('--head_bias_steps', default=BIAS_BRANCH_CONFIG['steps'], type=int)
    p.add_argument('--head_gate_rule', default='class_balanced',
                   choices=['class_balanced', 'fixed_half_ablation', 'sample_mean_ablation'])
    p.add_argument('--head_gate_solver_tolerance', default=SOLVER_TOLERANCE, type=float)
    p.add_argument('--head_gate_solver_max_iterations', default=SOLVER_MAX_ITERATIONS,
                   type=int)
    p.add_argument('--feat_distill_weight', default=0.0, type=float,
                   help='proto_evolve: PRL-style summed-squared-L2 feature-KD weight (~1 calibrated). 0=off')
    p.add_argument('--current_supcon_weight', default=0.0, type=float,
                   help='proto_evolve: supervised contrastive weight on current-task embeddings')
    p.add_argument('--current_supcon_temperature', default=0.1, type=float)
    p.add_argument('--current_supcon_start_task', default=5, type=int,
                   help='First task index that enables current-task supervised contrastive loss')
    p.add_argument('--dep_tracking_enabled', default=0, type=int,
                   help='If 1, track per-class per-party dependency statistics online during training')
    p.add_argument('--dep_tracking_momentum', default=0.9, type=float,
                   help='EMA momentum for online dependency tracking')
    p.add_argument('--party_kd_enabled', default=0, type=int,
                   help='If 1, add a party-aware auxiliary KD term on per-party logits')
    p.add_argument('--party_kd_mode', default='uniform',
                   choices=['uniform', 'static', 'task_marginal', 'inverse', 'shuffled'],
                   help='uniform, static, task-marginal, inverse, or manifest-based shuffled weights')
    p.add_argument('--party_kd_lambda', default=1.0, type=float,
                   help='Weight for the auxiliary party-wise KD term')
    p.add_argument('--expected_party_kd_variant', default='',
                   choices=['', 'base', 'uniform', 'static', 'task_marginal', 'inverse', 'shuffled'],
                   help='Fail fast unless the effective party-KD flags match this experiment variant')
    p.add_argument('--party_weight_manifest', default='', type=str,
                   help='Canonical static weight manifest used by shuffled party-KD')
    p.add_argument('--party_shuffle_seed', default=-1, type=int,
                   help='Frozen task-wise derangement seed for shuffled party-KD')
    p.add_argument('--party_proto_enabled', default=0, type=int,
                   help='If 1, reweight prototype replay by class-party dependency structure')
    p.add_argument('--party_proto_mode', default='uniform', choices=['uniform', 'static'],
                   help='uniform = no structural change, static = use frozen class_party_weights-derived class weights')
    p.add_argument('--own_concentrate_weight', default=0.0, type=float,
                   help='unlearning-aware CL: weight on the per-class ownership-entropy '
                        'penalty (concentrate each class on few parties -> smaller future '
                        '|S*| -> cheaper certified unlearning). 0=off. Composes with any cl_method.')
    p.add_argument('--proto_sdc', default=True, type=lambda s: str(s).lower() == 'true',
                   help='proto_evolve: enable Semantic Drift Compensation of stored prototypes (default true)')
    p.add_argument('--repr_loss_weight', default=0.1, type=float)
    p.add_argument('--pass_ssl_weight', default=1.0, type=float,
                   help='Weight for rotation self-supervision in PASS (0 = no SSL)')
    p.add_argument('--gpm_threshold', default=0.95, type=float,
                   help='Energy threshold for GPM SVD basis selection')
    p.add_argument('--fedprotip_tip_threshold', type=float, default=0.775)
    p.add_argument('--fedprotip_max_batches', type=int, default=20)
    p.add_argument('--prl_lambda_fkd', default=1.0, type=float,
                   help='PRL feature-distillation weight (scales mean per-sample squared-L2 anchor)')
    p.add_argument('--prl_lambda_proto', default=15.0, type=float,
                   help='PRL prototype-replay weight')
    p.add_argument('--prl_temp', default=1.0, type=float,
                   help='PRL softmax temperature for CE (0.1 saturated the cosine head)')
    p.add_argument('--prl_fixed_anchor', action='store_true',
                   help='Snapshot the distillation reference once (task 0 backbone) instead of '
                        're-snapshotting each task; avoids compounding chained-distillation drift')
    p.add_argument('--prl_freeze_after', default=-1, type=int,
                   help='Freeze bottoms (weights+BN) after this task index to make a fixed feature '
                        'extractor (e.g. 0 = freeze after task 0). -1 = never freeze')
    p.add_argument('--prl_sdc', action='store_true',
                   help='Semantic Drift Compensation: shift stored old-class prototypes by the '
                        'kernel-weighted drift of current-task features, so a trainable backbone '
                        'keeps protos valid (use with chained anchor, not prl_fixed_anchor)')
    p.add_argument('--afc_distill_weight', default=2.0, type=float,
                   help='AFC base feature-distillation weight (scaled by sqrt(n_seen/task_size))')
    p.add_argument('--lwf_alpha', default=0.5, type=float,
                   help='Weight for KD vs CE in LwF / LwF+FIM')
    p.add_argument('--lwf_temperature', default=2.0, type=float,
                   help='Softmax temperature for LwF KD')
    p.add_argument('--lwf_lambda', default=1.0, type=float,
                   help='LwF KD weight, PyCIL-style additive: loss = CE_new + lambda*KD_old '
                        '(replaces the old convex (1-alpha)*CE + alpha*KD)')
    p.add_argument('--lwf_ce_newonly', default=True, type=lambda s: str(s).lower() == 'true',
                   help='Canonical class-IL LwF: compute CE only on the NEW-class logit slice '
                        '(PyCIL/FACIL style) so old-class columns are not suppressed every batch. '
                        'False = legacy full-head CE (over-suppresses old classes -> exact-0 collapse).')
    p.add_argument('--fim_k0', default=15, type=int,
                   help='FIM freeze threshold base (mean - (k0 + alpha*log(t+2)) * std)')
    p.add_argument('--fim_alpha', default=3, type=float,
                   help='FIM freeze threshold log-scaling coeff (legacy mean-k*std formula)')
    p.add_argument('--fim_freeze_frac', default=0.25, type=float,
                   help='Fraction of most-FIM-important bottom param-tensors to freeze per task '
                        '(quantile-based; replaces the mis-calibrated mean-k*std threshold that froze 100%%)')
    # EWC baseline (Kirkpatrick et al., PNAS 2017 / online EWC, Schwarz et al. 2018)
    p.add_argument('--ewc_lambda', default=1000.0, type=float,
                   help='EWC penalty strength: loss += (lambda/2) * sum F (theta-theta*)^2. '
                        'Was hardcoded 1e6 (pinned bottoms, zero plasticity); ~100-2000 is the '
                        'stability/plasticity range with per-sample Fisher.')
    p.add_argument('--ewc_fisher_decay', default=0.9, type=float,
                   help='Online-EWC EMA across tasks: F <- decay*F + (1-decay)*F_new. '
                        'Keeps Fisher bounded (was unnormalized accumulation that grew each task).')
    p.add_argument('--ewc_fisher_samples', default=1024, type=int,
                   help='Max samples used for the per-sample diagonal-Fisher estimate per task '
                        '(<=0 = use the whole task loader).')
    # ER baseline
    p.add_argument('--er_per_class', default=300, type=int,
                   help='Reservoir buffer size per class (standard iCaRL/Mammoth CIFAR-10 budget; '
                        'was 20, which starved replay and let middle tasks collapse)')
    p.add_argument('--er_batch', default=64, type=int, help='Buffer mini-batch size at replay step')
    p.add_argument('--er_alpha', default=1.0, type=float,
                   help='Replay-loss up-weight: total = CE_new + alpha * CE_replay (alpha=1 is standard ER)')
    # ER-ACE baseline
    p.add_argument('--er_ace_buffer_size', default=argparse.SUPPRESS, type=int,
                   help='Flat reservoir capacity. 0 = auto (20*num_classes).')
    p.add_argument('--er_ace_batch', default=argparse.SUPPRESS, type=int,
                   help='Replay mini-batch size, capped by current buffer size.')
    # DER++ baseline (Buzzega et al., NeurIPS 2020) ??online reservoir
    p.add_argument('--der_buffer_size', default=0, type=int,
                   help='Reservoir buffer size, total. 0 = auto (20*num_classes). Like ER, a '
                        'small buffer (20/class) starves replay and lets old tasks collapse late.')
    p.add_argument('--der_batch', default=0, type=int,
                   help='Buffer mini-batch size per replay draw. 0 = use --batch_size.')
    p.add_argument('--der_alpha', default=0.5, type=float,
                   help='DER++ dark-experience MSE weight (logit replay)')
    p.add_argument('--der_beta', default=0.5, type=float,
                   help='DER++ CE replay weight (label replay)')
    # AdaGauss baseline
    p.add_argument('--adagauss_lambda_ac', default=0.2, type=float,
                   help='AdaGauss anti-collapse loss weight')
    p.add_argument('--adagauss_lambda_pkd', default=1.0, type=float,
                   help='AdaGauss projected feature distillation weight')
    p.add_argument('--adagauss_shrinkage', default=0.1, type=float,
                   help='AdaGauss covariance shrinkage alpha: Sigma <- (1-a)Sigma + a*I')
    p.add_argument('--adagauss_adapter_epochs', default=30, type=int,
                   help='Epochs to train per-task adapter MLP')
    p.add_argument('--adagauss_n_samples', default=256, type=int,
                   help='Samples per old class when adapting Gaussians through new adapter')
    # PROTOCOL: sanitize CL cached state (prototypes/teachers/anchors/replay
    # bans/buffers) at every UL event; 0 = ablation showing relapse.
    p.add_argument('--sanitize_cl_state', default=1, type=int)
    # UL
    p.add_argument('--ul_method', default='luv',
                   choices=['retrain','gradient_ascent','luv','mode','fucrt','fedup','fedosd','fedau','fudp','radapt_router','roar'])
    p.add_argument('--ul_epochs', default=10, type=int)
    # ROAR (Redundancy-Optimal Ablation & Recovery) ??ownership-bounded VFL unlearning
    p.add_argument('--roar_tau_own', default=0.80, type=float,
                   help='Cumulative ownership share that defines the minimal scrub set S*(f)')
    p.add_argument('--roar_scrub_epochs', default=3, type=int,
                   help='Bottom-scrub epochs on S* parties (gradient ascent + retain preservation)')
    p.add_argument('--roar_lambda_preserve', default=1.0, type=float,
                   help='Weight of retain-embedding MSE preservation during scrub')
    p.add_argument('--roar_recovery_epochs', default=2, type=int,
                   help='Top-only retain recovery epochs (server-side, no comm)')
    p.add_argument('--roar_scrub_lr', default=1e-4, type=float,
                   help='LR for the bottom scrub on S* parties')
    p.add_argument('--forget_class', default=6, type=int,
                   help='single forget class for attack_baseline.py (retrain-oracle AUC floor)')
    p.add_argument('--roar_scrub_mode', default='ascent', choices=['ascent', 'ortho', 'retrain_s'],
                   help="removal mechanism: 'ascent'=legacy GA+MSE+recovery; "
                        "'ortho'=orthogonalized UCE erasure on S*; "
                        "'retrain_s'=ROAR-v3 SISA x ownership (re-init+retrain S* encoders "
                        "on retain-only, freeze non-S*) ??exact removal at |S*| comm.")
    p.add_argument('--roar_retrain_epochs', default=30, type=int,
                   help="epochs for retrain_s mode (re-train of the S* encoders on retain)")
    p.add_argument('--roar_scrub_raw_weight', default=0.0, type=float,
                   help='Weight on direct raw-residual-score suppression in the '
                        'scrub (minimizes <W_f^k,e_k>^2, the head-reconnection '
                        'certified quantity). 0=legacy cosine-CE-only scrub.')
    p.add_argument('--ul_lr', default=1e-4, type=float)
    # MoDe (Zhao et al., IoT 2024)
    p.add_argument('--mode_lambda', default=0.95, type=float,
                   help='Momentum-degradation interpolation factor: W <- lam*W + (1-lam)*W_de')
    p.add_argument('--mode_warmup_rounds', default=3, type=int,
                   help='Pre-train degradation model on retain for this many epochs')
    p.add_argument('--mode_rounds', default=5, type=int,
                   help='Rounds of (degrade+memory-guidance)')
    p.add_argument('--mode_guidance_only_rounds', default=3, type=int,
                   help='Memory-guidance-only rounds after the degrade phase')
    p.add_argument('--mode_de_lr', default=1e-2, type=float,
                   help='LR for training the degradation model')
    p.add_argument('--mode_respect_fim', default=False, type=lambda s: str(s).lower() == 'true',
                   help='If true, restore FIM-frozen params after each momentum-degradation step')
    # FUCRT (Guo et al., ICCV 2025)
    p.add_argument('--fucrt_lambda_t', default=1.0, type=float, help='Transform MSE weight')
    p.add_argument('--fucrt_lambda_l', default=0.5, type=float, help='Logit-alignment KL weight')
    p.add_argument('--fucrt_lambda_r', default=1.0, type=float, help='Retain CE weight')
    # FedUP (Huang et al., TSC 2025)
    p.add_argument('--fedup_mu',     default=1.0, type=float, help='Forget-proto push-away weight')
    p.add_argument('--fedup_lambda', default=1.0, type=float, help='Retain-proto MSE weight in recovery')
    p.add_argument('--fedup_unlearn_epochs',  default=-1, type=int,
                   help='Phase-1 epochs (default: ul_epochs // 2)')
    p.add_argument('--fedup_recovery_epochs', default=-1, type=int,
                   help='Phase-2 epochs (default: ul_epochs - unlearn_epochs)')
    # FedOSD (Pan et al., AAAI 2025)
    p.add_argument('--fedosd_unlearn_rounds', default=5,  type=int)
    p.add_argument('--fedosd_post_rounds',    default=5,  type=int)
    p.add_argument('--fedosd_local_lr',       default=1e-2, type=float,
                   help='LR used inside each pseudo-gradient local-training pass')
    p.add_argument('--fedosd_local_epochs',   default=1,  type=int)
    p.add_argument('--fedosd_global_lr',      default=0.5, type=float,
                   help='Step size when applying the orthogonal descent direction')
    p.add_argument('--fedosd_max_step_norm',  default=5.0, type=float)
    # FedAU (Pan et al., 2022) ??linear-subtraction unlearning via auxiliary head
    p.add_argument('--fedau_mode', default='ul_class', choices=['ul_class', 'ul_samples'],
                   help='ul_class: W <- W - W_ul ; ul_samples: W <- alpha*W + (1-alpha)*W_ul')
    p.add_argument('--fedau_alpha', default=0.9, type=float,
                   help='Alpha for ul_samples mode (paper default 0.9)')
    p.add_argument('--fedau_aux_lr', default=1e-2, type=float,
                   help='LR for training the auxiliary head on forget data')
    p.add_argument('--fedau_aux_epochs', default=3, type=int)
    p.add_argument('--fedau_recovery_epochs', default=2, type=int)
    # FUDP (Wang et al., WWW 2022) ??class-discriminative channel pruning
    p.add_argument('--fudp_R', default=0.05, type=float,
                   help='Top fraction of channels to prune per layer')
    p.add_argument('--fudp_finetune_epochs', default=-1, type=int,
                   help='Epochs of retain fine-tune (-1 = use ul_epochs)')
    p.add_argument('--fudp_freeze_backbone', default=True, type=lambda s: str(s).lower() == 'true',
                   help='If true, freeze bottoms during recovery (safer for soft pruning)')
    # Experiment
    p.add_argument('--replay_mode', default='prototype', choices=['full','prototype'],
                   help='full = joint training with extras (not real CL); prototype = real CL (new task only)')
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--seeds', default='', type=str,
                   help='Comma-separated multi-seed list (e.g., "42,43,44"). Overrides --seed.')
    p.add_argument('--oracle_lr', default=0.1, type=float, help='LR for Oracle joint training (uses cosine annealing)')
    p.add_argument('--exp_name', default='')
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--save_task_checkpoints', default=0, type=int,
                   choices=[0, 1, 2, 3],
                   help='0=off, 1=every CIL event, 2=final only, 3=rolling resume + final')
    p.add_argument('--resume_run_dir', default='', type=str,
                   help='Existing incomplete output directory to resume')

    args = p.parse_args()
    explicit_er_ace_options = {
        name for name in ('er_ace_buffer_size', 'er_ace_batch')
        if hasattr(args, name)
    }
    args.er_ace_buffer_size = getattr(args, 'er_ace_buffer_size', 0)
    args.er_ace_batch = getattr(args, 'er_ace_batch', 64)
    args.formal_deferred_evaluation = bool(args.formal_deferred_evaluation)
    try:
        if args.cl_method != 'er_ace' and explicit_er_ace_options:
            option = '--' + sorted(explicit_er_ace_options)[0]
            raise ValueError(
                f'ER-ACE option requires --cl_method er_ace: {option}')
        if args.formal_deferred_evaluation and args.cl_method == 'er_ace':
            if args.er_ace_buffer_size < 0:
                raise ValueError(
                    'formal ER-ACE requires --er_ace_buffer_size >= 0')
            if args.er_ace_batch <= 0:
                raise ValueError('formal ER-ACE requires --er_ace_batch > 0')
        validate_party_kd_variant(vars(args), args.expected_party_kd_variant)
        validate_adaptive_head_consolidation(args)
    except ValueError as exc:
        p.error(str(exc))
    args.party_weight_manifest_hash = ''
    if args.party_kd_mode == 'shuffled':
        try:
            from party_weight_manifest import load_manifest
            manifest = load_manifest(
                args.party_weight_manifest,
                expected_seed=args.seed,
                expected_parties=args.num_parties,
                expected_classes=args.num_classes,
            )
            args.party_weight_manifest_hash = manifest['tensor_sha256']
        except (OSError, ValueError) as exc:
            p.error(str(exc))
    print(
        '[config] party_kd '
        f"expected={args.expected_party_kd_variant or 'unspecified'} "
        f"dep_tracking_enabled={args.dep_tracking_enabled} "
        f"party_kd_enabled={args.party_kd_enabled} "
        f"party_kd_mode={args.party_kd_mode}"
    )
    args.unlearn_after_tasks = [int(x) for x in args.unlearn_after_tasks.split(',')]
    args.unlearn_classes = [[int(c) for c in g.split(',')] for g in args.unlearn_classes.split(';')]

    # Image width per dataset (drives the vertical feature split for image data;
    # unused for vector/multi-view data, which splits by the column-range map).
    args.img_size = 64 if args.data == 'tinyimagenet' else 32

    # Vector/multi-view data (mfeat): no image-width feature split, so the
    # --party_widths (image-column heterogeneity) knob does not apply; the actual
    # per-party feature widths come from the dataset's view column ranges. The
    # party_col_ranges map is filled in by VFLDataset._init_vector at build time.
    from data_utils import VECTOR_DATASETS
    is_vector = args.data in VECTOR_DATASETS
    if is_vector:
        if args.party_widths.strip():
            print(f"[config] --party_widths is ignored for {args.data} (views fix "
                  "the per-party feature widths).")
        args.party_widths = None
        args.party_col_ranges = None
        if args.data == 'mfeat' and not args.mfeat_path:
            args.mfeat_path = os.path.join(args.data_path, 'mfeat', 'mfeat_6view.npz')
    # Feature-quantity heterogeneity: parse per-party widths (image data only)
    elif args.party_widths.strip():
        args.party_widths = [int(x) for x in args.party_widths.split(',') if x.strip()]
        assert len(args.party_widths) == args.num_parties, \
            f"party_widths ({args.party_widths}) must have num_parties={args.num_parties} entries"
        assert sum(args.party_widths) == args.img_size, \
            f"party_widths must sum to img_size={args.img_size}, got {sum(args.party_widths)}"
    else:
        args.party_widths = None

    # Auto-set for small_cnn
    if args.model_type == 'small_cnn':
        args.embed_dim = 128
        if args.aggregation == 'concat':
            args.aggregation = 'sum'  # V-LETO uses sum
    # MLP bottom (tabular/multi-view): modest 128-d embedding; keep the requested
    # aggregation (concat is what makes per-party top-weight blocks differ ->
    # ownership can concentrate, which is the whole point of the mfeat pivot).
    elif args.model_type == 'mlp':
        args.embed_dim = 128

    if args.exp_name == '':
        args.exp_name = f"{args.cl_method}_x_{args.ul_method}_{args.data}"
    if args.resume_run_dir:
        args.output_dir = os.path.abspath(args.resume_run_dir)
        if not os.path.isdir(args.output_dir):
            raise FileNotFoundError(f"Resume run directory does not exist: {args.output_dir}")
        config_path = os.path.join(args.output_dir, 'config.json')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Resume run has no config.json: {args.output_dir}")
        with open(config_path, encoding='utf-8') as handle:
            validate_resume_config(args, json.load(handle))
    elif args.formal_deferred_evaluation:
        if (os.path.basename(args.exp_name) != args.exp_name
                or args.exp_name in ('', '.', '..')):
            raise ValueError('Formal exp_name must be one exact safe child name')
        results_dir = os.path.abspath(args.results_dir)
        args.output_dir = os.path.abspath(
            os.path.join(results_dir, args.exp_name))
        if os.path.dirname(args.output_dir) != results_dir:
            raise ValueError('Formal output directory escapes results_dir')
        if os.path.lexists(args.output_dir):
            if os.path.islink(args.output_dir) \
                    or not os.path.isdir(args.output_dir):
                raise ValueError('Formal output directory is not a regular directory')
        else:
            os.makedirs(args.output_dir, exist_ok=False)
        with open(os.path.join(args.output_dir, 'config.json'), 'x') as f:
            json.dump(vars(args), f, indent=2)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(args.results_dir, f"{args.exp_name}_{ts}")
        if os.path.exists(args.output_dir):
            raise FileExistsError(f"Refusing to reuse existing output_dir: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    return args
