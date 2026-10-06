# Replay Marginal Utility: Seed-42 Mechanism Screen

Date: 2026-10-07. Status: prespecified exploratory negative screen. No PartyKD training change.

## Question and protocol

The previous party-weighted encoder cosine drift did not beat uniform or shuffled controls. This screen asks whether a stronger, task-specific party weight changes that result. At each pre-transition checkpoint, it measures how much removing one party's **logit contribution** raises old-class cross-entropy on the existing bounded training replay. Negative marginal utilities are clipped to zero and the positive values are normalized; an all-zero class falls back to uniform. This is fixed-model reliance, not the effect of retraining without that party.

For the same replay, the screen measures each party's squared class-logit change across the adjacent checkpoints. It compares four weighted sums: replay marginal utility, uniform, the checkpoint's frozen absolute-logit contribution weights, and a one-position cyclic shuffle of marginal-utility weights. The outcome is change in old-class-only cross-entropy on the frozen training-validation split. Ordinary class-incremental old-class error change is secondary. No validation labels enter the contribution weights or logit-change scores. The final transition is excluded because final head consolidation changes the classifier.

The design and fixed stop rule are in `docs/superpowers/specs/2026-10-07-party-utility-screen-design.md`. Final implementation commit: `7247a7b`; the isolated 4090 checkout is `909457e`. The analysis script SHA-256 is `b6e9a7005386fee0e1befe07b4e7b117f110542569f700fcfd5c3f53d631cd60` in both checkouts. Source runs and formal result roots were read-only; screen outputs were written to new roots. The source configurations enable `lambda_validation_enabled=1`; the script compares the entire rebuilt validation manifest with the original saved manifest before evaluation and calls `get_validation_loader` directly. It does not call a final-test loader or retrain either model.

## Evidence

| Dataset | Source v2 run | Config SHA-256 | Dataset SHA-256 | Validation manifest SHA-256 |
|---|---|---|---|---|
| ISOLET | `party-drift-isolet-dev-20261007-v2/isolet_drift_dev_numeric_fix_20261007_004217` | `7bce8a7295add50e1a072489d0b02cde11893940c73dea08e877899a718dc1ac` | `d34312670de93198afcd2b126c95b79bae2b4cffeb30d480f097faf046b69514` | `487e81663a12d4a663a1f421407aa3f88d788cc3c83f7323166cf7aff82902d3` |
| UPMC clean image | `party-drift-upmc-clean-dev-20261007-v2/upmc_drift_dev_numeric_fix_20261007_004255` | `6200f333208ebebc9cdf15e2c05422051624a09c87e0a5e08bc872923e6ecf3d` | `dea0569f0d4195eaaecc9cab3a6cce533f07b8d88472e7dbca5113152a314339` | `932c84ea194059068ce3baf07577d818fcc43bc865c0c8446bb2fa0e5104653d` |

The exact source-checkpoint SHA-256 values and each class-boundary aggregate are stored in the separate `screen.json` outputs. ISOLET output: `/home/chase/Yangxx/VF-CL/results/party-utility-isolet-20261007-v2/screen.json`, SHA-256 `92bec8358d3b914aef8598b48f97ee4ae2e2bc00f566b52703c630a791447572`. UPMC output: `/home/chase/Yangxx/VF-CL/results/party-utility-upmc-20261007-v2/screen.json`, SHA-256 `396f42203ca1efa385171095101f0b3727bcc0f0628d0c94f385818c10cd2b1c`. Separate repeat runs produced byte-identical output hashes for both datasets. The v2 class-boundary rows and correlations also match the preceding screen exactly. They contain aggregate class-level values, not individual examples, embeddings, or predictions.

| Dataset | Class-boundary rows | Transitions | Utility rho | Uniform rho | Frozen rho | Shuffled rho | Positive boundaries | Utility fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ISOLET | 132 | 11 | -0.462 | -0.512 | -0.555 | -0.426 | 0/10 eligible | 9.1% |
| UPMC clean image | 368 | 8 | -0.639 | -0.662 | -0.648 | -0.264 | 1/8 eligible | 7.1% |

Rho is pooled Spearman after centering the score and the old-class-only validation CE change within each task boundary. ISOLET's first boundary has only two old classes, so it is excluded from the directional boundary count. Mean absolute utility-score gaps versus uniform, frozen, and shuffled controls are respectively 0.0688, 0.0841, and 0.1364 for ISOLET, and 0.1868, 0.0975, and 0.3737 for UPMC. The methods genuinely produce different scores. Nevertheless, utility is negative on both datasets and fails the required positive association and 0.10 control margin.

The primary outcome itself decreased on average: mean old-class-only CE changes were -0.6845 (ISOLET) and -0.2490 (UPMC), with positive changes in 36.4% and 47.8% of class-boundary rows. Mean ordinary CIL old-class error changes were -0.1008 and -0.0316; the secondary centered Spearman correlations with the utility score were -0.334 and -0.569. These runs thus mostly show old-class improvement, which limits their ability to validate a forgetting-risk weight. Class-boundary rows are dependent, and these correlations are descriptive rather than significance tests.

## Verification and decision

On the Linux analysis host, `python -m unittest -q test_party_utility_screen test_party_drift_telemetry` passed 13 tests. The synthetic integration case checked that the screen leaves its source directory unchanged, excludes the final transition, and rejects a tampered source validation manifest. The local Windows suite passed 12 and skipped that integration case because the repository's existing manifest directory `fsync` requires POSIX. The real screens completed with finite scores, exact replay/class coverage, and recorded checkpoint/config/dataset hashes.

The prespecified rule failed on both datasets. Do not introduce replay-utility-weighted PartyKD or start a CIFAR-100 run from this evidence. A later attempt needs a protocol with measurable old-class degradation and an independent validation plan; selecting a new weighting formula from these already inspected rows would turn this screen into tuning rather than confirmation.
