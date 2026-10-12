# CIFAR replay-factorial development relocation

The 4090 host still had an external GPU process after more than one hour. Its waiting launcher was cancelled before any CIFAR run directory, launch log, or exit code was created. No external process was signalled.

The eight training-only CIFAR arms are re-registered for `vf-cl-3080` GPU 0 with source commit `29de04603ae6bdcc0ede8c951eb18318c1a8a055`. The data files on both hosts have identical train/test/meta SHA-256. All arms keep seed 54/55, the same disjoint training holdout, sample capacity 20, and the selected Adaptive coefficients.

The initial 3080 v2 registration used `num_workers=0` from the 4090 config. The source deterministic protocol requires `num_workers=2`, so its first arm exited before training. It has no holdout readout or formal test access; the failed directory remains intact. The v3 configs change only `data_path`, `results_dir`, `output_dir`, and `num_workers` from the original 4090 configs. The v3 manifest SHA-256 is `4b1b6327c005dff5907bcb6b89d4db69db8015fc19e3e7ccd4338cdeaf9b4b6c`.

The 3080 system disk has limited free space, so run output is in `/dev/shm/vfcl-replay-factorial-cifar-20261012-v3` (24 GiB tmpfs). The first arm starts alone. Each completed run must pass exit/freeze/readout/no-formal-test/config/data/source checks, then be archived on local D: with archive SHA-256 and per-file verification before its scratch directory is removed. Keep the readout and audit summary on persistent storage. Launch the next arm only after the prior run is verified and enough scratch space is free. A tmpfs reset before verified backup invalidates that unfinished run; do not infer a result from partial files.

These are development holdout results, not formal test `AA_final`.