# CIFAR replay-factorial development relocation

The 4090 host still had an external GPU process after more than one hour. Its waiting launcher was cancelled before any CIFAR run directory, launch log, or exit code was created. No external process was signalled.

The eight training-only CIFAR arms are re-registered for `vf-cl-3080` GPU 0 with source commit `29de04603ae6bdcc0ede8c951eb18318c1a8a055`. The data files on both hosts have identical train/test/meta SHA-256. The only fields changed from the locked 4090 configs are `data_path`, `results_dir`, and `output_dir`; the new manifest binds their exact bytes. All arms keep seed 54/55, the same disjoint training holdout, `num_workers=0`, sample capacity 20, and the selected Adaptive coefficients.

The 3080 system disk has limited free space, so run output is in `/dev/shm/vfcl-replay-factorial-cifar-20261012-v2` (24 GiB tmpfs). The first arm starts alone. Each completed run must pass exit/freeze/readout/no-formal-test/config/data/source checks, then be archived on local D: with archive SHA-256 and per-file verification before its scratch directory is removed. Keep the readout and audit summary on persistent storage. Launch the next arm only after the prior run is verified and enough scratch space is free. A tmpfs reset before verified backup invalidates that unfinished run; do not infer a result from partial files.

These are development holdout results, not formal test `AA_final`.