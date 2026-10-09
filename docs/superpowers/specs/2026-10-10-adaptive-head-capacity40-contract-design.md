# Adaptive head capacity 40 production contract

Date: 2026-10-10. User-approved objective: make 40 raw replay examples/class a real versioned Adaptive method, keeping the existing 20/class method and its published artifacts unchanged. This task is code and audit integration; formal three-dataset accuracy runs and matched-budget baseline runs are separate follow-ups.

## Version and compatibility

Preserve the current 20/class Adaptive method as version 1 with its exact candidate-config schema, optimizer steps, gate solver, checkpoint fields and numerical path. Capacity 40 uses version 2. The v2 Full and Bias candidate configs add `samples_per_class: 40`; both fits consume 40/class. `TopModel` accepts v2 state for the mixed adaptive head while continuing to load v1. Checkpoint `adaptive_method_version`, top version, history result version, audit-bundle method version, run provenance source version and strict audit expectations must all agree with the configured capacity. The v2 audit verifies the retained raw and re-encoded replay each have exactly 40 examples for every retained class. Capacity other than 20 or 40 is rejected in production adaptive mode.

## Scientific and resource constraints

The three selected loss hyperparameters, 13 ISOLET tasks, class order, validation split, optimizer and 50 epochs/task remain unchanged. Forty raw examples/class are chosen at task time and re-encoded by the current encoder for the final head; no offline access to the complete old training corpus. No test loader is used while developing the code. Running 40/class changes persistent memory and head-fit compute, so new results require resource reporting and memory-matched replay baseline comparisons for a same-budget paper claim.

## Verification

Tests must prove v1 candidate configs/state hashes stay byte-identical and v1 checkpoint resume/audit remains valid; v2 must fit 40/class in both branches and produce/load/audit a v2 frozen checkpoint. Reject mismatched v1/v2 config, top state, replay count, and tampered audit. A small development ISOLET smoke run may validate 40/class end to end before any formal test evaluation.
