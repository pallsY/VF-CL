"""Restore one frozen checkpoint using only the pinned producer checkout."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main(producer_root, checkpoint_path, config_path, ranges_json):
    sys.path.insert(0, str(Path(producer_root).resolve()))
    from adaptive_consolidation_audit import _fresh_formal_state, _safe_torch_load
    from runner import (_capture_rng_state, _restore_rng_state,
                        _validate_resume_checkpoint_payload)

    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    args = SimpleNamespace(**config)
    args.device = 'cpu'
    ranges = json.loads(ranges_json)
    args.party_col_ranges = (None if ranges is None else
                             [tuple(pair) for pair in ranges])
    checkpoint = _safe_torch_load(checkpoint_path)
    _validate_resume_checkpoint_payload(checkpoint, args)
    live_rng = _capture_rng_state()
    try:
        trainer, method, tracker = _fresh_formal_state(checkpoint, args)
    finally:
        _restore_rng_state(live_rng)
    return {'strict_reloaded': True, 'method': type(method).__name__,
            'parties': len(trainer.bottoms),
            'tracker_rows': len(tracker.task_acc_matrix)}


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('usage: restore_worker PRODUCER_ROOT CHECKPOINT CONFIG RANGES_JSON')
    print('REPLAY11_RESTORE_JSON=' + json.dumps(main(*sys.argv[1:]),
                                           sort_keys=True))
