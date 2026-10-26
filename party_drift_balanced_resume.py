"""Scheduler-only, append-only resume for the frozen party-drift matrix."""

import hashlib
from pathlib import Path

from party_drift_audit_gate import verify_sealed_job
from party_drift_manifest import REPLAY_ARMS


def validate_history(manifest_rows, events, result_root, manifest_sha256):
    """Classify sealed, reconstructable, and unstarted jobs without scores."""
    root = Path(result_root)
    rows = {row['job_key']: row for row in manifest_rows}
    if len(rows) != len(manifest_rows):
        raise ValueError('manifest contains duplicate job keys')
    latest = {}
    for event in events:
        key = event.get('job_key')
        if key not in rows:
            raise ValueError('ledger contains an unlisted job')
        kind = event.get('kind')
        attempt = event.get('attempt')
        if type(attempt) is not int or attempt < 1:
            raise ValueError('ledger has invalid attempt')
        if kind == 'start':
            if (event.get('config_sha256') != rows[key]['config_sha256'] or
                    (key in latest and latest[key]['kind'] != 'finish') or
                    (key in latest and latest[key]['classification'] !=
                     'infrastructure')):
                raise ValueError('ledger start differs from approved retry')
        elif kind == 'finish':
            if (key not in latest or latest[key]['kind'] != 'start' or
                    latest[key]['attempt'] != attempt or
                    event.get('classification') not in
                    ('success', 'training_anomaly', 'infrastructure')):
                raise ValueError('ledger finish does not match start')
        else:
            raise ValueError('ledger contains an unknown event')
        latest[key] = event
    pending, reconstructable, anomalies, finished = [], [], [], {}
    for row in manifest_rows:
        key = row['job_key']
        event = latest.get(key)
        if event is None:
            pending.append(key)
            continue
        if event['kind'] == 'finish' and event['classification'] == 'training_anomaly':
            anomalies.append(key)
            finished[key] = 'training_anomaly'
            continue
        if event['kind'] == 'finish' and event['classification'] == 'infrastructure':
            raise ValueError('unresolved infrastructure failure requires its own audit')
        run = root / key / f"attempt-{event['attempt']:02d}"
        gate = verify_sealed_job(
            run, key, manifest_sha256, replay=row['method'] in REPLAY_ARMS)
        digest = hashlib.sha256((run / 'JOB_SUCCESS.json').read_bytes()).hexdigest()
        if event['kind'] == 'start':
            reconstructable.append(key)
        elif event.get('job_success_sha256') != digest:
            raise ValueError('finished gate differs from append-only ledger')
        finished[key] = 'success'
    return {'pending': pending, 'reconstructable': reconstructable,
            'anomalies': anomalies, 'finished': finished}
