"""Read-only validation of producer publication and one-shot test evidence."""

import hashlib
from pathlib import Path
from types import SimpleNamespace


def verify_formal_seal(run):
    # Import lazily: queue admission does not need the training runtime.
    import adaptive_consolidation_audit as producer
    from data_utils import VFLDataset

    run = Path(run)
    paths = {
        'results_file_sha256': run / 'results.json',
        'checkpoint_sha256': run / 'checkpoints/formal_final.pt',
        'formal_complete_sha256': run / 'FORMAL_EVALUATION_COMPLETE.json',
        'formal_sealed_sha256': run / 'FORMAL_EVALUATION_SEALED.json',
        'formal_published_sha256': run / 'FORMAL_EVALUATION_PUBLISHED.json',
        'formal_publishing_sha256': run / 'FORMAL_EVALUATION_PUBLISHING.json',
        'test_access_sha256': run / 'formal_access/test.consumed.json',
        'data_flow_audit_sha256': run / 'data_flow_audit.jsonl',
    }
    try:
        if any(not path.is_file() for path in paths.values()):
            raise ValueError('formal result or single test-access seal is absent')
        # Reuse the producer's exact transaction, evaluation, and file-record
        # validation, without loading checkpoints or opening any data loaders.
        complete, _seal = producer._load_sealed_complete_artifact(run)
        identity = complete['identity']
        marker_keys = {
            'schema_version', 'status', 'transaction_sha256',
            'evaluation_sha256', 'results_sha256', 'seal', 'source_provenance',
        }
        publishing = producer._formal_marker(
            paths['formal_publishing_sha256'], 'publishing', marker_keys)
        published = producer._formal_marker(
            paths['formal_published_sha256'], 'published',
            marker_keys | {'checkpoint', 'results'})
        results = producer._safe_json(paths['results_file_sha256'])
        content_sha = hashlib.sha256(
            producer._strict_json(results).encode('utf-8')).hexdigest()
        expected = dict(
            schema_version=1, status='publishing',
            transaction_sha256=complete['transaction_sha256'],
            evaluation_sha256=complete['evaluation_sha256'],
            results_sha256=content_sha,
            seal=producer._formal_file_record(paths['formal_sealed_sha256'], run),
            source_provenance=identity['source_provenance'])
        if (publishing != expected or published != dict(
                expected, status='published',
                checkpoint=producer._formal_file_record(paths['checkpoint_sha256'], run),
                results=producer._formal_file_record(paths['results_file_sha256'], run))
                or results.get('source_provenance') != identity['source_provenance']
                or any(results.get(key) != value
                       for key, value in complete['evaluation'].items())):
            raise ValueError('formal publication content or file record mismatch')
        protocol = identity['protocol']
        if (identity['protocol_sha256'] != hashlib.sha256(
                producer._strict_json(protocol).encode('utf-8')).hexdigest()):
            raise ValueError('formal protocol digest mismatch')
        # Construct no dataset: these methods only read durable access metadata.
        dataset = object.__new__(VFLDataset)
        num_tasks = protocol['num_tasks']
        task_classes = identity['task_classes']
        if (type(num_tasks) is not int or num_tasks <= 0
                or set(task_classes) != {str(task) for task in range(num_tasks)}):
            raise ValueError('formal task class protocol mismatch')
        dataset.args = SimpleNamespace(
            output_dir=str(run), num_tasks=num_tasks,
            custom_tasks=[task_classes[str(task)] for task in range(num_tasks)])
        access = dataset._read_formal_marker(paths['test_access_sha256'], 'consumed')
        access_files = list((run / 'formal_access').glob('test.*.json'))
        history = dataset._formal_audit_records()
        if (access_files != [paths['test_access_sha256']]
                or access['access']['split'] != 'test'
                or access['access']['classes'] != [
                    class_id
                    for task in range(protocol['num_tasks'])
                    for class_id in identity['task_classes'][str(task)]]
                or [item for item in history if item['split'] == 'test']
                != [access['access']]):
            raise ValueError('formal test was not accessed exactly once')
        hashes = {key: producer._formal_file_record(path, run)['sha256']
                  for key, path in paths.items()}
        return dict(hashes, results_content_sha256=content_sha)
    except (OSError, KeyError, TypeError, RuntimeError) as error:
        raise ValueError('formal publication or access evidence is invalid') from error
