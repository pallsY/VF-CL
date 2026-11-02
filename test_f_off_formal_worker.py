import hashlib
import tempfile
import unittest
from pathlib import Path

from f_off_formal_worker import validate_sources


class FormalWorkerTests(unittest.TestCase):
    def test_source_hashes_must_match_before_formal_access(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.json'
            data = Path(folder) / 'data.bin'
            validation = Path(folder) / 'validation.json'
            for path in (source, data, validation):
                path.write_bytes(path.name.encode())
            config = {'formal_followup_job_key': 'isolet/f_off/0.25/42',
                      'formal_deferred_evaluation': True}
            row = {'job_key': 'isolet/f_off/0.25/42',
                   'source_config_path': str(source),
                   'source_data_path': str(data),
                   'validation_manifest_path': str(validation),
                   'source_config_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'source_data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                   'validation_manifest_sha256': hashlib.sha256(validation.read_bytes()).hexdigest()}
            validate_sources(config, row)
            data.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                validate_sources(config, row)


if __name__ == '__main__':
    unittest.main()
