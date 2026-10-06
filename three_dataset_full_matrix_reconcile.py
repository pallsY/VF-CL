"""Read-only, fail-closed reuse evidence for the full public matrix."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from urllib.parse import quote

import three_dataset_seed42_reconcile as legacy
import three_dataset_formal_registry as registry
from three_dataset_formal_metrics import FORMULA_VERSION
from adaptive_consolidation_audit import FORMAL_SOURCE_FILES


FORMAL_ROOT = '/home/c3080/YangXiaoXiang/VF-CL/results/three_dataset_formal_20260906_021954'
FORMAL_COMMIT = 'ee0fb4c749259a4c3be0db21453de3539b73217b'
ADAPTIVE_ROOT = '/home/c3080/YangXiaoXiang/VF-CL/results/three_dataset_seed42_adaptive_recovery_20260911T013501Z'
ADAPTIVE_PIN = {
    'FORMAL_ROOT_IDENTITY.json': 'afc515548b430c0a79045d6eb6208e48e7404e4f53312c7b012f7d6d0ba90838',
    'FORMAL_REGISTRY.json': 'bcae44877102c4e40ccfbbc3a2916799554a414c0a396ff5f934877cab19ae4f',
    'FORMAL_PLAN.json': 'd4b8bced5d659c5ce8a44d8fa24835b84953c82dc129a8cf7c84e9592b5f03d5',
    'MISSING_JOBS.json': '0902c3907a84f82157ba1b29d7afab3711a375cd4bf7b4389b7278b299b388c8',
    'COMPATIBILITY_CENSUS.json': 'f199c90fa9caaa62d47cab3c53230eade679201dd5b61793efee0cddae6e8d1d',
    'RECOVERY_PHASE_SUCCESS': 'a4931e5257599f218c9d2744ed2e8b2545487cc08e91bff96f2a7d2bd18f0218',
    'RECOVERY_EXECUTION_SUCCESS': '438c66ebe7658a55632c46c113b9eec194eaffd057f8afc1d40ed8a90a642a3b',
    'records/cifar100%3Aadaptive%3A42.json': 'ca2a0807eba3881bbcd91beb47875373958f9209595fa244c3268b6aa93da2e9',
    'records/isolet%3Aadaptive%3A42.json': '1a7b27f388f970d65e129d5e8206e871dbae5964fd5d0aadd931c4d52c8204cd',
    'records/upmc_food101%3Aadaptive%3A42.json': '8d4740de02611d879261e5d274307cf18bd03ebf2285ea9adb6720543efaefca',
    'runs/cifar100%3Aadaptive%3A42/PRUNE_PLAN.json': 'e968a5665df426bf35902054ca56d8d96778019c2706e25c7746de69b5512f83',
    'runs/isolet%3Aadaptive%3A42/PRUNE_PLAN.json': '981d44b5b0cc16ad8fe51c5b13ace670a7ea1d7588233db919556afd47c08427',
    'runs/upmc_food101%3Aadaptive%3A42/PRUNE_PLAN.json': '35887318e065ebd77dbaf44be8cf361a962c1581a3c2a93258fd01a8d1b7171e',
    'runs/cifar100%3Aadaptive%3A42/PRUNED_EVIDENCE.json': '4ea7fb01c591e5ac69cf0adff3aa56338d0651c956a51c5731d409591e5a9a26',
    'runs/isolet%3Aadaptive%3A42/PRUNED_EVIDENCE.json': '1069e419ad45f5bd83164f43064d27c76d1dfc3fb344c44b565efcda6aa3aa1b',
    'runs/upmc_food101%3Aadaptive%3A42/PRUNED_EVIDENCE.json': 'f91caf017289443e3ee078dc4d8e209c23d4236369be7f78a93584ce4ff81472',
}
FORMAL_ROOT_PIN = {
    'FORMAL_ROOT_IDENTITY.json': 'b2ae3c83516c2db6cebe23e18b93e86ff1f5776c52e69fa75f2b81949d9ec0aa',
    'FORMAL_REGISTRY.json': 'f8f02516c7ec215cf1bd70d566e2e01d98ea1aca0d5995b6d8d54bc298000781',
    'COMPATIBILITY_CENSUS.json': '56330f4177604e43cb341bc037f2033807c3d3680925965702da7a55d5b620c3',
    'FORMAL_PLAN.json': 'b602128f2647bcb5f5678e2feca09abcd5d36e9d93cfac475d713269a00840a4',
    'MISSING_JOBS.json': '3fe6db1d918e9be2a9db475eafd23952bfbab4c0e82320f42b8186bc498c5205',
    'FAILED_JOB': '7436fb917b36c0019fb2713c50023f607200ca23d50528f220f0735a656a35f8',
    'FORMAL_STOPPED': '723d1bca4b8b46be09f606427f504c52444f3caf7f04bc6a088f07e14ba20cab',
}
FORMAL_EVIDENCE_PIN = {
    'records/cifar100%3Afinetune%3A42.json': '23fc81e41ebf540ee1186fae91361d8945a796b07c93c6869282beb235c9551c',
    'records/cifar100%3Afinetune%3A43.json': '6c16159fd46c3304a4acb160684b585406d91d875ccd4ca685ce876fdd2b42da',
    'records/cifar100%3Afinetune%3A44.json': '8b429cc392f9853a98b8308f93dadb22813b3715ee36054c19a1720e549987b8',
    'records/cifar100%3Alwf%3A42.json': '273aeb62af450b24ac1f5aea724dd1d3df4326c5143ff76b2980cef3a7547522',
    'records/cifar100%3Alwf%3A43.json': 'e5c2c15b88f8c579b62cc2b45f7503db82fbecf352d5054e27f09119f7da5391',
    'records/cifar100%3Alwf%3A44.json': '0537efbb13884f319305b5d7e20ad15bbbd2e5c73acdf64ce699d0cad3a3a103',
    'runs/cifar100%3Afinetune%3A42/PRUNE_PLAN.json': '6c44d957de8b2d0ebfb6189a51b6f7ce67d50408842fa67001061d757d215e28',
    'runs/cifar100%3Afinetune%3A43/PRUNE_PLAN.json': 'c477edcd3ebeb6fd25865ea301f5cbb929dfcf40fd73b203d271b50b280a53b7',
    'runs/cifar100%3Afinetune%3A44/PRUNE_PLAN.json': '500351ba14c55d3caf0d3d4af73d36f71c3776c4e638c19674941e1d0375b746',
    'runs/cifar100%3Alwf%3A42/PRUNE_PLAN.json': '416f71bcc63a9d0e1e064a0c8a1c545d297baf36cfc1bd56e2abb607c7e21f6f',
    'runs/cifar100%3Alwf%3A43/PRUNE_PLAN.json': 'ed7769bbb6da396ab5bb32c2f2478effe8c298b1ccfaf27c28dec85fb8be5db4',
    'runs/cifar100%3Alwf%3A44/PRUNE_PLAN.json': 'd3ec27b64c69be6491aba69fd05f2d14c7705321208e1c3958032219628a689a',
    'runs/cifar100%3Afinetune%3A42/PRUNED_EVIDENCE.json': '9209f923440ba38de11975f9b881005e13216abdb44db67972676a2766ed54d6',
    'runs/cifar100%3Afinetune%3A43/PRUNED_EVIDENCE.json': '5a3105604a1f107d3bc3306a45cf47cc7cc59700f9b0f869c60afe4fff309f7c',
    'runs/cifar100%3Afinetune%3A44/PRUNED_EVIDENCE.json': '1af83cbdbe9f2048123a28a49ce9ad7ec180a0330858f3566b10835ccb416be7',
    'runs/cifar100%3Alwf%3A42/PRUNED_EVIDENCE.json': '00c5b1aca30b64fb9e11bed812050d86e919ee4658cac5b83dd8258143cad3ff',
    'runs/cifar100%3Alwf%3A43/PRUNED_EVIDENCE.json': '7b6045c8c813ccd21411ea96aa879b854d8c63c8eb44194aff6e7b2ac163b03e',
    'runs/cifar100%3Alwf%3A44/PRUNED_EVIDENCE.json': '99e90b77f1023d7e76656f58c785ca2d0fd5e2847e68c1894665fc56aeeacab8',
}
FORMAL_KEYS = tuple(f'cifar100:{m}:{s}' for m in ('finetune', 'lwf') for s in (42, 43, 44))
OPTIONAL_KEYS = tuple(k for k in FORMAL_KEYS if not k.endswith(':42'))

# Reviewed Tasks 3-7 and CIFAR smoke amendments. Later edits need a new proof.
APPROVED_OPERATIONAL_MIGRATIONS = {
    "three_dataset_formal_driver.py": [
        {
            "commit": "1d1e59331fa5800678454ef063c531dda1d3020d",
            "parent": "c86f03dd14dc12556beeb260daa4476fdf3170af",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "94da8cd843f019f7a1074bce54fb68d636bb2d35",
            "new_blob": "62024156ef0c33d4c33fbff5ecf79c8d90cc2789",
            "old_sha256": "951d9f0797a1204fc654c2bb1ef310b60c47b3dca39478cd2e32eab1405039c2",
            "new_sha256": "ffa3b29225724fbec188391ce6f629c599e3f94c06ab0a9f4e7eaa748b93f391"
        },
        {
            "commit": "5238d731a8310e3aac842899c1eed200d69aa622",
            "parent": "1d1e59331fa5800678454ef063c531dda1d3020d",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "62024156ef0c33d4c33fbff5ecf79c8d90cc2789",
            "new_blob": "81deb7835ab37d7f687f520a5ed9f797e5ef703f",
            "old_sha256": "ffa3b29225724fbec188391ce6f629c599e3f94c06ab0a9f4e7eaa748b93f391",
            "new_sha256": "d1baa6c9cf589b85420d13eb4745e3c2671618e896475db4fa6a93d075483cd6"
        },
        {
            "commit": "510faaf5f44a15d02946be8e52e6fb390d476d8f",
            "parent": "5238d731a8310e3aac842899c1eed200d69aa622",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "81deb7835ab37d7f687f520a5ed9f797e5ef703f",
            "new_blob": "aa71c12e35eb36d299fd202010438328527a3af1",
            "old_sha256": "d1baa6c9cf589b85420d13eb4745e3c2671618e896475db4fa6a93d075483cd6",
            "new_sha256": "65bb1481b76e9085e364788490e8ca757b28e2e87779326f2c06eb7863baa6f4"
        },
        {
            "commit": "4363151e4581605f58606a3220a451bf05541bd3",
            "parent": "510faaf5f44a15d02946be8e52e6fb390d476d8f",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "aa71c12e35eb36d299fd202010438328527a3af1",
            "new_blob": "7b507ccc59f0b1f279d2ce3345e5a3783484cca4",
            "old_sha256": "65bb1481b76e9085e364788490e8ca757b28e2e87779326f2c06eb7863baa6f4",
            "new_sha256": "a9cc5f50d8c9b253f63ca06aa078676338761e840b02e56f41ae61fdf1364b94"
        },
        {
            "commit": "8d72a2f89fb5bebd7d5196ff95991e17f5a6a67e",
            "parent": "4363151e4581605f58606a3220a451bf05541bd3",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "7b507ccc59f0b1f279d2ce3345e5a3783484cca4",
            "new_blob": "53194e7cae8466838fd848971de15f21822b30ea",
            "old_sha256": "a9cc5f50d8c9b253f63ca06aa078676338761e840b02e56f41ae61fdf1364b94",
            "new_sha256": "c92ad548cd51967153188ca4d1ce688b2ba2b1075ea85556bf500851d4df5efc"
        },
        {
            "commit": "86a31ea5ab6ad5254fc06e1328b9dc4dbf30f392",
            "parent": "1c5e556aab145698036a4e28f283592a3c6645b0",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "53194e7cae8466838fd848971de15f21822b30ea",
            "new_blob": "73488fbd18f4c559227e2f5f3c4172b3f50e743b",
            "old_sha256": "c92ad548cd51967153188ca4d1ce688b2ba2b1075ea85556bf500851d4df5efc",
            "new_sha256": "444e4ff90ba3d3f3fd40a5b88dd726bf8ed25c0cb9e27f3f34a5a516a2bf4dcc"
        },
        {
            "commit": "8cfcdaaed05a090d84bc20412a3ef8b8732b01aa",
            "parent": "36181fcfaac66dca309228627fdf8f86212effce",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "73488fbd18f4c559227e2f5f3c4172b3f50e743b",
            "new_blob": "3f30a727aaee8325f81599b2dc673ab8c1c3a920",
            "old_sha256": "444e4ff90ba3d3f3fd40a5b88dd726bf8ed25c0cb9e27f3f34a5a516a2bf4dcc",
            "new_sha256": "14b8e113b0ab09e2103f9210a2556f43584fc54311c9d6ce9a300b13a45195fa"
        },
        {
            "commit": "ca5a80b40f5b47b8531cdde19d06cdeffc438af2",
            "parent": "86f2caa4847ca6cdeffe25f79135ac0f2e25df09",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "3f30a727aaee8325f81599b2dc673ab8c1c3a920",
            "new_blob": "21c8d641d2b13a284564ac30c5306adc7caac5bd",
            "old_sha256": "14b8e113b0ab09e2103f9210a2556f43584fc54311c9d6ce9a300b13a45195fa",
            "new_sha256": "a5afd117f7c2d5d2ca584045d1401bdc730e8ad89d53e345afbbb4c78dfa92fb"
        },
        {
            "commit": "e0ff10dd00bfb492a96bbf7cd60b9c4827cb89ee",
            "parent": "276f0d702af9c8160f2636ea7f245851ba3a0c0a",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "21c8d641d2b13a284564ac30c5306adc7caac5bd",
            "new_blob": "7e1ac11301de3574ae6de85c9b0a8b7ed3ee7844",
            "old_sha256": "a5afd117f7c2d5d2ca584045d1401bdc730e8ad89d53e345afbbb4c78dfa92fb",
            "new_sha256": "200ce77f8fb15811d13454aa072610428a2341615ac40c608589ea64e568f416"
        }
    ],
    "three_dataset_full_matrix_report.py": [
        {
            "commit": "1d1e59331fa5800678454ef063c531dda1d3020d",
            "parent": "c86f03dd14dc12556beeb260daa4476fdf3170af",
            "old_mode": "000000",
            "new_mode": "100644",
            "old_blob": "0000000000000000000000000000000000000000",
            "new_blob": "764ac912e182741b9c5aa7f7dc2d555e964d61d7",
            "old_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "new_sha256": "ed0e1fc8e431fc907eae0dd8ff1326ad0a32d8487573915c39aebd6d24ea2d7e"
        }
    ],
    "three_dataset_resource_gate.py": [
        {
            "commit": "5238d731a8310e3aac842899c1eed200d69aa622",
            "parent": "1d1e59331fa5800678454ef063c531dda1d3020d",
            "old_mode": "000000",
            "new_mode": "100644",
            "old_blob": "0000000000000000000000000000000000000000",
            "new_blob": "9314ae5dd60b4fa0806bd2a8f2290cb8c88cb97d",
            "old_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "new_sha256": "8321f467528636a88586b3a897a380d578a6d5567e50008a461a91b2dc42bd49"
        }
    ],
    "run_three_dataset_formal_comparison.sh": [
        {
            "commit": "510faaf5f44a15d02946be8e52e6fb390d476d8f",
            "parent": "5238d731a8310e3aac842899c1eed200d69aa622",
            "old_mode": "100755",
            "new_mode": "100755",
            "old_blob": "e2d90b9c65fab83d925a5adb1e3755c232306db5",
            "new_blob": "81d86b0be20e74a998c2276e9baa4cb440078640",
            "old_sha256": "8f11f4f5cf07852fd393169d644e0fb155724f0283c84337117a83e463c23e87",
            "new_sha256": "c61b243ac274b1e2dc1796be8cebe9b7fd1146f564f96a3d10a4bd1b0b1592a7"
        },
        {
            "commit": "4363151e4581605f58606a3220a451bf05541bd3",
            "parent": "510faaf5f44a15d02946be8e52e6fb390d476d8f",
            "old_mode": "100755",
            "new_mode": "100755",
            "old_blob": "81d86b0be20e74a998c2276e9baa4cb440078640",
            "new_blob": "c003905f841d66c869c2b4cf96c16a71ffcf6c21",
            "old_sha256": "c61b243ac274b1e2dc1796be8cebe9b7fd1146f564f96a3d10a4bd1b0b1592a7",
            "new_sha256": "fe319d1734950260d5f24d02047c61789b14eb36794b7c881a3f726d04e2619d"
        }
    ],
    "prune_completed_runs.py": [
        {
            "commit": "510faaf5f44a15d02946be8e52e6fb390d476d8f",
            "parent": "5238d731a8310e3aac842899c1eed200d69aa622",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "4682798f25c3b431671099c243aeb4605bfa7d9d",
            "new_blob": "92077b631f39cc62b8c96597cd7f27d1d1155f6b",
            "old_sha256": "025f1497c0fc96b84938a19a7e3bd9c5934ec5d64927179e288199a19e9651ac",
            "new_sha256": "5118689771fc0fbb5772975bbd12413dd8f7a0bb061f507eee01a1fe8f4c611c"
        },
        {
            "commit": "04a08f071eeeebcc86b7741e837c166c85165737",
            "parent": "4089e2e6f0aaf5dcd85bb27954243187f35d1bb0",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "92077b631f39cc62b8c96597cd7f27d1d1155f6b",
            "new_blob": "462a095d280bdc8ed780cc510be02db5dd3f2f46",
            "old_sha256": "5118689771fc0fbb5772975bbd12413dd8f7a0bb061f507eee01a1fe8f4c611c",
            "new_sha256": "e2570a1d6e806b2e796d02d7b0200b1dd00d1bce504a509700e45c0cd2a28d78"
        },
        {
            "commit": "cc4241c58be8882e6cefd52f62c42503947dd822",
            "parent": "772c4c3f33e25540d2d8ea278b339476e7c2dbc9",
            "old_mode": "100644",
            "new_mode": "100644",
            "old_blob": "462a095d280bdc8ed780cc510be02db5dd3f2f46",
            "new_blob": "275e7b86cca909d1e9feff5c6c0713a3d4650566",
            "old_sha256": "e2570a1d6e806b2e796d02d7b0200b1dd00d1bce504a509700e45c0cd2a28d78",
            "new_sha256": "eafbd5822ab5a30892dc064a0871c8c0ff68da14d00dbbcabcb6ac8082e40b5e"
        }
    ]
}
OPERATIONAL_PATHS = frozenset((
    "three_dataset_formal_driver.py", "three_dataset_full_matrix_report.py",
    "three_dataset_resource_gate.py", "run_three_dataset_formal_comparison.sh",
    "prune_completed_runs.py",
))



def _row(root, row, migration_hash):
    path = 'records/' + quote(row['spec_key'], safe='') + '.json'
    record = root.json(path)
    legacy.require(legacy.sha(root.read(path, immutable=True)) == row['record_file_sha256'],
                   'confirmed record changed')
    return {**{k: row[k] for k in ('spec_key', 'dataset', 'method', 'seed', 'metrics')},
            'resource': record['resource'], 'artifact_sha256': record['artifact_sha256'],
            'source_sha256': {k[7:]: v for k, v in record['artifact_sha256'].items() if k.startswith('source:')},
            'trajectory_sha256': record['trajectory_sha256'], 'origin_root': str(root.path),
            'origin_record_path': str(root.path / path), 'origin_record_sha256': row['record_file_sha256'],
            'origin_source_commit': row['source_commit'], 'compatibility': 'ADMITTED',
            'migration_sha256': migration_hash}


def _resource_protocol(resource):
    return {k: v for k, v in resource.items() if k not in ('runtime_seconds', 'peak_gpu_memory_bytes')}


def _command_family(command):
    result = list(command)
    if result[1] in (
            '/home/c3080/YangXiaoXiang/VF-CL-worktrees/three-dataset-formal-comparison-3080/main.py',
            '/home/c3080/YangXiaoXiang/VF-CL-worktrees/formal-pipelined-audit-3080/main.py'):
        result[1] = '<reviewed-worktree>/main.py'
    # Only seed and output names vary within a pinned historical method family.
    for option in ('--results_dir', '--exp_name', '--seed'):
        result[result.index(option) + 1] = '<' + option[2:] + '>'
    return result


def _formal_authority(root, worktree, head):
    for name, expected in {**FORMAL_ROOT_PIN, **FORMAL_EVIDENCE_PIN}.items():
        legacy.require(legacy.sha(root.read(name, immutable=True)) == expected,
                       'formal_pin_mismatch: ' + name)
    identity = root.json('FORMAL_ROOT_IDENTITY.json')
    legacy.require(identity['root_dev'] == root.details.st_dev and identity['root_inode'] == root.details.st_ino
                   and identity['source_commit'] == FORMAL_COMMIT, 'formal_root_identity_mismatch')
    legacy.git(worktree, 'merge-base', '--is-ancestor', FORMAL_COMMIT, head)
    plan = root.json('FORMAL_PLAN.json')
    census = root.json('COMPATIBILITY_CENSUS.json')
    legacy.require(root.names('records') == {quote(k, safe='') + '.json' for k in FORMAL_KEYS},
                   'formal_record_membership_mismatch')
    owner = {'dev': root.details.st_dev, 'inode': root.details.st_ino,
             'ctime_ns': root.seen['FORMAL_ROOT_IDENTITY.json'][-1],
             'size': root.seen['FORMAL_ROOT_IDENTITY.json'][4],
             'hash': legacy.sha(root.read('FORMAL_ROOT_IDENTITY.json', immutable=True))}
    auth = {'identity': identity, 'owner_identity': owner, 'plan': plan, 'binding': {},
            'commit': FORMAL_COMMIT, 'protocols': {r['spec_key']: r['protocol_sha256'] for r in census['records']}}
    return auth


def _formal_rows(root, pilot, admitted, worktree, head, migration_hash):
    auth = _formal_authority(root, worktree, head)
    by_key = {r['spec_key']: r for r in admitted}
    rows = []
    for key in FORMAL_KEYS:
        baseline = key.rsplit(':', 1)[0] + ':42'
        job, _, tasks = legacy.job_spec(root, auth, key)
        collected = legacy.completed(root, auth, key, job, tasks)
        legacy.require(type(collected['seed']) is int and collected['seed'] in (42, 43, 44),
                       'formal_seed_mismatch: ' + key)
        pilot_job = pilot.json('runs/' + quote(baseline, safe='') + '/FORMAL_JOB_SPEC.json')
        legacy.require(legacy.identical(job['source_sha256'], by_key[baseline]['source_sha256']),
                       'formal_source_mismatch: ' + key)
        legacy.require(legacy.identical(_command_family(job['command']), _command_family(pilot_job['command'])),
                       'formal_command_family_mismatch: ' + key)
        row = _row(root, collected, migration_hash)
        if key.endswith(':42'):
            legacy.require(legacy.identical(row['metrics'], by_key[key]['metrics']),
                           'duplicate_seed42_metrics_differ: ' + key)
            legacy.require(legacy.identical(_resource_protocol(row['resource']),
                                           _resource_protocol(by_key[key]['resource'])),
                           'duplicate_seed42_resource_protocol_differs: ' + key)
            pilot_record = pilot.json('records/' + quote(key, safe='') + '.json')
            legacy.require(auth['protocols'][key] == pilot_record['protocol_sha256'],
                           'duplicate_seed42_protocol_differs: ' + key)
        else:
            rows.append(row)
    root.verify()
    source = {'root': str(root.path), 'source_commit': FORMAL_COMMIT, 'authority_identity': auth['identity'],
              'identity_file_sha256': auth['owner_identity']['hash']}
    return rows, source


def _zero_reuse(pilot, adaptive, formal_root, worktree, head, origin, stack, inputs):
    """Prove rejection only; a single possible survivor requires the strict path."""
    legacy.require(not legacy.git(worktree, 'status', '--porcelain', '--untracked-files=all').strip(),
                   'recovery worktree is not clean')
    current = stack.enter_context(legacy.Evidence(worktree))
    inputs.append(current)
    source_hashes = {}
    for path in FORMAL_SOURCE_FILES:
        source_hashes[path] = legacy.sha(current.read(path))
        legacy.require(source_hashes[path] == legacy.sha(legacy.git_bytes(worktree, 'show', head + ':' + path)),
                       'formal source differs from Git commit: ' + path)
    sources, rejected = [], {}
    roots = [(pilot, legacy.authority(pilot, False, legacy.PILOT_COMMIT)),
             (adaptive, legacy.authority(adaptive, True, origin))]
    for index in range(3):
        if index == 2:
            # Unavailable/invalid optional evidence cannot prove all candidates rejected.
            # Let the unchanged strict path decide it, outside its optional catch.
            try:
                legacy.require(str(Path(formal_root)) == FORMAL_ROOT, 'formal_root_path_mismatch')
                formal = stack.enter_context(legacy.Evidence(formal_root))
                inputs.append(formal)
                roots.append((formal, _formal_authority(formal, worktree, head)))
            except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError):
                return None
        root, auth = roots[index]
        keys = (FORMAL_KEYS if index == 2 else
                [key for key in auth['keys'] if index == 1 or ':adaptive:' not in key])
        legacy.require(root.names('records') == {quote(k, safe='') + '.json' for k in keys},
                       'completed record membership differs (old adaptive prohibited)')
        sources.append({'root': str(root.path), 'source_commit': auth['commit'],
                        'authority_identity': auth['identity'],
                        'identity_file_sha256': auth['owner_identity']['hash']})
        for key in keys:
            job, _, _ = legacy.job_spec(root, auth, key)
            path = 'records/' + quote(key, safe='') + '.json'
            record = root.json(path)
            legacy.require(record['spec_key'] == key and record['source_commit'] == auth['commit'],
                           'completed record authority differs')
            recorded = {k[7:]: v for k, v in record['artifact_sha256'].items() if k.startswith('source:')}
            legacy.require(recorded == job['source_sha256'], 'record source provenance differs')
            missing, extra = sorted(set(source_hashes) - set(recorded)), sorted(set(recorded) - set(source_hashes))
            if missing or extra:
                reason = 'source-inventory-mismatch: missing=' + ','.join(missing) + '; extra=' + ','.join(extra)
            else:
                changed = sorted(path for path in source_hashes if recorded[path] != source_hashes[path]
                                 and path not in legacy.APPROVED_SOURCE_MIGRATIONS)
                if not changed:
                    return None
                reason = 'source-hash-mismatch: ' + ','.join(changed)
            candidate = {'origin_root': str(root.path), 'origin_source_commit': auth['commit'],
                         'origin_record_path': str(root.path / path),
                         'origin_record_sha256': legacy.sha(root.read(path, immutable=True)),
                         'compatibility': 'REJECTED', 'reason': reason}
            rejected.setdefault(key, []).append(candidate)
    rows = [{'spec_key': key, 'origin_root': candidates[0]['origin_root'],
             'compatibility': 'REJECTED', 'reason': '; '.join(sorted({r['reason'] for r in candidates})),
             'candidates': candidates} for key, candidates in rejected.items()]
    rows.sort(key=lambda r: (registry.DATASETS.index(r['spec_key'].split(':')[0]),
                            registry.FULL_MATRIX_METHODS.index(r['spec_key'].split(':')[1]),
                            int(r['spec_key'].split(':')[2])))
    for root in inputs:
        root.verify()
    legacy.require(legacy.git(worktree, 'rev-parse', 'HEAD').strip() == head, 'current Git HEAD changed')
    return {'kind': 'full_public_matrix_reuse_v1', 'experiment_profile': registry.FULL_MATRIX_PROFILE,
            'registry_sha256': registry.registry_sha256(), 'metric_formula_version': FORMULA_VERSION,
            'current_commit': head, 'sources': sources, 'admitted': [], 'rejected': rows, 'ambiguous': []}


def _operational_tree_entry(worktree, revision, path):
    entry = legacy.git(worktree, 'ls-tree', revision, '--', path).strip().split(None, 3)
    if not entry:
        return '000000', '0' * 40, legacy.sha(b'')
    legacy.require(len(entry) == 4 and entry[1] == 'blob' and entry[3] == path
                   and entry[0] in ('100644', '100755'),
                   'operational migration tree entry differs')
    return entry[0], entry[2], legacy.sha(legacy.git_bytes(worktree, 'cat-file', 'blob', entry[2]))


def _approved_operational_migrations(worktree, origin, head, changes):
    legacy.require(set(APPROVED_OPERATIONAL_MIGRATIONS) <= OPERATIONAL_PATHS,
                   'operational migration paths differ')
    approved = {}
    fields = {'commit', 'parent', 'old_mode', 'new_mode', 'old_blob', 'new_blob',
              'old_sha256', 'new_sha256'}
    try:
        for path in sorted(set(changes) & OPERATIONAL_PATHS):
            chain = APPROVED_OPERATIONAL_MIGRATIONS.get(path)
            legacy.require(type(chain) in (list, tuple) and bool(chain),
                           'operational migration chain missing')
            for item in chain:
                legacy.require(type(item) is dict and set(item) == fields,
                               'operational migration schema differs')
                legacy.require(all(type(item[k]) is str and legacy.COMMIT.fullmatch(item[k])
                                   for k in ('commit', 'parent', 'old_blob', 'new_blob'))
                               and all(type(item[k]) is str and legacy.SHA.fullmatch(item[k])
                                       for k in ('old_sha256', 'new_sha256')),
                               'operational migration commit/blob/hash invalid')
            touches = legacy.git(worktree, 'log', '--full-history', '-m', '--format=%H',
                                 origin + '..' + head, '--', path).splitlines()
            legacy.require(touches == [item['commit'] for item in reversed(chain)],
                           'operational migration history differs')
            previous = _operational_tree_entry(worktree, origin, path)
            for item in chain:
                legacy.require(legacy.git(worktree, 'rev-list', '--parents', '-n', '1',
                                          item['commit']).split() == [item['commit'], item['parent']],
                               'operational migration parent differs')
                old = tuple(item['old_' + k] for k in ('mode', 'blob', 'sha256'))
                new = tuple(item['new_' + k] for k in ('mode', 'blob', 'sha256'))
                legacy.require(old == previous == _operational_tree_entry(worktree, item['parent'], path)
                               and new == _operational_tree_entry(worktree, item['commit'], path),
                               'operational migration blob/mode/hash chain differs')
                previous = new
            legacy.require(previous == _operational_tree_entry(worktree, head, path),
                           'operational migration final differs')
            approved[path] = list(chain)
    except subprocess.SubprocessError as error:
        raise ValueError('operational migration Git proof failed') from error
    return approved


def _build(pilot_root, adaptive_root, formal_root, worktree, stack, inputs):
    legacy.require(os.environ.get(registry.PROFILE_ENV) == registry.FULL_MATRIX_PROFILE,
                   'VFCL_EXPERIMENT_PROFILE must be explicitly full-public-matrix')
    legacy.require(str(Path(pilot_root)) == legacy.PILOT_ROOT, 'pilot root path differs')
    legacy.require(str(Path(adaptive_root)) == ADAPTIVE_ROOT, 'adaptive root path differs')
    pilot = stack.enter_context(legacy.Evidence(pilot_root))
    adaptive = stack.enter_context(legacy.Evidence(adaptive_root))
    inputs.extend((pilot, adaptive))
    for name, expected in ADAPTIVE_PIN.items():
        legacy.require(legacy.sha(adaptive.read(name, immutable=True)) == expected,
                       'adaptive_pin_mismatch: ' + name)
    origin = adaptive.json('FORMAL_ROOT_IDENTITY.json')['source_commit']
    legacy.require(type(origin) is str and legacy.COMMIT.fullmatch(origin), 'recovery origin commit invalid')
    head = legacy.git(worktree, 'rev-parse', '--verify', 'HEAD^{commit}').strip()
    legacy.git(worktree, 'merge-base', '--is-ancestor', origin, head)
    rejected_bundle = _zero_reuse(pilot, adaptive, formal_root, worktree, head, origin, stack, inputs)
    if rejected_bundle is not None:
        return rejected_bundle
    changes = sorted({p.strip('\n') for p in legacy.git(worktree, 'log', '--format=', '--name-only',
                      '-z', '-m', origin + '..' + head, '--').split('\x00') if p.strip('\n')})
    allowed = {'three_dataset_formal_registry.py', 'three_dataset_seed42_reconcile.py',
               'three_dataset_full_matrix_reconcile.py'}
    operational = _approved_operational_migrations(worktree, origin, head, changes)
    for path in changes:
        legacy.require(path in allowed or path in operational
                       or legacy.re.fullmatch(r'test_[a-zA-Z0-9_]+\.py', path)
                       or (path.startswith('docs/') and path.endswith('.md')),
                       'origin-to-current Git scope forbids source: ' + path)
    report = legacy._collect_seed42_rows(pilot, adaptive, worktree, worktree, recovery_commit=origin,
                                         operational_paths=tuple(operational))
    legacy.require(report['git_scope']['recovery_head'] == head, 'current Git HEAD changed')
    report['git_scope']['approved_operational_migration'] = operational
    migration_hash = legacy.digest({'scientific': report['git_scope']['approved_source_migration'],
                                    'operational': operational})
    roots = {str(r.path): r for r in inputs}
    admitted = [_row(roots[r['source_root']], r, migration_hash) for r in report['rows']]
    sources = report['sources']
    rejected = []
    try:
        legacy.require(str(Path(formal_root)) == FORMAL_ROOT, 'formal_root_path_mismatch')
        try:
            formal = stack.enter_context(legacy.Evidence(formal_root))
        except FileNotFoundError:
            raise ValueError('formal_root_unavailable') from None
        additional, source = _formal_rows(formal, pilot, admitted, worktree, head, migration_hash)
        inputs.append(formal)
        admitted.extend(additional)
        sources.append(source)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        reason = str(error)
        rejected = [{'spec_key': key, 'origin_root': str(formal_root), 'reason': reason} for key in OPTIONAL_KEYS]
    for source in sources:
        source['git_scope'] = {**report['git_scope'], 'origin_commit': source['source_commit'],
                               'current_commit': head, 'adaptive_origin_to_current_paths': changes}
        source['migration_sha256'] = migration_hash
    admitted.sort(key=lambda r: (registry.DATASETS.index(r['dataset']),
                                registry.FULL_MATRIX_METHODS.index(r['method']), r['seed']))
    keys = [r['spec_key'] for r in admitted]
    legacy.require(len(keys) == len(set(keys)) and set(keys) <= {registry.registered_spec_key(s) for s in registry.formal_specs()},
                   'reuse membership differs from full matrix')
    for root in inputs:
        root.verify()
    legacy.require(legacy.git(worktree, 'rev-parse', 'HEAD').strip() == head, 'current Git HEAD changed')
    return {'kind': 'full_public_matrix_reuse_v1', 'experiment_profile': registry.FULL_MATRIX_PROFILE,
            'registry_sha256': registry.registry_sha256(), 'metric_formula_version': FORMULA_VERSION,
            'current_commit': head, 'sources': sources, 'admitted': admitted, 'rejected': rejected, 'ambiguous': []}


def build_reuse_bundle(pilot_root: Path, adaptive_root: Path, formal_root: Path, worktree: Path) -> dict:
    with ExitStack() as stack:
        return json.loads(legacy.canonical(_build(pilot_root, adaptive_root, formal_root, worktree, stack, [])))


def _publish(output, bundle, inputs, worktree):
    parent = legacy.directory(output.parent)
    fd = None
    try:
        os.mkdir(output.name, 0o700, dir_fd=parent)
        fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        legacy.require(stat.S_IMODE(os.fstat(fd).st_mode) == 0o700 and not os.listdir(fd), 'output mode/contents invalid')
        audit = {'kind': 'full_public_matrix_reuse_audit_v1', 'validation': 'ADMITTED',
                 'bundle_sha256': legacy.digest(bundle), 'admitted_count': len(bundle['admitted']),
                 'rejected': bundle['rejected'], 'ambiguous': bundle['ambiguous']}
        payloads = {'FULL_MATRIX_REUSE.json': legacy.canonical(bundle) + b'\n',
                    'FULL_MATRIX_REUSE_AUDIT.json': legacy.canonical(audit) + b'\n'}
        payloads['REUSE_AUDIT_SUCCESS'] = legacy.canonical({'kind': 'full_public_matrix_reuse_audit_success',
            'row_count': len(bundle['admitted']), 'artifact_sha256': {n: legacy.sha(b) for n, b in payloads.items()}}) + b'\n'
        for name, content in payloads.items():
            legacy.verify_output(output, parent, fd)
            temporary = '.' + name + '.tmp'
            handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            with os.fdopen(handle, 'wb') as stream:
                stream.write(content)
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
            legacy.require(name not in os.listdir(fd), 'output target already exists')
            if name == 'REUSE_AUDIT_SUCCESS':
                for root in inputs:
                    root.verify()
                legacy.require(legacy.git(worktree, 'rev-parse', 'HEAD').strip() == bundle['current_commit'],
                               'current Git HEAD changed')
            legacy.verify_output(output, parent, fd)
            # link is atomic and fails if the target exists; rename would overwrite it.
            os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            os.unlink(temporary, dir_fd=fd)
            os.fsync(fd)
        os.fsync(parent)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def reconcile(pilot_root, adaptive_root, formal_root, worktree, output_root):
    raw_output = os.fspath(output_root)
    output = Path(raw_output)
    legacy.require(output.anchor == '/' and not raw_output.startswith('//'),
                   'output must have exactly a single leading slash')
    legacy.require(output.is_absolute() and output.name not in ('', '.', '..'), 'output must be an absolute fresh path')
    for root in map(Path, (legacy.PILOT_ROOT, ADAPTIVE_ROOT, FORMAL_ROOT,
                           pilot_root, adaptive_root, formal_root)):
        legacy.require(output != root and root not in output.parents, 'output cannot be inside an input root')
    parent = legacy.directory(output.parent)
    try:
        legacy.require(output.name not in os.listdir(parent), 'output already exists')
    finally:
        os.close(parent)
    with ExitStack() as stack:
        inputs = []
        bundle = _build(pilot_root, adaptive_root, formal_root, worktree, stack, inputs)
        _publish(output, bundle, inputs, worktree)
        return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-root', 'adaptive-root', 'formal-root', 'worktree', 'output-root'):
        parser.add_argument('--' + name, type=str if name == 'output-root' else Path, required=True)
    args = parser.parse_args(argv)
    try:
        reconcile(args.pilot_root, args.adaptive_root, args.formal_root, args.worktree, args.output_root)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('reuse rejected: ' + str(error), file=sys.stderr)
        return 1
    print('REUSE_AUDIT_SUCCESS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
