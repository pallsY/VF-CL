import unittest

import replay_fim_queue as q


class ReplayFIMQueueTests(unittest.TestCase):
    def test_gpu_headroom_and_uuid_are_required(self):
        snapshot = {'index': 0, 'name': 'NVIDIA GeForce RTX 3080',
                    'uuid': q.GPU_UUID[0], 'free_mib': 4000}
        self.assertTrue(q.gpu_eligible(snapshot, 0))
        self.assertFalse(q.gpu_eligible({**snapshot, 'free_mib': 3000}, 0))
        self.assertFalse(q.gpu_eligible({**snapshot, 'index': 1}, 0))
        self.assertFalse(q.gpu_eligible({**snapshot, 'name': 'Other GPU'}, 0))
        self.assertFalse(q.gpu_eligible({**snapshot, 'free_mib': 5000}, 1))

    def test_any_non_success_stops_new_jobs(self):
        self.assertTrue(q.must_stop('training_anomaly'))
        self.assertTrue(q.must_stop('infrastructure'))
        self.assertTrue(q.must_stop('unknown'))
        self.assertFalse(q.must_stop('success'))

    def test_gpu_admission_requires_120_continuous_seconds(self):
        class Stop:
            elapsed = 0
            def is_set(self):
                return False
            def wait(self, seconds):
                self.elapsed += seconds

        stop = Stop()
        calls = []
        def snapshot(index):
            calls.append(stop.elapsed)
            return {'index': index, 'name': 'NVIDIA GeForce RTX 3080',
                    'uuid': q.GPU_UUID[index],
                    'free_mib': 2000 if stop.elapsed == 40 else 6000}
        result = q.wait_headroom(1, stop, snapshot_fn=snapshot,
                                 clock_fn=lambda: stop.elapsed)
        self.assertEqual(result['uuid'], q.GPU_UUID[1])
        self.assertGreaterEqual(stop.elapsed, 160)
        self.assertLess(stop.elapsed, 200)


if __name__ == '__main__':
    unittest.main()
