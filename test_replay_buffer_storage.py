import unittest

import torch

from cl_methods.der_pp import ReservoirBuffer as DERBuffer
from cl_methods.er import ReservoirBuffer as ERBuffer


class ReplayBufferStorageTests(unittest.TestCase):
    def test_er_copies_samples_out_of_loader_shared_storage(self):
        batch = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).share_memory_()
        labels = torch.tensor([0, 1])
        expected = batch[0].clone()

        buffer = ERBuffer(per_class_size=2)
        buffer.add_batch(batch, labels)
        saved = buffer.data[0][0]

        self.assertNotEqual(
            saved.untyped_storage().data_ptr(),
            batch.untyped_storage().data_ptr(),
        )
        batch[0].zero_()
        torch.testing.assert_close(saved, expected)

    def test_der_copies_example_and_logit_out_of_caller_storage(self):
        example = torch.arange(12, dtype=torch.float32).reshape(3, 4).share_memory_()
        logit = torch.arange(5, dtype=torch.float32).share_memory_()
        expected_example = example.clone()
        expected_logit = logit.clone()

        buffer = DERBuffer(buffer_size=2)
        buffer.add(example, torch.tensor(1), logit)

        self.assertNotEqual(
            buffer.ex[0].untyped_storage().data_ptr(),
            example.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            buffer.lg[0].untyped_storage().data_ptr(),
            logit.untyped_storage().data_ptr(),
        )
        example.zero_()
        logit.zero_()
        torch.testing.assert_close(buffer.ex[0], expected_example)
        torch.testing.assert_close(buffer.lg[0], expected_logit)


if __name__ == '__main__':
    unittest.main()
