"""Parameter-free task-mass and within-task probability composition."""

import torch
import torch.nn.functional as F


def factorize_task_probabilities(log_p_mix, pre_logits, task_for_column):
    """Keep Mixed task masses and pre-head within-task class probabilities."""
    if (not isinstance(log_p_mix, torch.Tensor)
            or log_p_mix.ndim != 2 or log_p_mix.dtype != torch.float64
            or min(log_p_mix.shape) == 0 or not torch.isfinite(log_p_mix).all()):
        raise ValueError('Mixed probabilities must be finite float64 [N, C]')
    if (not isinstance(pre_logits, torch.Tensor)
            or pre_logits.shape != log_p_mix.shape
            or pre_logits.device != log_p_mix.device
            or not pre_logits.is_floating_point()
            or not torch.isfinite(pre_logits).all()):
        raise ValueError('pre-head logits must be finite and aligned')
    if (not isinstance(task_for_column, torch.Tensor)
            or task_for_column.shape != log_p_mix.shape[1:]
            or task_for_column.dtype != torch.long
            or task_for_column.device != log_p_mix.device
            or torch.any(task_for_column < 0)):
        raise ValueError('task map must contain one non-negative ID per class')
    if not torch.allclose(
            torch.logsumexp(log_p_mix, dim=1),
            torch.zeros(log_p_mix.size(0), dtype=torch.float64,
                        device=log_p_mix.device),
            rtol=0, atol=1e-10):
        raise ValueError('Mixed probabilities must be normalized')

    output = torch.empty_like(log_p_mix)
    for task in torch.unique(task_for_column, sorted=True):
        columns = task_for_column == task
        output[:, columns] = (
            torch.logsumexp(log_p_mix[:, columns], dim=1, keepdim=True)
            + F.log_softmax(pre_logits[:, columns].to(torch.float64), dim=1)
        )
    if not torch.isfinite(output).all():
        raise ValueError('factorized probabilities are not finite')
    return output
