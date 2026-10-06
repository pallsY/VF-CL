"""Strict, deterministic reconstruction of common formal CL metrics."""
from dataclasses import dataclass
import math
from numbers import Real
import statistics


FORMULA_VERSION = 'final-minus-diagonal-v1'


@dataclass(frozen=True)
class FormalMetrics:
    aa_final: float
    bwt: float
    taskil_final: float
    aa_trajectory: tuple
    class_final: tuple
    taskil_final_by_task: tuple


def _task_ids(expected_task_ids):
    if not isinstance(expected_task_ids, (list, tuple)):
        raise ValueError('task identities must be an ordered sequence')
    task_ids = tuple(expected_task_ids)
    if len(task_ids) < 2:
        raise ValueError('at least two task identities are required')
    if any(isinstance(task_id, bool) or not isinstance(task_id, int)
           or task_id < 0 for task_id in task_ids):
        raise ValueError('task identities must be non-negative integers')
    if len(set(task_ids)) != len(task_ids):
        raise ValueError('task identities must be unique')
    return task_ids


def _matrix(rows, task_ids, name):
    if not isinstance(rows, (list, tuple)):
        raise ValueError(f'{name} trajectory must be an ordered sequence')
    expected_steps = tuple(
        f'event_{index}_CIL' for index in range(len(task_ids))
    )
    if len(rows) != len(expected_steps):
        raise ValueError(f'{name} trajectory has an invalid number of rows')

    matrix = []
    for index, (row, step) in enumerate(zip(rows, expected_steps)):
        if not isinstance(row, dict) or set(row) != {'step', 'values'}:
            raise ValueError(f'{name} row schema is invalid')
        if row['step'] != step:
            raise ValueError(f'{name} trajectory does not match frozen steps')
        values = row['values']
        expected_ids = task_ids[:index + 1]
        expected_keys = {f'task_{task_id}' for task_id in expected_ids}
        if not isinstance(values, dict) or set(values) != expected_keys:
            raise ValueError(f'{name} trajectory task identity mismatch')

        ordered = []
        for task_id in expected_ids:
            value = values[f'task_{task_id}']
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value)):
                raise ValueError(f'{name} trajectory contains an invalid value')
            ordered.append(float(value))
        matrix.append(tuple(ordered))
    return tuple(matrix)


def reconstruct_metrics(class_rows, taskil_rows, expected_task_ids):
    """Validate frozen CIL trajectories and reconstruct their common metrics."""
    task_ids = _task_ids(expected_task_ids)
    class_matrix = _matrix(class_rows, task_ids, 'Class-IL')
    taskil_matrix = _matrix(taskil_rows, task_ids, 'Task-IL')
    final_class = class_matrix[-1]
    final_taskil = taskil_matrix[-1]
    diagonal = tuple(class_matrix[index][index]
                     for index in range(len(task_ids)))
    bwt = statistics.fmean(
        final_class[index] - diagonal[index]
        for index in range(len(task_ids) - 1)
    )
    return FormalMetrics(
        aa_final=statistics.fmean(final_class),
        bwt=bwt,
        taskil_final=statistics.fmean(final_taskil),
        aa_trajectory=tuple(statistics.fmean(row) for row in class_matrix),
        class_final=final_class,
        taskil_final_by_task=final_taskil,
    )
