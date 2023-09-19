"""Baselines for FP control."""

import functools
import numpy as np
import scipy.stats

from src import utils


def predict_top_k(cal_dataset, test_dataset, k, delta=None):
    """Compute predictions using top-K calibration.

    Args:
        cal_dataset: Instance of ItemDataset for calibration.
        test_dataset: Instance of ItemDataset for testing.
        k: Target FP level.
        delta: Target confidence level.

    Returns:
        Predicted sets (binary 0/1 if item is included in output).
    """

    def rank(scores):
        return scipy.stats.rankdata(scores, axis=-1, method="ordinal")

    # Flip scores so that lower is better.
    cal_scores = rank(1 - cal_dataset.item_scores)
    test_scores = rank(1 - test_dataset.item_scores)

    def eval_fn(t):
        in_set = cal_scores <= t
        fps = np.sum((1 - cal_dataset.item_labels) * in_set, axis=-1)
        if delta is not None:
            fps = fps > k
        return np.divide(np.sum(fps), len(in_set) + utils.EPS)

    thresholds = np.arange(cal_dataset.item_scores.shape[1] + 1)
    bound = delta if delta is not None else k
    selected_k = utils.max_value_search(thresholds, bound, eval_fn)

    return test_scores <= selected_k


def predict_inner_sets(cal_dataset, test_dataset, k, delta=None):
    """Compute predictions using inner sets calibration.

    Args:
        cal_dataset: Instance of ItemDataset for calibration.
        test_dataset: Instance of ItemDataset for testing.
        k: Target FP level.
        delta: Target confidence level.

    Returns:
        Predicted sets (binary 0/1 if item is included in output).
    """
    cal_scores = cal_dataset.item_scores
    test_scores = test_dataset.item_scores

    # Mask out padding and positive labels.
    cal_scores[cal_dataset.item_mask] = -utils.INF
    cal_scores[cal_dataset.item_labels == 1] = -utils.INF

    # Collect worst-case statistics for "0" labels.
    # Scores are conformal, so higher scores are worse.
    worst_case = np.max(cal_scores, axis=1)

    # Add +INF for finite sample.
    worst_case = np.pad(worst_case, (0, 1), constant_values=utils.INF)

    # Compute 1 - * quantile of worst-case scores. With probability ≥ 1 - *
    # 0-label scores will be ≤ than this quantile. This leaves probability at
    # at most * for 0 scores above.
    if delta is not None:
        bound = 1 - delta
    else:
        B = test_dataset.item_scores.shape[-1]
        bound = 1 - k / B
    threshold = np.quantile(worst_case, bound, interpolation="higher")

    return test_scores > threshold


def predict_outer_sets(cal_dataset, test_dataset, epsilon=0.1):
    """Compute predictions using outer sets calibration.

    Args:
        cal_dataset: Instance of ItemDataset for calibration.
        test_dataset: Instance of ItemDataset for testing.
        epsilon: tolerance for outer sets miscoverage.

    Returns:
        Predicted sets (binary 0/1 if item is included in output).
    """
    # Flip scores so that lower is better.
    cal_scores = 1 - cal_dataset.item_scores
    test_scores = 1 - test_dataset.item_scores

    # Mask out padding and negative labels.
    cal_scores[cal_dataset.item_mask] = -utils.INF
    cal_scores[cal_dataset.item_labels == 0] = -utils.INF

    # Collect worst-case statistics for "1" labels.
    # Scores are conformal, so higher scores are worse.
    worst_case = np.max(cal_scores, axis=1)

    # Add +INF for finite sample.
    worst_case = np.pad(worst_case, (0, 1), constant_values=utils.INF)

    # Compute 1 - epsilon quantile of worst-case scores.
    threshold = np.quantile(worst_case, 1 - epsilon, interpolation="higher")

    return test_scores <= threshold


# ------------------------------------------------------------------------------
# Experiment running.
# ------------------------------------------------------------------------------


def run_iter(order, pred_fn, eval_fn):
    """Run of a single draw of (X_1, Z_1), ..., (X_{n+1}, Z_{n+1})"""
    # Split calibration and test.
    full_dataset = utils.g_trial_inputs
    cal_dataset, test_dataset = utils.split_item_dataset(full_dataset, order)

    # Make prediction.
    preds = pred_fn(cal_dataset, test_dataset)
    labels = test_dataset.item_labels

    return eval_fn(preds, labels)


def run_experiment(trials, item_dataset, k, delta=None, threads=0):
    """Resample calibration, test data multiple times and compute metrics."""
    # Init worker function.
    methods = {
        'top-k': functools.partial(predict_top_k, k=k, delta=delta),
        'inner': functools.partial(predict_inner_sets, k=k, delta=delta),
        'outer': functools.partial(predict_outer_sets, epsilon=0.1),
    }
    eval_fn = functools.partial(utils.evaluate_predictions, k=k, delta=delta)
    results = {}
    for method, pred_fn in methods.items():
        worker_fn = functools.partial(
            run_iter, pred_fn=pred_fn, eval_fn=eval_fn)
        result = utils.compute_trials(trials, item_dataset, worker_fn, threads)
        results[method] = result
    return results
