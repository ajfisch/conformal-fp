"""Implementation for FP-controlling CP."""

import functools
import numpy as np
from src import utils


def argmax_set_at_threshold(set_scores, set_sizes, threshold):
    """Calculate output sets at given threshold.

    Decision rule takes argmax_c {|c| : score(c) ≤ threshold}.
    Picking the largest set maximizes the TDP lower bound.

    Args:
        set_scores: [num_examples, max_sets].
        set_sizes: [num_examples, max_sets].
        threshold: maximum non-conformity score a set can have.

    Returns:
       Set indices. [num_examples, 1].
    """
    set_idx = np.argmax(set_sizes * (set_scores <= threshold), axis=-1)
    return set_idx.reshape(-1, 1)


def calibrate_fp(set_scores, set_sizes, set_fps, k):
    """Calibrate a threshold t for finding E[# FPs in C] ≤ alpha.

    Args:
        set_scores: [num_examples, max_sets].
        set_sizes: [num_examples, max_sets].
        set_fps: [num_examples, max_sets].
        k: target expected FP level.

    Returns:
        The threshold.
    """
    # FP upper bound.
    B = set_sizes.max()

    def bound(t):
        """Evaluate bound at given threshold."""
        set_idx = argmax_set_at_threshold(set_scores, set_sizes, t)
        selected_fps = np.take_along_axis(set_fps, set_idx, axis=-1)
        return np.divide(np.sum(selected_fps) + B, len(set_scores) + 1)

    # Generate potential thresholds.
    thresholds = utils.subsample_thresholds(set_scores)

    # Choose best threshold for criteria.
    threshold = utils.max_value_search(thresholds, k, bound)

    return threshold


def calibrate_fp_delta(set_scores, set_sizes, set_fps, k, delta):
    """Calibrate a threshold t for finding P(# FP in C ≤ k) ≥ 1 - delta.

    Args:
        set_scores: [num_examples, max_sets].
        set_sizes: [num_examples, max_sets].
        set_fps: [num_examples, max_sets].
        k: target expected FP level.

    Returns:
        The threshold.
    """

    def bound(t):
        """Evaluate bound at given threshold."""
        set_idx = argmax_set_at_threshold(set_scores, set_sizes, t)
        selected_fps = np.take_along_axis(set_fps, set_idx, axis=-1)
        return np.divide(np.sum(selected_fps > k) + 1, len(set_scores) + 1)

    # Generate potential thresholds.
    thresholds = utils.subsample_thresholds(set_scores)

    # Choose best threshold for criteria.
    threshold = utils.max_value_search(thresholds, delta, bound)

    return threshold


# ------------------------------------------------------------------------------
# Experiment running.
# ------------------------------------------------------------------------------


def run_iter(order, cal_fn, eval_fn):
    """Run of a single draw of (X_1, Z_1), ..., (X_{n+1}, Z_{n+1})"""
    # Split calibration and test.
    full_dataset = utils.g_trial_inputs
    cal_dataset, test_dataset = utils.split_set_dataset(full_dataset, order)

    # Calibrate threshold.
    cal_sizes = np.sum(cal_dataset.item_mask, axis=-1)
    cal_fps = cal_sizes - np.sum(cal_dataset.item_labels, axis=-1)
    threshold = cal_fn(
        set_scores=np.maximum.accumulate(cal_dataset.set_scores, axis=-1),
        set_sizes=cal_sizes,
        set_fps=cal_fps)

    # Evaluate threshold.
    set_idx = argmax_set_at_threshold(
        set_scores=np.maximum.accumulate(test_dataset.set_scores, axis=-1),
        set_sizes=np.sum(test_dataset.item_mask, axis=-1),
        threshold=threshold)

    # Make predictions. Mask encodes set prediction (0/1 if in set or not).
    set_idx = set_idx.reshape(-1, 1, 1)
    preds = np.take_along_axis(test_dataset.item_mask, set_idx, axis=1)
    labels = test_dataset.item_labels.max(axis=1)

    return eval_fn(preds.squeeze(1), labels)


def run_experiment(trials, set_dataset, k, delta=None, threads=0):
    """Resample calibration, test data multiple times and compute metrics."""
    # Always pad set_scores and set preds with -INF and 0s.
    # This corresponds to the empty set option.
    set_dataset = utils.add_empty_set(set_dataset)

    # Init worker function.
    if delta is None:
        cal_fn = functools.partial(calibrate_fp, k=k)
    else:
        cal_fn = functools.partial(calibrate_fp_delta, k=k, delta=delta)
    eval_fn = functools.partial(utils.evaluate_predictions, k=k, delta=delta)
    worker_fn = functools.partial(run_iter, cal_fn=cal_fn, eval_fn=eval_fn)

    return utils.compute_trials(trials, set_dataset, worker_fn, threads)
