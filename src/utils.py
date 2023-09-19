"""Utility library."""

import collections
import functools
import json
import multiprocessing
import os
import pickle
import prettytable
import subprocess
import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression

INF = 1e8

EPS = 1e-8

MAX_THRESHOLDS = 50000

MAX_SET_SIZE = 100

MIN_SAMPLE_SIZE = 25

STRATAS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Fields:
#    calibration: indices for calibration set.
#    test: indices for test set.
SplitIndex = collections.namedtuple(
    "SplitIndex",
    ["calibration", "test"])

# Fields:
#    tpr: true positive rate.
#    fp: average number of false positives per set.
#    fp_k: fraction of sets with FP ≤ k.
#    wcss_fp: worst-case size-stratified version of fp.
#    wcss_fp_k: worst-case size-stratified version of fp_k.
Result = collections.namedtuple(
    "Result",
    ["tpr", "fp", "fp_k", "wcss_fp", "wcss_fp_k"])

# Fields:
#     uids: unique identifier for each example. [num_examples].
#     item_scores: item score matrix. [num_examples, max_items].
#     item_labels: label matrix (0 = FP). [num_examples, max_items].
#     item_mask: padding matrix (0 = pad). [num_examples, max_items].
ItemDataset = collections.namedtuple(
    "ItemDataset",
    ["uids", "item_scores", "item_labels", "item_mask"])

# Fields:
#     set_scores: set score matrix. [num_examples, max_sets].
#     item_scores: item score matrix. [num_examples, max_sets, max_items].
#     item_labels: label matrix (0 = FP). [num_examples, max_sets, max_items].
#     item_mask: padding matrix (0 = pad). [num_examples, max_sets, max_items].
SetDataset = collections.namedtuple(
    "SetDataset",
    ["uids", "set_scores", "item_scores", "item_labels", "item_mask"])


# ------------------------------------------------------------------------------
#
# I/O helpers.
#
# ------------------------------------------------------------------------------


def count_lines(filename):
    wc = subprocess.check_output(["wc", "-l", filename], encoding="utf8")
    return int(wc.split()[0])


def load_item_scores(filename, max_set_size=MAX_SET_SIZE):
    """Load independent item scores from jsonlines file.

    Only the top `max_set_size` are taken.

    Args:
        filename: name of .jsonl file.
        max_set_size: largest set size to consider

    Returns:
        An ItemDataset.
        item_scores: item score matrix. [num_examples, max_set_size].
        item_labels: label matrix (0 = FP). [num_examples, max_set_size].
        item_mask: padding matrix (0 = pad). [num_examples, max_set_size].
    """
    data = []
    num_lines = count_lines(filename)
    with tqdm.tqdm(total=num_lines, desc=f"reading {filename}") as pbar:
        with open(filename, "r") as f:
            for line in f:
                items = []
                for uid, score, label in json.loads(line):
                    if isinstance(score, list):
                        score = np.exp(np.sum(np.log(np.clip(score, EPS, 1))))
                    items.append((uid, score, label))
                data.append(sorted(items, key=lambda x: -x[1]))
                pbar.update()

    # Convert to numpy matrix.
    num_examples = len(data)
    max_set_size = max_set_size or max([len(ex) for ex in data])
    uids = []
    item_scores = np.zeros((num_examples, max_set_size))
    item_labels = np.zeros((num_examples, max_set_size))
    item_mask = np.zeros((num_examples, max_set_size), dtype=int)
    for i, example in enumerate(data):
        example = example[:max_set_size]
        example_uids = []
        for j, (uid, score, label) in enumerate(example):
            example_uids.append(uid)
            item_scores[i, j] = score
            item_labels[i, j] = label
            item_mask[i, j] = 1
        uids.append(example_uids)

    return ItemDataset(uids, item_scores, item_labels, item_mask)


def create_subsets(item_dataset):
    """Create candidate sets from an ItemDataset.

    Sets are created greedily by taking top-ranked individual candidates first.
    Given [A, B, C, ...], constructs [A], [A, B], [A, B, C], [...].

    Args:
        item_dataset: instance of ItemDataset.

    Returns:
        A SetDataset.
    """
    uids = item_dataset.uids
    scores = item_dataset.item_scores
    if item_dataset.item_mask is not None:
        scores -= (1 - item_dataset.item_mask) * INF

    num_examples = len(scores)
    max_set_size = item_dataset.item_scores.shape[1]
    item_scores = np.zeros((num_examples, max_set_size, max_set_size))
    item_labels = np.zeros((num_examples, max_set_size, max_set_size))
    item_mask = np.zeros((num_examples, max_set_size, max_set_size), dtype=int)
    for i in tqdm.tqdm(range(num_examples), desc='creating sets'):
        for j in range(0, max_set_size):
            item_scores[i, j, :j + 1] = item_dataset.item_scores[i, :j + 1]
            item_mask[i, j, :j + 1] = item_dataset.item_mask[i, :j + 1]
            item_labels[i, j, :j + 1] = item_dataset.item_labels[i, :j + 1]

    return SetDataset(uids, None, item_scores, item_labels, item_mask)


def split(cls, dataset, idx):
    """Split an object by given index."""
    split_values = {}
    for attr in cls._fields:
        value = getattr(dataset, attr)
        if value is not None:
            if isinstance(value, np.ndarray):
                value = value[idx]
            else:
                value = [value[i] for i in idx]
        split_values[attr] = value
    return cls(**split_values)


def split_set_dataset(set_dataset, split_index):
    """Return split SetDataset given a SplitIndex."""
    _split = functools.partial(split, SetDataset, set_dataset)
    return _split(split_index.calibration), _split(split_index.test)


def split_item_dataset(item_dataset, split_index):
    """Return split ItemDataset given a SplitIndex."""
    _split = functools.partial(split, ItemDataset, item_dataset)
    return _split(split_index.calibration), _split(split_index.test)


# ------------------------------------------------------------------------------
#
# Calibration helpers.
#
# ------------------------------------------------------------------------------


def platt_scale(cal_dataset, pred_dataset):
    """Apply platt scaling to scores.

    Args:
        cal_dataset: instance of ItemDataset.
        pred_dataset: instance of ItemDataset.

    Returns:
        A new ItemDataset with scaled item_scores.
    """
    _, cal_scores, cal_labels, cal_mask = cal_dataset
    uids, pred_scores, pred_labels, pred_mask = pred_dataset
    if cal_mask is not None:
        cal_labels *= cal_mask
        cal_scores -= (1 - cal_mask) * INF
    if pred_mask is not None:
        pred_scores -= (1 - pred_mask) * INF

    # Flatten data.
    train_X = cal_scores.reshape(-1, 1)
    train_y = cal_labels.reshape(-1)
    predict_X = pred_scores.reshape(-1, 1)

    # Train LR.
    clf = LogisticRegression(random_state=0, C=10, class_weight="balanced")
    clf.fit(train_X, train_y)

    # Predict probabilities.
    pred_proba = clf.predict_proba(predict_X)[:, 1]
    pred_proba = pred_proba.reshape(pred_scores.shape)

    return ItemDataset(uids, pred_proba, pred_labels, pred_mask)


def add_empty_set(set_dataset):
    """Add an empty set to a input SetDataset."""
    set_scores = np.pad(set_dataset.set_scores, [(0, 0), (1, 0)])
    set_scores[:, 0] = -INF
    item_scores = np.pad(set_dataset.item_scores, [(0, 0), (1, 0), (0, 0)])
    item_labels = np.pad(set_dataset.item_labels, [(0, 0), (1, 0), (0, 0)])
    item_mask = np.pad(set_dataset.item_mask, [(0, 0), (1, 0), (0, 0)])
    set_dataset = SetDataset(
        set_dataset.uids, set_scores, item_scores, item_labels, item_mask)
    return set_dataset


def subsample_thresholds(values, max_thresholds=MAX_THRESHOLDS):
    values = values.reshape(-1)
    if len(values) <= max_thresholds:
        thresholds = np.unique(values)
    else:
        quantiles = np.linspace(0, 1, max_thresholds)
        thresholds = np.unique(np.quantile(values, quantiles))
    thresholds = np.concatenate([[-INF], thresholds, [INF]])
    return thresholds


def max_value_search(values, bound, func=lambda x: x):
    """Search for sup{x: f(x) ≤ t} for a function f(x).

    Assumes monotonic decreasing f(x).

    Args:
        values: valid x values to search.
        bound: the value t that f(x) must exceed.
        func: the evaluated function f.

    Returns:
        bound: sup{x: f(x) ≤ t}.
    """
    low = 0
    high = len(values) - 1
    max_value = -INF
    while low <= high:
        mid = (high + low) // 2
        value = values[mid]
        result = func(value)
        if result <= bound:
            max_value = max(max_value, value)
            low = mid + 1
        else:
            high = mid - 1
    return max_value


# ------------------------------------------------------------------------------
#
# Evaluation helpers
#
# ------------------------------------------------------------------------------


def evaluate_wcss(false_positives, set_sizes, k, delta, stratas=STRATAS):
    """Evaluate worst-case size-stratified metric."""
    # Measure FP metrics per group.
    group_fps = []
    group_fp_ks = []

    for i in range(len(stratas)):
        start = stratas[i]
        if i < len(stratas) - 1:
            end = stratas[i + 1]
        else:
            end = INF

        # Take all sets with size in [start, end].
        mask = (set_sizes >= start) * (set_sizes < end)

        # Only count if there are enough members in this group.
        group_fp = group_fp_k = -1
        if np.sum(mask) >= MIN_SAMPLE_SIZE:
            group_fp = np.mean(false_positives[mask])
            if delta is not None:
                group_fp_k = max(np.mean(false_positives[mask] > k) - delta, 0)
        group_fps.append(group_fp)
        group_fp_ks.append(group_fp_k)

    # Return all.
    return np.array(group_fps), np.array(group_fp_ks)


def evaluate_predictions(predictions, labels, k, delta=None):
    """Compute metrics on set predictions.

    Args:
        predictions: binary encoding of set. [num_examples, max_items].
        labels: 1 if labels[i, j] is a TP. [num_examples, max_items].
        k: target FP level.
        delta: target probability.

    Returns:
        Instance of Result.
    """
    set_sizes = predictions.sum(axis=-1)
    true_positives = np.sum(predictions * labels, axis=-1)
    false_positives = set_sizes - true_positives

    # Count maximum positives.
    total_positives = labels.sum(axis=-1)

    # Convert to rates.
    fp = np.mean(false_positives)
    fp_k = np.mean(false_positives <= k)
    tpr = np.mean(true_positives / (total_positives + EPS))

    # Compute worst-case size-stratified FDR constraint violations.
    wcss_fp, wcss_fp_k = evaluate_wcss(false_positives, set_sizes, k, delta)

    return Result(tpr, fp, fp_k, wcss_fp, wcss_fp_k)


# ------------------------------------------------------------------------------
#
# Experimentation helpers.
#
# ------------------------------------------------------------------------------


def print_result(result):
    """Format results for printing."""
    t = prettytable.PrettyTable()
    t.field_names = ["Metric", "Mean", "p16", "p84"]
    t.add_row(["TPR"] + [f"{x:.3f}" for x in result.tpr])
    t.add_row(["E[FP]"] + [f"{x:.3f}" for x in result.fp])
    t.add_row(["P(FP ≤ k)"] + [f"{x:.3f}" for x in result.fp_k])
    t.add_row(["WCSS E[FP]"] + [f"{x:.3f}" for x in result.wcss_fp])
    t.add_row(["WCSS P(FP ≤ k)"] + [f"{x:.3f}" for x in result.wcss_fp_k])
    print(t)


def create_trials(num_examples, ratio=0.8, iterations=1000):
    """Sample calibration and test permutations."""
    trials = []
    num_calibration = int(ratio * num_examples)
    for _ in range(iterations):
        indices = np.random.permutation(num_examples)
        calibration = indices[:num_calibration]
        test = indices[num_calibration:]
        trials.append(SplitIndex(calibration, test))
    return trials


def save_trials(trials, filename):
    """Save index orders to disk."""
    with open(filename, "w") as f:
        for trial in trials:
            calibration = trial.calibration.tolist()
            test = trial.test.tolist()
            f.write(json.dumps([calibration, test]) + "\n")


def load_trials(filename):
    """Load index orders from disk."""
    trials = []
    with open(filename, "r") as f:
        for line in f:
            trials.append(SplitIndex(*json.loads(line)))
    return trials


def save_result(result, key, filename):
    """Save result to file."""
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}
    results[key] = result

    with open(filename, "wb") as f:
        pickle.dump(results, f)


def init_trial_globals(shared_inputs):
    """Multiprocessing helper to initialize shared globals."""
    global g_trial_inputs
    g_trial_inputs = shared_inputs


def compute_trials(trials, shared_inputs, worker_fn, threads=0):
    """Helper to compute marginalized trial results with multiprocessing."""
    # Init worker pool.
    if threads > 1:
        workers = multiprocessing.Pool(
            threads,
            initializer=init_trial_globals,
            initargs=(shared_inputs,))
        map_fn = workers.imap_unordered
    else:
        init_trial_globals(shared_inputs)
        map_fn = map

    # Map all results.
    results = []
    with tqdm.tqdm(total=len(trials), desc="running trials") as pbar:
        for result in map_fn(worker_fn, trials):
            results.append(result)
            pbar.update()

    if threads > 1:
        workers.close()
        workers.terminate()

    def stats(values):
        values = np.array(values)
        if values.size == 0:
            return (np.nan, np.nan, np.nan)
        mean = np.mean(values)
        p16 = np.percentile(values, 16)
        p84 = np.percentile(values, 84)
        return mean, p16, p84

    def wcss_stats(values):
        avgs = []
        for i in range(values.shape[-1]):
            mask = values[:, i] != -1
            if mask.sum() >= MIN_SAMPLE_SIZE:
                avgs.append(np.mean(values[:, i][mask]))
            else:
                avgs.append(-1)
        wcss_i = np.argmax(avgs)
        values_i = values[:, wcss_i]
        mask = values_i != -1
        return stats(values_i[mask])

    # Convert to tuples of (median, p16, p84).
    tpr = stats([res.tpr for res in results])
    fp = stats([res.fp for res in results])
    fp_k = stats([res.fp_k for res in results])

    # Compute WCSS metrics.
    wcss_fp = wcss_stats(np.stack([res.wcss_fp for res in results]))
    wcss_fp_k = wcss_stats(np.stack([res.wcss_fp_k for res in results]))

    return Result(tpr, fp, fp_k, wcss_fp, wcss_fp_k)
