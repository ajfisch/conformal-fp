"""Script for computing conformal set scores.

Given some set scoring function f: 2^Y --> R, we take in a candidate set C,
and give a measure as to our confidence that C has more than K false positives.

The input format is a jsonlines file where each row is a list of individually
scored candidates, i.e., an estimate of P(Y_i = 1) where Y_i is the candidate.
Each list element should be a tuple of (id, individual score, label).
"""

import argparse
import pickle
import os
import numpy as np

from src import nn
from src import utils


def main(args):
    # Load individual item scores.
    item_dataset = utils.load_item_scores(args.item_scores_file)

    # We might want to calibrate independent scores (i.e., with platt scaling).
    if args.platt_scale:
        print("Platt scaling...")
        cal_dataset = utils.load_item_scores(args.calibration_file)
        item_dataset = utils.platt_scale(cal_dataset, item_dataset)

    # Construct sets (without scores).
    set_dataset = utils.create_subsets(item_dataset)

    # Compute all desired scores.
    os.makedirs(args.output_dir, exist_ok=True)
    for method in args.methods:
        print(f"scoring {method}...")
        # Compute set scores for method.
        if method == "max":
            set_scores = np.max((1 - set_dataset.item_scores)
                                * set_dataset.item_mask, axis=-1)
        elif method == "sum":
            set_scores = np.sum((1 - set_dataset.item_scores)
                                * set_dataset.item_mask, axis=-1)
        elif method == "topk":
            set_scores = np.sum(set_dataset.item_mask, axis=-1)
        elif method == "nn":
            set_scores = nn.compute_set_scores(
                item_scores=set_dataset.item_scores,
                item_mask=set_dataset.item_mask,
                target_k=args.k,
                checkpoints=args.nn_ckpts,
                batch_size=args.nn_batch_size)
        else:
            raise ValueError(f"Unknown method {method}.")

        # Save.
        scored_set_dataset = utils.SetDataset(
            uids=set_dataset.uids,
            set_scores=set_scores,
            item_scores=set_dataset.item_scores,
            item_mask=set_dataset.item_mask,
            item_labels=set_dataset.item_labels)

        suffix = "%03d" % args.k if args.k is not None else "exp"
        if args.platt_scale:
            prefix = "platt"
        else:
            prefix = "raw"
        method_name = prefix + "-" + args.name + "-" + method + "-" + suffix
        output_file = os.path.join(args.output_dir, method_name + ".pkl")
        print(f"Saving to {output_file}.")
        with open(output_file, "wb") as f:
            pickle.dump(scored_set_dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="default",
                        help="Identifier to add to output file name.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results.")
    parser.add_argument("--item_scores_file", type=str, default=None,
                        help="File containing scores for all items.")
    parser.add_argument("--platt_scale", action="store_true")
    parser.add_argument("--calibration_file", type=str, default=None,
                        help="File to use for platt scaling (same format).")
    parser.add_argument("--methods", type=str, nargs="+", default=[],
                        help="Set scoring methods to compute.")
    parser.add_argument("--k", type=int, default=None,
                        help="Target k if estimating P(FP ≤ k).")
    parser.add_argument("--nn_ckpts", type=str, nargs="+", default=None,
                        help="Checkpoint(s) to use for nn scoring.")
    parser.add_argument("--nn_batch_size", type=int, default=64,
                        help="Batch size to use for nn scoring.")

    args = parser.parse_args()
    main(args)
