"""Check basic statistics about input file."""

import argparse
import numpy as np
from src import utils


def stats(vals):
    median = np.median(vals)
    p16 = np.percentile(vals, 16)
    p84 = np.percentile(vals, 84)
    return "%2.2f (%2.2f - %2.2f)" % (median, p16, p84)


def main(args):
    dataset = utils.load_item_scores(args.input_file, args.max_set_size)

    # Stats.
    num_with_positive = np.mean(dataset.item_labels.sum(axis=1) > 0)

    print("Num examples: %d" % len(dataset.item_mask))
    print("Positives: %s" % stats(dataset.item_labels.sum(axis=1)))
    print("Negatives: %s" % stats((1 - dataset.item_labels).sum(axis=1)))
    print("Answerable: %2.3f" % num_with_positive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--max_set_size", type=int, default=None)
    args = parser.parse_args()
    main(args)
