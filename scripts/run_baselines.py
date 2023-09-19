"""Script for running baseline experiments."""

import argparse
import os
import numpy as np

from src import utils
from src import baselines


def main(args):
    np.random.seed(42)

    # Load individual scores.
    item_dataset = utils.load_item_scores(args.item_scores_file)

    # Load trials.
    args.trials_file = os.path.join(
        args.trials_dir, "trials=%d.json" % args.num_trials)
    print("Loading trials from %s" % args.trials_file)
    trials = utils.load_trials(args.trials_file)

    # Save to output.
    output_file = os.path.join(
        args.output_dir,
        "baselines-results-trials=%d.json" % args.num_trials)
    os.makedirs(args.output_dir, exist_ok=True)

    # Run for all k.
    print(f"{args.item_scores_file}")
    for k in args.k:
        print("=" * 50)
        print(f'K = {k}')
        result = baselines.run_experiment(
            trials=trials,
            item_dataset=item_dataset,
            k=k,
            delta=args.delta,
            threads=args.threads)
        for method, res in result.items():
            print(method)
            utils.print_result(res)
        name = (args.item_scores_file, k, args.delta)
        utils.save_result(result, name, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results.")
    parser.add_argument("--trials_dir", type=str, default=None,
                        help="Directory to look for and/or store trials.")
    parser.add_argument("--num_trials", type=int, default=1000,
                        help="Number of calibration/test permutations.")
    parser.add_argument("--item_scores_file", type=str, default=None,
                        help="File containing individual item scores.")
    parser.add_argument("--k", type=float, default=[5, 15, 25, 35], nargs="+",
                        help="FP tolerance.")
    parser.add_argument("--delta", type=float, default=None,
                        help="Confidence for P(FP ≤ k) ≥ 1 - delta.")
    parser.add_argument("--threads", type=int, default=0,
                        help="Number of processes to use.")
    args = parser.parse_args()
    main(args)
