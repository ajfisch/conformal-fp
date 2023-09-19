"""Script for running main experiments."""

import argparse
import pickle
import os
import numpy as np

from src import utils
from src import conformal


def main(args):
    np.random.seed(42)

    # Load score file.
    with open(args.set_scores_file, "rb") as f:
        set_dataset = pickle.load(f)

    # Load trials from saved file, if it exists. If it doesn't exist, make it.
    # Trials are just permutations of example indices into calibration/test.
    args.trials_file = os.path.join(
        args.trials_dir, "trials=%d.json" % args.num_trials)
    os.makedirs(os.path.dirname(args.trials_file), exist_ok=True)
    if os.path.exists(args.trials_file) and not args.overwrite_trials:
        print("Loading trials from %s" % args.trials_file)
        trials = utils.load_trials(args.trials_file)
    else:
        n_examples = len(set_dataset.item_scores)
        trials = utils.create_trials(n_examples, args.ratio, args.num_trials)
        print("Writing trials to %s" % args.trials_file)
        utils.save_trials(trials, args.trials_file)

    # Save to output.
    output_file = os.path.join(
        args.output_dir,
        "conformal-results-trials=%d.json" % args.num_trials)
    os.makedirs(args.output_dir, exist_ok=True)

    # Run for all k.
    print(f"{args.set_scores_file}")
    for k in args.k:
        print("=" * 50)
        print(f'K = {k}')
        result = conformal.run_experiment(
            trials=trials,
            set_dataset=set_dataset,
            k=k,
            delta=args.delta,
            threads=args.threads)
        utils.print_result(result)
        name = (args.set_scores_file, k, args.delta)
        utils.save_result(result, name, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results.")

    parser.add_argument("--trials_dir", type=str, default=None,
                        help="Directory to look for and/or store trials.")
    parser.add_argument("--overwrite_trials", action="store_true",
                        help="Overwrite the existing trials file.")
    parser.add_argument("--num_trials", type=int, default=1000,
                        help="Number of calibration/test permutations.")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Ratio of calibration to test examples to use.")

    parser.add_argument("--set_scores_file", type=str, default=None,
                        help="File containing conformal set scores.")

    parser.add_argument("--k", type=float, default=[5, 15, 25, 35], nargs="+",
                        help="FP tolerance.")
    parser.add_argument("--delta", type=float, default=None,
                        help="Confidence for P(FP ≤ k) ≥ 1 - delta.")
    parser.add_argument("--threads", type=int, default=0,
                        help="Number of processes to use.")
    args = parser.parse_args()
    main(args)
