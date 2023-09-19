# Conformal Prediction Sets with Limited False Positives

Code for [Conformal Prediction Sets with Limited False Positives](https://arxiv.org/abs/2202.07650).

We assume that individual confidence scores have already been obtained from dataset specific models (see paper and the associated model repos cited for details). We provide all raw score files used in the `data` directory, which can be obtained by running:

```
./download_data.sh
```

This also includes pre-computed set scores (but not for target k when using the NN model for controlling P(FP ≤ k) ≥ 1 - delta, see below). To generate (or re-generate) the set scores, run the `compute_set_scores.py` script in the `scripts` folder. As an example, to generate the NN-based scores for the ChEMBL dataset (using the Message Passing Network base individual confidence scores), run:

```
python scripts/compute_set_scores.py \
    --name "mpn" \
    --output_dir "data/chembl/scores" \
    --methods "nn" \
    --item_scores_file "data/chembl/test.jsonl" \
    --calibration_file "data/chembl/dev.jsonl" \
    --nn_ckpts ckpts/chembl/*/model.pt
```

The NN model can be trained by running (again, for the ChEMBL example):

```
python src/nn.py \
    --seed <SEED> \
    --checkpoint_dir "ckpts/chembl/<SEED>" \
    --train_data "data/chembl/train_dev.jsonl" \
    --dev_data "data/chembl/dev.jsonl" \
    --use_set_features
```
Seeds used in the paper's experiments were:

```
42 1013 3533 4793 296 1947 2174 7776 5309 495 1216 7753 4110 1599 9102
```

Pre-trained NN models can be downloaded via:
```
./dowload_models.sh
```

Finally, to perform conformal prediction & evaluation, run:

```
python scripts/run_conformal.py \
       --output_dir "results/chembl" \
       --trials_dir "data/chembl" \
       --set_scores_file "data/chembl/scores/raw-mpn-nn-exp.pkl" \
       --threads 30
```

Note that `run_baselines.py` runs the baseline methods, e.g.:

```
python scripts/run_baselines.py \
       --output_dir "results/chembl" \
       --trials_dir "data/chembl" \
       --item_scores_file "data/chembl/test.jsonl" \
       --threads 30
```

For control in probability, see the `delta` option. Also note that some scores (i.e., NN) are specific to a target k, so the `compute_set_scores.py` script should be re-run with the desired k specified via the `--k` argument.

Results are saved as a dict mapping from key (e.g., tuple of `(args.set_scores_file, k, args.delta)` for conformal methods) to `Result` object (see `src/utils.py`).