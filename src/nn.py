"""Script for training DeepSets predictor for FP."""

import argparse
import numpy as np
import os
import tqdm
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl

from src import utils

# Default batch size for nn.
BATCH_SIZE = 64


# ------------------------------------------------------------------------------
#
# DeepSet class.
#
# ------------------------------------------------------------------------------


class DeepSet(pl.LightningModule):
    """DeepSets model that regresses the # FP in an input set of items."""

    def __init__(self, hparams):
        super(DeepSet, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)

        # {x_1, ..., x_k} ---> {h_1, ..., h_k}
        self.encoder = make_ffnn(
            input_dim=1,
            hidden_dim=hparams.set_hidden_dim,
            output_dim=hparams.set_hidden_dim,
            num_layers=hparams.num_encoder_layers,
            dropout=hparams.dropout)

        # If adding in some hand-derived set features (min/max/mean/size)...
        num_set_features = hparams.set_hidden_dim
        if hparams.use_set_features:
            num_set_features += 4

        # {h_1, ..., h_k} ---> z
        self.set_encoder = make_ffnn(
            input_dim=num_set_features,
            hidden_dim=hparams.set_hidden_dim,
            output_dim=hparams.set_hidden_dim,
            num_layers=hparams.num_decoder_layers,
            dropout=hparams.dropout)

        # z --> output
        self.decoder = nn.Sequential(
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.set_hidden_dim, hparams.max_set_size + 1))

    def predict(self, scores, mask):
        """Compute P(FP = k | X) on input batch.

        Args:
            scores: [batch_size, max_set_size].
            mask: 0 = pad. [batch_size, max_set_size].

        Returns:
            logits: [batch_size, max_set_size].
        """
        # [batch_size, max_items, set_hidden_dim]
        scores = scores * mask
        input_encs = self.encoder(scores.unsqueeze(-1))

        # [batch_size, set_hidden_dim]
        input_encs = input_encs * mask.unsqueeze(-1)
        input_encs = input_encs.sum(dim=1)

        # [batch_size, set_hidden_dim + 4]
        if self.hparams.use_set_features:
            sizes = mask.sum(dim=-1)
            mean = torch.div(scores.sum(dim=-1), sizes).view(-1, 1)
            mins = scores.min(dim=-1).values.view(-1, 1)
            maxs = scores.max(dim=-1).values.view(-1, 1)
            feats = [input_encs, mean, mins, maxs, sizes.view(-1, 1)]
            input_encs = torch.cat(feats, dim=1)

        # [batch_size, set_hidden_dim]
        set_enc = self.set_encoder(input_encs).squeeze()

        # [batch_size, max_set_size]
        logits = self.decoder(set_enc).squeeze()
        values = torch.arange(logits.size(-1)).view(1, -1).to(logits.device)
        logits = logits - (values > mask.sum(dim=-1).view(-1, 1)) * utils.INF

        return logits

    def confidence(self, scores, mask, k=None):
        """Compute the confidence for given inputs.

        Args:
            scores: [batch_size, max_set_size].
            mask: 0 = pad. [batch_size, max_set_size].
            k: The target FP level, if given.

        Returns:
            score: E[FP | x] if k is none, else P(FP > k | x).
        """
        logits = self.predict(scores, mask)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        if k is None:
            values = np.arange(logits.shape[-1]).reshape(1, -1)
            score = np.sum(probs * values, axis=-1)
        else:
            score = 1 - np.cumsum(probs, axis=-1)[:, k]

        return score

    def forward(self, scores, mask, targets):
        """Compute loss of input batch."""
        logits = self.predict(scores, mask)
        loss = F.cross_entropy(logits, targets)
        return loss

    def training_step(self, batch, batch_idx=None):
        loss = self.forward(*batch)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log("dev_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=2, min_lr=1e-6, verbose=True)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "dev_loss"}}

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_data", type=str)
        parser.add_argument("--dev_data", type=str)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--max_epochs", type=int, default=20)
        parser.add_argument("--checkpoint_dir", type=str, default="ckpts/debug")
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--max_set_size", type=int, default=utils.MAX_SET_SIZE)
        parser.add_argument("--use_set_features", action="store_true")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--num_encoder_layers", type=int, default=2)
        parser.add_argument("--num_decoder_layers", type=int, default=2)
        parser.add_argument("--set_hidden_dim", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=20)
        parser.add_argument("--seed", type=int, default=42)
        return parser


# ------------------------------------------------------------------------------
#
# Helpers.
#
# ------------------------------------------------------------------------------


def make_ffnn(input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
    """Construct a multilayer feedforward neural network."""
    ffnn = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        ffnn.extend([
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()])
    ffnn.extend([
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU()])
    return nn.Sequential(*ffnn)


def create_dataloader(
    filename,
    max_set_size=None,
    batch_size=1,
    num_workers=0,
    shuffle=False,
):
    item_dataset = utils.load_item_scores(filename, max_set_size)
    set_dataset = utils.create_subsets(item_dataset)
    size = set_dataset.item_scores.shape[-1]
    item_scores = torch.from_numpy(set_dataset.item_scores).view(-1, size)
    item_mask = torch.from_numpy(set_dataset.item_mask).view(-1, size)
    item_labels = torch.from_numpy(set_dataset.item_labels).view(-1, size)
    targets = item_mask.sum(dim=-1) - item_labels.sum(dim=-1)
    dataset = torch.utils.data.TensorDataset(
        item_scores.float(), item_mask.float(), targets.long())
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle)
    return loader


def compute_set_scores(
    item_scores,
    item_mask,
    checkpoints,
    target_k=None,
    batch_size=BATCH_SIZE,
):
    """Compute deep sets model."""
    # Prepare data loader.
    num_examples, num_sets_per_example, max_set_size = item_scores.shape
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(item_scores).view(-1, max_set_size).float(),
        torch.from_numpy(item_mask).view(-1, max_set_size).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Load and transfer to GPU (if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    models = [DeepSet.load_from_checkpoint(ckpt) for ckpt in checkpoints]
    models = [model.to(device) for model in models]
    for model in models:
        model.eval()

    # Evalate examples.
    output_scores = torch.zeros(num_examples * num_sets_per_example)
    index = 0
    with torch.no_grad():
        for scores, mask in tqdm.tqdm(loader, desc="running nn scores"):
            scores, mask = scores.to(device), mask.to(device)
            res = [m.confidence(scores, mask, target_k) for m in models]
            res = sum(res) / len(models)
            for i in range(len(scores)):
                output_scores[index] = res[i].item()
                index += 1

    return output_scores.view(num_examples, num_sets_per_example).numpy()


def train(args):
    """Train model."""
    pl.seed_everything(args.seed)
    model = DeepSet(hparams=args)
    print(model)
    if os.path.exists(args.checkpoint_dir) and os.listdir(args.checkpoint_dir):
        if not args.overwrite:
            raise RuntimeError("Experiment directory is not empty.")
        else:
            shutil.rmtree(args.checkpoint_dir)
    train_loader = create_dataloader(
        filename=args.train_data,
        max_set_size=args.max_set_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)
    dev_loader = create_dataloader(
        filename=args.dev_data,
        max_set_size=args.max_set_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        save_top_k=1,
        verbose=True,
        monitor="dev_loss",
        mode="min")
    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=os.path.join(args.checkpoint_dir, "logs"))
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs)
    trainer.validate(model, dev_loader)
    trainer.fit(model, train_loader, dev_loader)
    model = DeepSet.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.validate(model, dev_loader)
    shutil.copyfile(checkpoint_callback.best_model_path,
                    os.path.join(args.checkpoint_dir, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = DeepSet.add_argparse_args(parser)
    args = parser.parse_args()
    train(args)
