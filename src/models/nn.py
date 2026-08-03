"""PyTorch tabular + bio-embedding MLP shared by training and serving.

Defined here (not inside the training notebook) so the MLflow-logged
`nn_best` artifact unpickles outside the notebook kernel: torch.save
records the class by module path, and the BentoML service must be able to
import `src.models.nn.TabularBioMLP` at load time.
"""

from typing import override

import lightning as L
import torch
import torch.nn as nn


class TabularBioMLP(L.LightningModule):
    """Tabular + bio-embedding MLP over the canonical match row.

    Tabular branch encodes the balanced FEATURE_COLS row (player + opponent
    rolling stats, differentials, context). A shared bio branch encodes the
    canonical player's and opponent's bio embeddings. The fusion head gets
    concat(tabular, bio_p, bio_o, bio_p - bio_o), where the difference term
    is the symmetric matchup signal. Order-invariance comes from the
    canonical ATP-id-ordered row: swapping player/opponent flips the diff
    term and the target, so predictions stay consistent.
    """

    def __init__(self, tab_dim, bio_dim, hidden_dim=64, dropout=0.0, lr=1e-3):  # noqa: ARG002 — lr is persisted via save_hyperparameters()
        super().__init__()
        self.save_hyperparameters()
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bio_mlp = nn.Sequential(
            nn.Linear(bio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    @override
    def forward(self, tab, bio_p, bio_o):
        t = self.tab_mlp(tab)
        bp = self.bio_mlp(bio_p)
        bo = self.bio_mlp(bio_o)
        return self.head(torch.cat([t, bp, bo, bp - bo], dim=1)).squeeze(-1)

    @override
    def training_step(self, batch, _batch_idx):
        tab, bio_p, bio_o, labels = batch
        loss = nn.functional.binary_cross_entropy_with_logits(self(tab, bio_p, bio_o), labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @override
    def validation_step(self, batch, _batch_idx):
        tab, bio_p, bio_o, labels = batch
        logits = self(tab, bio_p, bio_o)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return {"probs": probs, "labels": labels}

    @override
    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        tab, bio_p, bio_o, _labels = batch
        return torch.sigmoid(self(tab, bio_p, bio_o))

    @override
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
