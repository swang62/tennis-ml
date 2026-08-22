"""PyTorch tabular MLP shared by training and serving.

Defined here (not inside the training notebook) so the MLflow-logged
`nn_best` artifact unpickles outside the notebook kernel: torch.save
records the class by module path.
"""

from typing import override

import lightning as L
import torch
import torch.nn as nn


class TabularMLP(L.LightningModule):
    """Tabular MLP over a directional player-perspective row.

    Encodes the balanced FEATURE_COLS row (supplied player + opponent rolling
    stats, differentials, context) through a single MLP head. The model is
    trained on directional rows and is not intrinsically antisymmetric:
    reversing the input does not negate the logit. Antisymmetrization happens
    downstream in the stacker, which scores both directions and projects to a
    symmetric probability.
    """

    def __init__(
        self,
        tab_dim,
        hidden_dim=64,
        n_layers=1,
        dropout=0.0,
        lr=1e-3,  # noqa: ARG002 — persisted via save_hyperparameters()
        weight_decay=0.0,  # noqa: ARG002 — persisted via save_hyperparameters()
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tab_mlp = self._mlp(tab_dim, hidden_dim, n_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    @staticmethod
    def _mlp(in_dim, hidden_dim, n_layers, dropout):
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        return nn.Sequential(*layers)

    @override
    def forward(self, tab):
        return self.head(self.tab_mlp(tab)).squeeze(-1)

    @override
    def training_step(self, batch, _batch_idx):
        tab, labels = batch
        loss = nn.functional.binary_cross_entropy_with_logits(self(tab), labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @override
    def validation_step(self, batch, _batch_idx):
        tab, labels = batch
        logits = self(tab)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return {"probs": probs, "labels": labels}

    @override
    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        tab, _labels = batch
        return torch.sigmoid(self(tab))

    @override
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
        )
