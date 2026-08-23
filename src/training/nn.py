"""PyTorch tabular MLP shared by training and serving."""

import lightning as L
import torch
import torch.nn as nn


class TabularMLP(L.LightningModule):
    """MLP for directional ``FEATURE_COLS`` rows; symmetry is applied downstream."""

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
        layers: list[nn.Module] = [nn.Linear(tab_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, tab):
        return self.net(tab).squeeze(-1)

    def training_step(self, batch, _batch_idx):
        tab, labels = batch
        loss = nn.functional.binary_cross_entropy_with_logits(self(tab), labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        tab, labels = batch
        logits = self(tab)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return {"probs": probs, "labels": labels}

    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        tab, _labels = batch
        return torch.sigmoid(self(tab))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
        )
