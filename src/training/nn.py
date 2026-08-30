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
        dropout=0.0,
        lr=1e-3,  # noqa: ARG002 — persisted via save_hyperparameters()
        weight_decay=0.0,  # noqa: ARG002 — persisted via save_hyperparameters()
    ):
        super().__init__()
        self.save_hyperparameters()
        layers: list[nn.Module] = [
            nn.Linear(tab_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, tab):
        return self.net(tab).squeeze(-1)

    def training_step(self, batch, _batch_idx):
        if len(batch) == 3:
            tab, labels, weights = batch
        else:
            tab, labels = batch
            weights = None
        logits = self(tab)
        if weights is None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        else:
            # Normalized weighted BCE for training only; validation/predict stay unweighted.
            bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = (bce * weights).mean()
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


class SymmetricGRU(L.LightningModule):
    """Compact symmetric GRU over per-side histories with one shared scorer."""

    def __init__(
        self,
        hist_dim: int,
        context_dim: int = 7,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        lr: float = 1e-3,  # noqa: ARG002 — persisted via save_hyperparameters()
        weight_decay: float = 0.0,  # noqa: ARG002 — persisted via save_hyperparameters()
    ):
        super().__init__()
        self.save_hyperparameters()
        self.gru = nn.GRU(hist_dim, hidden_dim, batch_first=True)
        self.zero_emb = nn.Parameter(torch.zeros(hidden_dim))
        layers: list[nn.Module] = [
            nn.Linear(hidden_dim * 2 + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ]
        self.head = nn.Sequential(*layers)

    def _lengths(self, valid: torch.Tensor, seq_len: int) -> torch.Tensor:
        lengths = valid if valid.dim() == 1 else valid.sum(dim=1)
        return lengths.long().clamp(max=seq_len)

    def _encode(self, hist: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch = hist.size(0)
        seq_len = hist.size(1)
        if seq_len == 0:
            return self.zero_emb.unsqueeze(0).expand(batch, -1)
        # Left-align the right-justified valid window so leading padding is dropped.
        cols = torch.arange(seq_len, device=hist.device).unsqueeze(0)
        src = (seq_len - lengths.unsqueeze(1) + cols).clamp(min=0, max=seq_len - 1)
        aligned = hist.gather(1, src.unsqueeze(-1).expand(*src.shape, hist.size(2)))
        # Run the full (left-aligned) sequence and read the hidden state at the
        # last valid timestep. This is numerically identical to packing (which
        # stops after the last valid step) but exports cleanly to ONNX Runtime.
        out, _ = self.gru(aligned)
        last = (
            (lengths.clamp(min=1) - 1)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(batch, 1, self.gru.hidden_size)
        )
        emb = out.gather(1, last).squeeze(1)
        # Trailing padding (after the last valid step) never affects `last`;
        # empties fall back to the zero-history embedding (branchless so the
        # graph stays exportable to ONNX Runtime).
        empty = lengths == 0
        emb = torch.where(empty.unsqueeze(1), self.zero_emb.unsqueeze(0), emb)
        return emb

    def _score(
        self, player_emb: torch.Tensor, opponent_emb: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return self.head(torch.cat([player_emb, opponent_emb, context], dim=-1)).squeeze(-1)

    def forward(
        self,
        player_hist: torch.Tensor,
        opponent_hist: torch.Tensor,
        player_valid: torch.Tensor,
        opponent_valid: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        p_emb = self._encode(player_hist, self._lengths(player_valid, player_hist.size(1)))
        o_emb = self._encode(opponent_hist, self._lengths(opponent_valid, opponent_hist.size(1)))
        return self._score(p_emb, o_emb, context) - self._score(o_emb, p_emb, context)

    def training_step(self, batch, _batch_idx):
        if len(batch) == 7:
            ph, oh, pv, ov, ctx, labels, weights = batch
        else:
            ph, oh, pv, ov, ctx, labels = batch
            weights = None
        logits = self(ph, oh, pv, ov, ctx)
        if weights is None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        else:
            # Normalized weighted BCE for training only; validation/predict stay unweighted.
            bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = (bce * weights).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        ph, oh, pv, ov, ctx, labels = batch[:6]
        logits = self(ph, oh, pv, ov, ctx)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"probs": probs, "labels": labels}

    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        ph, oh, pv, ov, ctx = batch[:5]
        return torch.sigmoid(self(ph, oh, pv, ov, ctx))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
        )
