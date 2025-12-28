"""
Transformer encoder model for financial time-series prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TransformerConfig:
    input_size: int
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerAlpha(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_size, cfg.d_model)
        self.pos_encoder = PositionalEncoding(cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        enc = self.encoder(x)
        last = enc[:, -1, :]
        preds = self.head(last).squeeze(-1)
        return preds


def train_transformer(
    model: TransformerAlpha,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TransformerAlpha:
    X_train, y_train = train_data
    X_val, y_val = val_data

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            _ = loss_fn(val_preds, y_val_t)

    return model


class TransformerWrapper:
    """
    Wrapper to make TransformerAlpha compatible with EnsembleAlpha (numpy input/output).
    """
    def __init__(self, model: TransformerAlpha, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: [N, T, F] -> numpy
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().to(self.device)
            preds = self.model(X_t)
            return preds.cpu().numpy()


