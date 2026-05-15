from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.local_train import positive_class_weight


class FraudMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.2,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_layers:
            layers.append(nn.Linear(current, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class FederatedMLP:
    """PyTorch MLP wrapper exposing NumPy parameters for Flower."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.2,
        batch_norm: bool = False,
        learning_rate: float = 0.001,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = FraudMLP(input_dim, hidden_layers, dropout, batch_norm).to(self.device)
        self.learning_rate = learning_rate

    def get_parameters(self) -> list[np.ndarray]:
        return [value.detach().cpu().numpy().copy() for value in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        keys = list(self.model.state_dict().keys())
        state = OrderedDict(
            (key, torch.tensor(value, dtype=self.model.state_dict()[key].dtype))
            for key, value in zip(keys, parameters)
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 256) -> list[float]:
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        pos_weight = torch.tensor([positive_class_weight(y)], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        losses: list[float] = []
        self.model.train()
        for _ in range(max(1, epochs)):
            running = 0.0
            seen = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * len(xb)
                seen += len(xb)
            losses.append(running / max(seen, 1))
        return losses

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                logits = self.model(xb.to(self.device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
