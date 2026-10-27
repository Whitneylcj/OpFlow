"""Small Torch MLP utilities used for baseline experiments."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


class StandardScaler:
    """Lightweight feature scaler."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve the Torch device from config."""
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchFeedForwardRegressor:
    """Two-layer Torch MLP regressor trained with Adam."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 256,
        l2: float = 1e-5,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.device = resolve_device(device)
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ).to(self.device)

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 30,
    ) -> dict[str, list[float]]:
        """Train with mini-batch Adam and early stopping."""
        x_train_t = self._tensor(x_train)
        y_train_t = self._tensor(y_train)
        x_val_t = self._tensor(x_val) if x_val is not None else None
        y_val_t = self._tensor(y_val) if y_val is not None else None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
        criterion = nn.MSELoss()
        history = {"train_loss": [], "val_loss": []}
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        no_improve = 0
        for _ in range(self.epochs):
            self.model.train()
            order = torch.randperm(x_train_t.shape[0], device=self.device)
            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                pred = self.model(x_train_t[batch_idx])
                loss = criterion(pred, y_train_t[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                train_loss = criterion(self.model(x_train_t), y_train_t).item()
                history["train_loss"].append(train_loss)
                if x_val_t is not None and y_val_t is not None:
                    val_loss = criterion(self.model(x_val_t), y_val_t).item()
                    history["val_loss"].append(val_loss)
                    if val_loss < best_val - 1e-6:
                        best_val = val_loss
                        best_state = copy.deepcopy(self.model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return history

    def predict(self, x: np.ndarray) -> torch.Tensor:
        """Predict outputs."""
        with torch.no_grad():
            return self.model(self._tensor(x)).detach().cpu()

    def embed(self, x: np.ndarray) -> torch.Tensor:
        """Return hidden representations."""
        x_t = self._tensor(x)
        with torch.no_grad():
            hidden = self.model[1](self.model[0](x_t))
        return hidden.detach().cpu()


class TorchLinearRegressor:
    """Single-layer Torch regressor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        l2: float = 1e-5,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = resolve_device(device)
        torch.manual_seed(seed)
        self.model = nn.Linear(input_dim, output_dim).to(self.device)
        self.l2 = l2

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 30,
    ) -> dict[str, list[float]]:
        """Train the linear model with Adam."""
        x_train_t = self._tensor(x_train)
        y_train_t = self._tensor(y_train)
        x_val_t = self._tensor(x_val) if x_val is not None else None
        y_val_t = self._tensor(y_val) if y_val is not None else None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
        criterion = nn.MSELoss()
        history = {"train_loss": [], "val_loss": []}
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        no_improve = 0
        for _ in range(self.epochs):
            pred = self.model(x_train_t)
            loss = criterion(pred, y_train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss = criterion(self.model(x_train_t), y_train_t).item()
                history["train_loss"].append(train_loss)
                if x_val_t is not None and y_val_t is not None:
                    val_loss = criterion(self.model(x_val_t), y_val_t).item()
                    history["val_loss"].append(val_loss)
                    if val_loss < best_val - 1e-6:
                        best_val = val_loss
                        best_state = copy.deepcopy(self.model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return history

    def predict(self, x: np.ndarray) -> torch.Tensor:
        """Predict outputs."""
        with torch.no_grad():
            return self.model(self._tensor(x)).detach().cpu()
