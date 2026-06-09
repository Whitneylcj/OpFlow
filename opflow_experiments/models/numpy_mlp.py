"""Small NumPy MLP utilities used for baseline experiments."""

from __future__ import annotations

import copy

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU."""
    return (x > 0.0).astype(float)


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


class FeedForwardRegressor:
    """Two-layer MLP regressor trained with mini-batch Adam."""

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
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        scale_1 = np.sqrt(2.0 / max(input_dim, 1))
        scale_2 = np.sqrt(2.0 / max(hidden_dim, 1))
        self.params = {
            "w1": self.rng.normal(0.0, scale_1, size=(input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "w2": self.rng.normal(0.0, scale_2, size=(hidden_dim, output_dim)),
            "b2": np.zeros(output_dim),
        }
        self.opt_state = {key: {"m": np.zeros_like(value), "v": np.zeros_like(value)} for key, value in self.params.items()}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Forward pass."""
        h_pre = x @ self.params["w1"] + self.params["b1"]
        h = relu(h_pre)
        out = h @ self.params["w2"] + self.params["b2"]
        cache = {"x": x, "h_pre": h_pre, "h": h, "out": out}
        return out, cache

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict outputs."""
        out, _ = self.forward(x)
        return out

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        out, cache = self.forward(x)
        err = out - y
        batch_size = max(len(x), 1)
        loss = float(np.mean(err ** 2))
        loss += 0.5 * self.l2 * (
            np.sum(self.params["w1"] ** 2) + np.sum(self.params["w2"] ** 2)
        )
        dout = (2.0 / batch_size) * err
        grads = {
            "w2": cache["h"].T @ dout + self.l2 * self.params["w2"],
            "b2": dout.sum(axis=0),
        }
        dh = dout @ self.params["w2"].T
        dh_pre = dh * relu_grad(cache["h_pre"])
        grads["w1"] = cache["x"].T @ dh_pre + self.l2 * self.params["w1"]
        grads["b1"] = dh_pre.sum(axis=0)
        return loss, grads

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 30,
    ) -> dict[str, list[float]]:
        """Train with mini-batch Adam and simple early stopping."""
        history = {"train_loss": [], "val_loss": []}
        best_params = copy.deepcopy(self.params)
        best_val = float("inf")
        no_improve = 0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        step = 0
        for _ in range(self.epochs):
            indices = self.rng.permutation(len(x_train))
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                loss, grads = self._loss_and_grads(x_train[batch_idx], y_train[batch_idx])
                step += 1
                for key in self.params:
                    state = self.opt_state[key]
                    state["m"] = beta1 * state["m"] + (1.0 - beta1) * grads[key]
                    state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grads[key] ** 2)
                    m_hat = state["m"] / (1.0 - beta1 ** step)
                    v_hat = state["v"] / (1.0 - beta2 ** step)
                    self.params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            train_loss, _ = self._loss_and_grads(x_train, y_train)
            history["train_loss"].append(train_loss)
            if x_val is not None and y_val is not None and len(x_val) > 0:
                val_pred = self.predict(x_val)
                val_loss = float(np.mean((val_pred - y_val) ** 2))
                history["val_loss"].append(val_loss)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_params = copy.deepcopy(self.params)
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
        self.params = best_params
        return history


class LinearRegressor:
    """Closed-form ridge linear regressor."""

    def __init__(self, l2: float = 1e-5) -> None:
        self.l2 = l2
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit ridge regression."""
        eye = np.eye(x.shape[1])
        self.coef_ = np.linalg.solve(x.T @ x + self.l2 * eye, x.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict outputs."""
        assert self.coef_ is not None
        return x @ self.coef_
