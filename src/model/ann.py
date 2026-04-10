import numpy as np
from utils.exceptions import ModelError
from utils.checks import check_positive_int


class ANN:
    def __init__(self, input_dim: int, hidden_dim: int):
        check_positive_int(input_dim, "input_dim")
        check_positive_int(hidden_dim, "hidden_dim")

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1

        # weights and biases — initialized in init_weights()
        self.W1 = None  # shape: (input_dim, hidden_dim)
        self.b1 = None  # shape: (1, hidden_dim)
        self.W2 = None  # shape: (hidden_dim, output_dim)
        self.b2 = None  # shape: (1, output_dim)

        # cache for backward pass
        self._cache = {}

    def init_weights(self, seed: int = 42) -> None:
        np.random.seed(seed)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * \
                  np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * \
                  np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, self.output_dim))

    def _relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def _relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)

    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._check_weights_initialized()

        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self._sigmoid(Z2)

        # cache for backward pass
        self._cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

        return A2.flatten()

    def backward(self, y_true: np.ndarray) -> dict:
        self._check_weights_initialized()

        X  = self._cache["X"]
        Z1 = self._cache["Z1"]
        A1 = self._cache["A1"]
        A2 = self._cache["A2"]
        n  = len(y_true)

        y_true = y_true.reshape(-1, 1)

        # output layer gradient
        # simplified: dZ2 = A2 - y (BCE loss + sigmoid combined)
        dZ2 = (A2 - y_true) / n          # shape: (n, 1)
        dW2 = A1.T @ dZ2                 # shape: (hidden_dim, 1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # shape: (1, 1)

        # hidden layer gradient
        dA1 = dZ2 @ self.W2.T            # shape: (n, hidden_dim)
        dZ1 = dA1 * self._relu_derivative(Z1)     # shape: (n, hidden_dim)
        dW1 = X.T @ dZ1                  # shape: (input_dim, hidden_dim)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # shape: (1, hidden_dim)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update(self, grads: dict, learning_rate: float) -> None:
        self.W1 -= learning_rate * grads["dW1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dW2"]
        self.b2 -= learning_rate * grads["db2"]

    def get_weights(self) -> dict:
        self._check_weights_initialized()
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist()
        }

    def set_weights(self, weights: dict) -> None:
        try:
            self.W1 = np.array(weights["W1"])
            self.b1 = np.array(weights["b1"])
            self.W2 = np.array(weights["W2"])
            self.b2 = np.array(weights["b2"])
        except KeyError as e:
            raise ModelError(f"Missing key in saved weights: {e}")

    def _check_weights_initialized(self) -> None:
        if self.W1 is None:
            raise ModelError(
                "Model weights are not initialized. "
                "Call init_weights() before forward() or backward()."
            )