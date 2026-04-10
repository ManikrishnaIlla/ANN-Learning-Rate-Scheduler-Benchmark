import numpy as np

class BinaryCrossEntropyLoss:
    def __init__(self, epsilon: float = 1e-8):
        # epsilon prevents log(0) which is undefined
        self.epsilon = epsilon

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return float(loss)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = len(y_true)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / n