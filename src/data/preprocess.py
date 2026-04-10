import numpy as np
from utils.exceptions import DataError
from utils.checks import check_not_empty


class Preprocessor:
    def __init__(self, test_size: float = 0.2, normalize: bool = True):
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0 and 1. Got: {test_size}")
        self.test_size = test_size
        self.normalize = normalize
        self.mean_ = None
        self.std_ = None

    def parse(self, raw_rows: list, is_training: bool = True) -> tuple:
        check_not_empty(raw_rows, "data rows")

        try:
            if is_training:
                X = np.array([list(map(float, row[:-1])) for row in raw_rows])
                y = np.array([int(row[-1]) for row in raw_rows])
            else:
                X = np.array([list(map(float, row)) for row in raw_rows])
                y = None
        except Exception as e:
            raise DataError(f"Failed to parse data into numeric arrays: {e}")

        return X, y

    def split(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple:
        np.random.seed(seed)
        n = len(X)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - self.test_size))

        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.mean_ = X_train.mean(axis=0)
        self.std_ = X_train.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return (X_train - self.mean_) / self.std_

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise DataError(
                "Preprocessor has not been fitted yet. "
                "Call fit_transform on training data first."
            )
        return (X - self.mean_) / self.std_

    def get_norm_params(self) -> dict:
        if self.mean_ is None:
            return {"mean": None, "std": None}
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist()
        }

    def set_norm_params(self, params: dict) -> None:
        if params["mean"] is None:
            return
        self.mean_ = np.array(params["mean"])
        self.std_ = np.array(params["std"])