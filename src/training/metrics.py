import numpy as np


class Metrics:
    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray,
                 threshold: float = 0.5) -> float:
       
        predicted = (y_pred >= threshold).astype(int)
        return float(np.mean(predicted == y_true))

    @staticmethod
    def precision(y_pred: np.ndarray, y_true: np.ndarray,
                  threshold: float = 0.5) -> float:
        predicted = (y_pred >= threshold).astype(int)
        tp = np.sum((predicted == 1) & (y_true == 1))
        fp = np.sum((predicted == 1) & (y_true == 0))
        if tp + fp == 0:
            return 0.0
        return float(tp / (tp + fp))

    @staticmethod
    def recall(y_pred: np.ndarray, y_true: np.ndarray,
               threshold: float = 0.5) -> float:
        predicted = (y_pred >= threshold).astype(int)
        tp = np.sum((predicted == 1) & (y_true == 1))
        fn = np.sum((predicted == 0) & (y_true == 1))
        if tp + fn == 0:
            return 0.0
        return float(tp / (tp + fn))

    @staticmethod
    def f1_score(y_pred: np.ndarray, y_true: np.ndarray,
                 threshold: float = 0.5) -> float:
        p = Metrics.precision(y_pred, y_true, threshold)
        r = Metrics.recall(y_pred, y_true, threshold)
        if p + r == 0:
            return 0.0
        return float(2 * p * r / (p + r))

    @staticmethod
    def all_metrics(y_pred: np.ndarray,
                    y_true: np.ndarray,
                    threshold: float = 0.5) -> dict:
        return {
            "accuracy":  Metrics.accuracy(y_pred, y_true, threshold),
            "precision": Metrics.precision(y_pred, y_true, threshold),
            "recall":    Metrics.recall(y_pred, y_true, threshold),
            "f1":        Metrics.f1_score(y_pred, y_true, threshold)
        }

    @staticmethod
    def predict_labels(y_pred: np.ndarray,
                       threshold: float = 0.5) -> np.ndarray:
        return (y_pred >= threshold).astype(int)