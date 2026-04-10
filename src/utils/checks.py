from utils.exceptions import DimensionError, DataError


def check_dimensions(X, expected_features: int):

    actual = X.shape[1] if len(X.shape) == 2 else X.shape[0]
    if actual != expected_features:
        raise DimensionError(
            f"Input has {actual} features, model expects {expected_features}. "
            f"Check your input CSV or retrain with matching data."
        )


def check_not_empty(data, name: str = "dataset"):
    if data is None or len(data) == 0:
        raise DataError(f"The {name} is empty. Please provide valid data.")


def check_binary_labels(labels):
    unique = set(labels)
    if not unique.issubset({0, 1}):
        raise DataError(
            f"Labels must be binary (0 or 1). Found unexpected values: {unique - {0, 1}}"
        )


def check_file_exists(filepath: str):
    import os
    if not os.path.exists(filepath):
        raise DataError(f"File not found: '{filepath}'. Please check the path.")


def check_positive_int(value: int, name: str):
    if value <= 0:
        raise ValueError(f"'{name}' must be a positive integer. Got: {value}")


def check_positive_float(value: float, name: str):
    if value <= 0.0:
        raise ValueError(f"'{name}' must be a positive number. Got: {value}")