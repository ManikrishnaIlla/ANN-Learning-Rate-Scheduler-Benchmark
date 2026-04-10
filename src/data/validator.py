from utils.exceptions import DataError
from utils.checks import check_not_empty


class DataValidator:

    def __init__(self, expected_columns: int = None, is_training: bool = True):
        self.expected_columns = expected_columns
        self.is_training = is_training

    def validate(self, headers: list, raw_rows: list) -> None:
        check_not_empty(raw_rows, "data rows")
        self._check_column_consistency(headers, raw_rows)
        self._check_numeric_features(raw_rows)

        if self.is_training:
            self._check_binary_labels(raw_rows)

    def _check_column_consistency(self, headers: list, raw_rows: list) -> None:
        expected = len(headers)
        for i, row in enumerate(raw_rows, start=2):
            if len(row) != expected:
                raise DataError(
                    f"Row {i} has {len(row)} columns, expected {expected}. "
                    f"Check your CSV for missing or extra values."
                )

    def _check_numeric_features(self, raw_rows: list) -> None:
        for i, row in enumerate(raw_rows, start=2):
            feature_values = row[:-1]
            for j, val in enumerate(feature_values):
                try:
                    float(val)
                except ValueError:
                    raise DataError(
                        f"Row {i}, column {j + 1}: expected numeric value, "
                        f"got '{val}'. All feature values must be numbers."
                    )

    def _check_binary_labels(self, raw_rows: list) -> None:
        for i, row in enumerate(raw_rows, start=2):
            label = row[-1].strip()
            if label not in {"0", "1"}:
                raise DataError(
                    f"Row {i}: invalid label '{label}'. "
                    f"Labels must be binary: 0 or 1."
                )