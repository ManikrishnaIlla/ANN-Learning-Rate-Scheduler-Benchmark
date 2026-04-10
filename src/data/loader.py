import csv
from utils.exceptions import DataError
from utils.checks import check_file_exists


class CSVLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> tuple:
        check_file_exists(self.filepath)

        try:
            with open(self.filepath, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as e:
            raise DataError(f"Could not read file '{self.filepath}': {e}")

        if len(rows) == 0:
            raise DataError(f"File '{self.filepath}' is completely empty.")

        if len(rows) < 2:
            raise DataError(
                f"File '{self.filepath}' has only a header row and no data rows."
            )

        headers = rows[0]
        raw_rows = rows[1:]

        return headers, raw_rows