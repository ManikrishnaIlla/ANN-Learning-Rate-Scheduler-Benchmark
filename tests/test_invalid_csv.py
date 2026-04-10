import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from data.loader import CSVLoader
from data.validator import DataValidator
from utils.exceptions import DataError

def test_non_numeric_feature():
    import tempfile, csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "label"])
        writer.writerow(["0.5", "abc", "1"])   # abc is not numeric
        tmpfile = f.name

    try:
        loader = CSVLoader(tmpfile)
        headers, raw_rows = loader.load()
        validator = DataValidator(is_training=True)
        validator.validate(headers, raw_rows)
        print("FAIL — should have raised DataError")
    except DataError as e:
        print(f"PASS — non-numeric feature caught: {e}")
    finally:
        os.unlink(tmpfile)


def test_invalid_label():
    import tempfile, csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "label"])
        writer.writerow(["0.5", "0.3", "3"])   # label 3 is invalid
        tmpfile = f.name

    try:
        loader = CSVLoader(tmpfile)
        headers, raw_rows = loader.load()
        validator = DataValidator(is_training=True)
        validator.validate(headers, raw_rows)
        print("FAIL — should have raised DataError")
    except DataError as e:
        print(f"PASS — invalid label caught: {e}")
    finally:
        os.unlink(tmpfile)


def test_inconsistent_columns():
    import tempfile, csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "label"])
        writer.writerow(["0.5", "0.3", "1"])
        writer.writerow(["0.5", "1"])           # missing column
        tmpfile = f.name

    try:
        loader = CSVLoader(tmpfile)
        headers, raw_rows = loader.load()
        validator = DataValidator(is_training=True)
        validator.validate(headers, raw_rows)
        print("FAIL — should have raised DataError")
    except DataError as e:
        print(f"PASS — inconsistent columns caught: {e}")
    finally:
        os.unlink(tmpfile)


if __name__ == "__main__":
    print("=== test_invalid_csv ===")
    test_non_numeric_feature()
    test_invalid_label()
    test_inconsistent_columns()