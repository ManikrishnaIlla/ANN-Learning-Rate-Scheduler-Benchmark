import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from data.loader import CSVLoader
from utils.exceptions import DataError


def test_completely_empty_file():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False) as f:
        tmpfile = f.name  # write nothing

    try:
        loader = CSVLoader(tmpfile)
        loader.load()
        print("FAIL — should have raised DataError")
    except DataError as e:
        print(f"PASS — empty file caught: {e}")
    finally:
        os.unlink(tmpfile)


def test_header_only_file():
    import tempfile, csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "label"])  # header only
        tmpfile = f.name

    try:
        loader = CSVLoader(tmpfile)
        loader.load()
        print("FAIL — should have raised DataError")
    except DataError as e:
        print(f"PASS — header-only file caught: {e}")
    finally:
        os.unlink(tmpfile)

if __name__ == "__main__":
    print("=== test_empty_file ===")
    test_completely_empty_file()
    test_header_only_file()