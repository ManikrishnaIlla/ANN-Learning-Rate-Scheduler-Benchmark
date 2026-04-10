import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from model_store.model_io import ModelIO
from utils.exceptions import ModelError, DataError


def test_missing_model_file():
    try:
        ModelIO.load("nonexistent_model.json")
        print("FAIL — should have raised DataError")
    except DataError as e:
        print(f"PASS — missing file caught: {e}")


def test_corrupted_json():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False) as f:
        f.write("{ this is not valid json {{")
        tmpfile = f.name

    try:
        ModelIO.load(tmpfile)
        print("FAIL — should have raised ModelError")
    except ModelError as e:
        print(f"PASS — corrupted JSON caught: {e}")
    finally:
        os.unlink(tmpfile)


def test_missing_keys_in_model():
    import tempfile, json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False) as f:
        json.dump({"architecture": {"input_dim": 4}}, f)
        tmpfile = f.name

    try:
        ModelIO.load(tmpfile)
        print("FAIL — should have raised ModelError")
    except ModelError as e:
        print(f"PASS — missing keys caught: {e}")
    finally:
        os.unlink(tmpfile)


if __name__ == "__main__":
    print("=== test_corrupted_model ===")
    test_missing_model_file()
    test_corrupted_json()
    test_missing_keys_in_model()