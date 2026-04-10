import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
from utils.checks import check_dimensions
from utils.exceptions import DimensionError


def test_dimension_mismatch():
    X = np.random.randn(10, 6)  # 6 features
    expected = 4                 # model expects 4

    try:
        check_dimensions(X, expected)
        print("FAIL — should have raised DimensionError")
    except DimensionError as e:
        print(f"PASS — dimension mismatch caught: {e}")


def test_dimension_match():
    X = np.random.randn(10, 4)
    expected = 4

    try:
        check_dimensions(X, expected)
        print("PASS — correct dimensions accepted")
    except DimensionError:
        print("FAIL — should not have raised DimensionError")


if __name__ == "__main__":
    print("=== test_wrong_dimensions ===")
    test_dimension_mismatch()
    test_dimension_match()