import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from scheduler import get_scheduler
from utils.exceptions import SchedulerError


def test_invalid_scheduler_name():
    try:
        get_scheduler("adam", initial_lr=0.01, epochs=100)
        print("FAIL — should have raised SchedulerError")
    except SchedulerError as e:
        print(f"PASS — invalid scheduler caught: {e}")


def test_negative_lr():
    try:
        get_scheduler("constant", initial_lr=-0.01, epochs=100)
        print("FAIL — should have raised ValueError")
    except ValueError as e:
        print(f"PASS — negative LR caught: {e}")


def test_valid_schedulers():
    for name in ["constant", "step", "exponential", "cosine"]:
        try:
            s = get_scheduler(name, initial_lr=0.01, epochs=100)
            print(f"PASS — {name} scheduler created: {s}")
        except Exception as e:
            print(f"FAIL — {name} raised unexpected error: {e}")


if __name__ == "__main__":
    print("=== test_invalid_scheduler ===")
    test_invalid_scheduler_name()
    test_negative_lr()
    test_valid_schedulers()