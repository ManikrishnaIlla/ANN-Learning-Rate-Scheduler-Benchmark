import math
from scheduler.base import BaseScheduler

class StepDecayScheduler(BaseScheduler):
    def __init__(self, initial_lr: float, drop: float = 0.5,
                 step_size: int = 10):
        super().__init__(initial_lr)
        if not 0 < drop < 1:
            raise ValueError(f"drop must be between 0 and 1. Got: {drop}")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive. Got: {step_size}")
        self.drop = drop
        self.step_size = step_size

    def get_lr(self, epoch: int) -> float:
        factor = math.floor(epoch / self.step_size)
        return self.initial_lr * (self.drop ** factor)