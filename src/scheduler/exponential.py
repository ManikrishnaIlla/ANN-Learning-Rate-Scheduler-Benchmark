import math
from scheduler.base import BaseScheduler

class ExponentialScheduler(BaseScheduler):
    def __init__(self, initial_lr: float, decay: float = 0.01):
        super().__init__(initial_lr)
        if decay <= 0:
            raise ValueError(f"decay must be positive. Got: {decay}")
        self.decay = decay

    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * math.exp(-self.decay * epoch)