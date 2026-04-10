import math
from scheduler.base import BaseScheduler

class CosineAnnealingScheduler(BaseScheduler):
    def __init__(self, initial_lr: float, lr_min: float = 0.0,
                 T: int = 100):
        super().__init__(initial_lr)
        if lr_min < 0:
            raise ValueError(f"lr_min must be >= 0. Got: {lr_min}")
        if T <= 0:
            raise ValueError(f"T must be positive. Got: {T}")
        if lr_min >= initial_lr:
            raise ValueError(
                f"lr_min ({lr_min}) must be less than initial_lr ({initial_lr})"
            )
        self.lr_min = lr_min
        self.T = T

    def get_lr(self, epoch: int) -> float:
        cosine_term = math.cos(math.pi * epoch / self.T)
        return self.lr_min + 0.5 * (self.initial_lr - self.lr_min) * \
               (1 + cosine_term)