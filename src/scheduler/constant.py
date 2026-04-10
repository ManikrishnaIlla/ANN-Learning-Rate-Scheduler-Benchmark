from scheduler.base import BaseScheduler

class ConstantScheduler(BaseScheduler):
    def __init__(self, initial_lr: float):
        super().__init__(initial_lr)

    def get_lr(self, epoch: int) -> float:
        return self.initial_lr