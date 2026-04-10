from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    def __init__(self, initial_lr: float):
        if initial_lr <= 0:
            raise ValueError(
                f"initial_lr must be positive. Got: {initial_lr}"
            )
        self.initial_lr = initial_lr

    @abstractmethod
    def get_lr(self, epoch: int) -> float:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(initial_lr={self.initial_lr})"