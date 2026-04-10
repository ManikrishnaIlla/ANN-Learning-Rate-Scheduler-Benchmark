from scheduler.base import BaseScheduler
from scheduler.constant import ConstantScheduler
from scheduler.step_decay import StepDecayScheduler
from scheduler.exponential import ExponentialScheduler
from scheduler.cosine import CosineAnnealingScheduler


def get_scheduler(name: str, initial_lr: float, epochs: int) -> BaseScheduler:
    from utils.exceptions import SchedulerError

    schedulers = {
        "constant":    ConstantScheduler(initial_lr=initial_lr),
        "step":        StepDecayScheduler(initial_lr=initial_lr),
        "exponential": ExponentialScheduler(initial_lr=initial_lr),
        "cosine":      CosineAnnealingScheduler(initial_lr=initial_lr, T=epochs),
    }

    if name not in schedulers:
        raise SchedulerError(
            f"Unknown scheduler '{name}'. "
            f"Valid options are: {list(schedulers.keys())}"
        )

    return schedulers[name]