from utils.exceptions import DataError, ModelError, SchedulerError, DimensionError
from utils.checks import (
    check_dimensions,
    check_not_empty,
    check_binary_labels,
    check_file_exists,
    check_positive_int,
    check_positive_float
)
from utils.seed import set_seed