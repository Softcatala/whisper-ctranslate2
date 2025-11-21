from enum import IntEnum


class ExitCode(IntEnum):
    # Returned when a major execution error occurs, such as attempting to use an unavailable GPU
    RUNTIME_ERROR = 100
