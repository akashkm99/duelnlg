"""Exception classes for internal use."""


class AlgorithmFinishedException(Exception):
    """Raised to terminate early when the next duel would exceed the time horizon."""
