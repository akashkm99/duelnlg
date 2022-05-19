"""Common metrics for algorithm performance."""

import time
from typing import Callable
from typing import Union

import numpy as np

from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix

__all__ = [
    "BestArmRate",
    "Metric",
    "AverageRegret",
    "StrongRegret",
    "WeakRegret",
    "AverageCopelandRegret",
    "TotalWallClock",
    "Cumulative",
    "ExponentialMovingAverage",
]


class Metric:
    """A metric that measures the performance of a PB-MAB algorithm."""

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Compute the metric value for a duel."""
        raise NotImplementedError()


class Regret(Metric):
    """The regret compared to the Condorcet winner.

    The regret of pulling an arm is defined by the calibrated preference
    probability of the best arm over the pulled one.

    This metric computes the per-duel regret. It is common practice to report
    the cumulative regret instead. You can combine this class with the
    ``Cumulative`` wrapper for that purpose.

    Parameters
    ----------
    preference_matrix
        The true preferences.
    aggregation_function
        A function that aggregates the regret of the two arms that are pulled
        in a single sample. Commonly mean, max or min. Also see
        ``AverageRegret``, ``StrongRegret`` and ``WeakRegret``.

    Examples
    --------
    >>> import numpy as np
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.9],
    ...     [0.1, 0.5],
    ... ]))
    >>> Regret(preference_matrix, max)(0, 1)
    0.4
    """

    def __init__(
        self,
        preference_matrix: Union[PreferenceMatrix, np.array],
        aggregation_function: Callable[[float, float], float],
    ):
        # Accept simple numpy arrays for convenience.
        if isinstance(preference_matrix, np.ndarray):
            preference_matrix = PreferenceMatrix(preference_matrix)
        self.preference_matrix = preference_matrix
        self.aggregation_function = aggregation_function
        self.best_arm = self.preference_matrix.get_condorcet_winner()
        assert (
            self.best_arm is not None
        ), "The regret can only be computed if a Condorcet winner exists."

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Compute the regret of a duel."""
        return (
            self.aggregation_function(
                self.preference_matrix[self.best_arm, arm_i_index],
                self.preference_matrix[self.best_arm, arm_j_index],
            )
            - 0.5
        )


class AverageRegret(Regret):
    """The average regret compared to the Condorcet winner.

    The regret of pulling an arm is defined by the calibrated preference
    probability of the best arm over the pulled one. This metric takes the
    average of the regret of the two pulled arms. Also see ``Regret``,
    ``StrongRegret`` and ``WeakRegret``.

    This metric computes the per-duel regret. It is common practice to report
    the cumulative regret instead. You can combine this class with the
    ``Cumulative`` wrapper for that purpose.

    Parameters
    ----------
    preference_matrix
        The true preferences.

    Examples
    --------
    >>> import numpy as np
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.9],
    ...     [0.1, 0.5],
    ... ]))
    >>> round(AverageRegret(preference_matrix)(0, 1), 2)
    0.2
    """

    def __init__(self, preference_matrix: Union[np.array, PreferenceMatrix]) -> None:
        super().__init__(
            preference_matrix, aggregation_function=lambda a, b: (a + b) / 2
        )


class StrongRegret(Regret):
    """The strong regret compared to the Condorcet winner.

    The regret of pulling an arm is defined by the calibrated preference
    probability of the best arm over the pulled one. This metric takes the
    maximum of the regret of the two pulled arms. Also see ``Regret``,
    ``AverageRegret`` and ``WeakRegret``.

    This metric computes the per-duel regret. It is common practice to report
    the cumulative regret instead. You can combine this class with the
    ``Cumulative`` wrapper for that purpose.

    Parameters
    ----------
    preference_matrix
        The true preferences.

    Examples
    --------
    >>> import numpy as np
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.9],
    ...     [0.1, 0.5],
    ... ]))
    >>> StrongRegret(preference_matrix)(0, 1)
    0.4
    """

    def __init__(self, preference_matrix: Union[np.array, PreferenceMatrix]) -> None:
        super().__init__(preference_matrix, aggregation_function=max)


class WeakRegret(Regret):
    """The weak regret compared to the Condorcet winner.

    The regret of pulling an arm is defined by the calibrated preference
    probability of the best arm over the pulled one. This metric takes the
    minimum of the regret of the two pulled arms. Also see ``Regret``,
    ``AverageRegret`` and ``StrongRegret``.

    This metric computes the per-duel regret. It is common practice to report
    the cumulative regret instead. You can combine this class with the
    ``Cumulative`` wrapper for that purpose.

    Parameters
    ----------
    preference_matrix
        The true preferences.

    Examples
    --------
    >>> import numpy as np
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.9],
    ...     [0.1, 0.5],
    ... ]))
    >>> WeakRegret(preference_matrix)(0, 1)
    0.0
    """

    def __init__(self, preference_matrix: Union[np.array, PreferenceMatrix]) -> None:
        super().__init__(preference_matrix, aggregation_function=min)


class AverageCopelandRegret:
    """Calculate Copeland regret with respect to normalized Copeland score.

    The average Copeland regret of a single comparison is the difference between the average normalized Copeland score of
    the pulled arms and the maximum normalized Copeland score. It can only be 0 if a Copeland winner is compared against
    another Copeland winner. Copeland score is normalized by the number of Arms(i.e number_of_arms-1). Finally, This function
    calculates the normalized cumulative Copeland regret accumulated over all time steps.

    Returns
    -------
    regret_history
        A list containing the Copeland regret per round.
    cumulative_regret
        The cumulative average regret.
    """

    def __init__(self, preference_matrix: Union[np.array, PreferenceMatrix]) -> None:
        # Accept simple numpy arrays for convenience.
        if isinstance(preference_matrix, np.ndarray):
            preference_matrix = PreferenceMatrix(preference_matrix)
        self.normalized_copeland_scores = (
            preference_matrix.get_normalized_copeland_scores()
        )
        self.max_normalized_copeland_score = np.amax(self.normalized_copeland_scores)

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Compute the Copeland regret of a duel."""
        return self.max_normalized_copeland_score - 0.5 * (
            self.normalized_copeland_scores[arm_i_index]
            + self.normalized_copeland_scores[arm_j_index]
        )


class TotalWallClock(Metric):
    """The wall clock time that has elapsed since initialization."""

    def __init__(self) -> None:
        super().__init__()
        self.start_time = time.time()

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Note the relative wall clock time at which a duel occurred."""
        return time.time() - self.start_time


class Cumulative(Metric):
    """Wraps a metric to make it cumulative.

    Metrics like the average regret are often reported in accumulated form.
    This wrapper can be used to convert any metric to a cumulative metric.

    Parameters
    ----------
    metric
        The metric to be wrapped.

    Examples
    --------
    >>> preference_matrix = np.array([
    ...     [0.5, 0.9],
    ...     [0.1, 0.5],
    ... ])
    >>> metric = Cumulative(AverageRegret(preference_matrix))
    >>> metric(1, 1)
    0.4
    >>> metric(1, 1)
    0.8
    """

    def __init__(self, metric: Metric):
        self.metric = metric
        self.accumulator = 0.0

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Compute the new metric value and add it to the accumulator."""
        self.accumulator += self.metric(arm_i_index, arm_j_index)
        return self.accumulator


class BestArmRate(Metric):
    """The rate of pulling the best arm.

    Parameters
    ----------
    best_arm
        The index of the best arm.

    Examples
    --------
    >>> BestArmRate(best_arm=1)(1, 2)
    0.5
    """

    def __init__(self, best_arm: int) -> None:
        self.best_arm = best_arm

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Compute the best arm rate for a duel."""
        best_arm_rate = 0.0
        if arm_i_index == self.best_arm:
            best_arm_rate += 0.5
        if arm_j_index == self.best_arm:
            best_arm_rate += 0.5
        return best_arm_rate


class ExponentialMovingAverage(Metric):
    r"""Wraps a metric to compute the exponential moving average.

    The exponential moving average is updated as follows:

    .. math::
      a' = \alpha v + (1 - \alpha) a

    where :math:`a` is the previous value of the average, :math:`\alpha` is a
    parameter that determines how much weight recent values have and :math:`v`
    is the new sample value.

    This can be used for some very basic smoothing if the metric will be
    plotted.

    Parameters
    ----------
    metric
        The metric to be wrapped.
    alpha
        The weight that is put on a new sample.
    initial_value
        The value to initialize the moving average with.

    Examples
    --------
    >>> metric = ExponentialMovingAverage(BestArmRate(best_arm=1), alpha=0.5, initial_value=0.0)
    >>> metric(1, 1)
    0.5
    >>> metric(1, 1)
    0.75
    >>> metric(1, 1)
    0.875
    >>> metric(0, 0)
    0.4375
    """

    def __init__(self, metric: Metric, alpha: float, initial_value: float = 0.0):
        self.metric = metric
        self.moving_average = initial_value
        self.alpha = alpha

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Compute the new metric value and update the moving average."""
        new_value = self.metric(arm_i_index, arm_j_index)
        self.moving_average = (self.alpha * new_value) + (
            1 - self.alpha
        ) * self.moving_average
        return self.moving_average
