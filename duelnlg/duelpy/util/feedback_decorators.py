"""Decorators to alter the behavior of feedback mechanisms."""

from typing import Dict
from typing import List

from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats.metrics import Metric
from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException


class FeedbackMechanismDecorator(FeedbackMechanism):
    """A feedback mechanism that delegates to another feedback mechanism.

    This is intended  to be used as a base class for wrappers that want to
    "inject" some behavior or checks into an existing feedback mechanism. See
    ``BudgetedFeedbackMechanism`` for an example.

    Parameters
    ----------
    feedback_mechanism
        The FeedbackMechanism object to delegate to.
    """

    def __init__(self, feedback_mechanism: FeedbackMechanism) -> None:
        # We override all functions and delegate to an existing feedback
        # mechanism. Therefore it does not make much sense to call the super
        # constructor here.
        # pylint: disable=super-init-not-called
        self.feedback_mechanism = feedback_mechanism

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        return self.feedback_mechanism.duel(arm_i_index, arm_j_index)

    def get_num_duels(self) -> int:
        """Get the number of duels that were already performed.

        Returns
        -------
        int
            The number of duels.
        """
        return self.feedback_mechanism.get_num_duels()

    def get_arms(self) -> list:
        """Get the pool of arms available."""
        return self.feedback_mechanism.get_arms()

    def get_num_arms(self) -> int:
        """Get the number of arms."""
        return self.feedback_mechanism.get_num_arms()


class MetricKeepingFeedbackMechanism(FeedbackMechanismDecorator):
    """A feedback mechanism that updates a set of metrics on every duel.

    This can be used if you want to keep track on some aspects of an algorithms
    performance during its execution.

    Parameters
    ----------
    feedback_mechanism
        The FeedbackMechanism object to delegate to.
    metrics
        A dictionary of metrics to apply, keyed by their name.
    sample_interval
        The number of time steps per sample.

    Attributes
    ----------
    feedback_mechanism
    metrics
    sample_interval
    results
        A dictionary of lists, keyed by the names of the metrics.

    Examples
    --------
    Define a very simple preference-based multi-armed bandit problem through a
    preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix
    >>> import numpy as np
    >>> random_state = np.random.RandomState(42)
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.8],
    ...     [0.2, 0.5],
    ... ]))
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)

    Now let's run an algorithm on this problem and keep track of the cumulative
    regret:

    >>> from duelnlg.duelpy.algorithms import Savage
    >>> from duelnlg.duelpy.stats.metrics import AverageRegret, Cumulative
    >>> metric_keeping_feedback = MetricKeepingFeedbackMechanism(
    ...     feedback_mechanism,
    ...     metrics={"average_regret": Cumulative(AverageRegret(preference_matrix))},
    ... )
    >>> algorithm = Savage(metric_keeping_feedback)
    >>> algorithm.run()
    >>> metric_keeping_feedback.results
    {'average_regret': [0.150..., 0.300...]}
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        metrics: Dict[str, Metric],
        sample_interval: int = 1,
    ):
        super().__init__(feedback_mechanism)
        self.metrics = metrics
        self.sample_interval = sample_interval
        self.results: Dict[str, List[float]] = {key: [] for key in metrics.keys()}
        # Always add an implicit "time step metric". Can be seen as a key or
        # index for the other metrics.
        self.results["time_step"] = []

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Raises
        ------
        AlgorithmFinishedException
            If the budget would be exceeded by this duel.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        result = super().duel(arm_i_index, arm_j_index)
        for (name, metric) in self.metrics.items():
            # Always call the metric, in case it keeps some internal state.
            metric_value = metric(arm_i_index, arm_j_index)
            if self.get_num_duels() % self.sample_interval == 0:
                self.results[name].append(metric_value)
        if self.get_num_duels() % self.sample_interval == 0:
            self.results["time_step"].append(self.get_num_duels())
        return result


class BudgetedFeedbackMechanism(FeedbackMechanismDecorator):
    """A feedback mechanism wrapper that ensures a duel budget is not exceeded.

    This can be used to provide an upper-bound on the number of duels that some
    function (that may be out of the algorithm's control) can perform. Examples
    are calls to sorting algorithms or other multi-armed bandit algorithms.
    Using this wrapper is different from directly passing a ``time_horizon``
    because this only provides a lower, but not an upper bound on the number
    duels.

    Examples
    --------
    Define a very simple preference-based multi-armed bandit problem through a
    preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> import numpy as np
    >>> random_state = np.random.RandomState(42)
    >>> preference_matrix = np.array([
    ...     [0.5, 0.7],
    ...     [0.3, 0.5],
    ... ])
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)

    Now run a PAC algorithm on it in various different configurations. First
    let's try to run it unmodified and without a time horizon:

    >>> from duelnlg.duelpy.algorithms import Savage

    >>> pac_algorithm = Savage(feedback_mechanism)
    >>> pac_algorithm.run()
    >>> feedback_mechanism.get_num_duels()
    2
    >>> pac_algorithm.get_copeland_winner()
    0

    Now let's say we only have one more duel to spare. We can use the wrapper
    for that:

    >>> from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException
    >>> pac_algorithm = Savage(BudgetedFeedbackMechanism(feedback_mechanism, max_duels=1))
    >>> try:
    ...     pac_algorithm.run()
    ... except AlgorithmFinishedException:
    ...     # The algorithm was not able to find a Copeland winner with the limited duel budget.
    ...     pass
    >>> feedback_mechanism.get_num_duels()  # Just one additional duel
    3
    >>> pac_algorithm.get_copeland_winner() is None  # Algorithm terminated early
    True

    But if the algorithm is able to complete within the budget, the behavior is
    unchanged:

    >>> pac_algorithm = Savage(BudgetedFeedbackMechanism(feedback_mechanism, max_duels=100))
    >>> try:
    ...     pac_algorithm.run()
    ... except AlgorithmFinishedException:
    ...     # This should not happen, the budget is sufficiently large
    ...     assert False
    >>> feedback_mechanism.get_num_duels()
    5
    >>> pac_algorithm.get_copeland_winner()
    0

    Which is different from how the algorithm would behave if we would pass a
    time horizon instead:

    >>> pac_algorithm = Savage(feedback_mechanism, time_horizon=100)
    >>> pac_algorithm.run()
    >>> feedback_mechanism.get_num_duels()  # Time horizon is both upper and lower limit.
    100
    >>> pac_algorithm.get_copeland_winner()
    0

    Parameters
    ----------
    feedback_mechanism
        The FeedbackMechanism object to delegate to.
    """

    def __init__(self, feedback_mechanism: FeedbackMechanism, max_duels: int) -> None:
        super().__init__(feedback_mechanism)
        self.max_duels = max_duels
        self.duels_conducted = 0

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Raises
        ------
        AlgorithmFinishedException
            If the budget would be exceeded by this duel.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        if self.duels_conducted >= self.max_duels:
            raise AlgorithmFinishedException()
        result = super().duel(arm_i_index, arm_j_index)
        self.duels_conducted += 1
        return result
