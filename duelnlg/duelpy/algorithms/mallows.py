"""Implementations of PB-MAB algorithms based on the Mallows model."""
from typing import List
from typing import Optional
from typing import Type

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import CondorcetProducer
from duelnlg.duelpy.algorithms.interfaces import CopelandRankingProducer
from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException
from duelnlg.duelpy.util.sorting import MergeSort
from duelnlg.duelpy.util.sorting import SortingAlgorithm
from duelnlg.duelpy.util.utility_functions import pop_random


class MallowsMPI(CondorcetProducer, PacAlgorithm):
    r"""Implementation of the Mallows Most Preferred Item algorithm.

    This algorithm finds the :term:`Condorcet winner` with a given error probability.

    It is assumed that the arms are sampled from a :term:`Mallows distribution`, which is stricter than the :term:`total order` assumption. See :cite:`busa2014preference` for details on this distribution.

    The amount of pairwise arm comparisons can is bound by :math:`\mathcal{O}\left(\frac{N}{\rho^2}\log\frac{N}{\delta\rho}\right)`, where :math:`N` is the number of arms, :math:`\delta` is the given error probability. The parameter :math:`\rho` is dependent on the :term:`Mallows distribution` parameter :math:`\phi` as follows: :math:`\rho=\frac{1-\phi}{1+\phi}`.

    This algorithm is part of the (:math:`\epsilon`,:math:`\delta`)-:term:`PAC` class of algorithms, with :math:`\epsilon = 0`. The :term:`Condorcet winner` is determined as the arm ranked first with the highest probability in the :term:`Mallows distribution`.
    The algorithm proceeds by selecting a random arm and comparing it against another arm until one of them can be considered worse than the other with sufficient confidence. The worse arm is discarded and the winner is compared against a new randomly chosen arm. This continues until only one arm is left, which is then returned. See :cite:`busa2014preference` for more details.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Optional, used for random choices in the algorithm.
    failure_probability
        An upper bound on the acceptable probability to fail, also called :math:`\delta` in :cite:`busa2014preference`.


    Attributes
    ----------
    feedback_mechanism
    failure_probability
    random_state
    preference_estimate
        Estimates for arm preferences.


    Examples
    --------
    Find the Condorcet winner in this example:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5],
    ... ])
    >>> random_state = np.random.RandomState(3)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix,
    ...                                       random_state=random_state)
    >>> mallows = MallowsMPI(feedback_mechanism, random_state=random_state, failure_probability=0.9)
    >>> mallows.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> arm = mallows.get_condorcet_winner()
    >>> arm, comparisons
    (2, 18)
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        failure_probability: float = 0.05,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.failure_probability = failure_probability
        self.random_state = (
            np.random.RandomState() if random_state is None else random_state
        )

        num_arms = self.feedback_mechanism.get_num_arms()

        def probability_scaling(num_samples: int) -> float:
            return num_arms * (2 * num_samples) ** 2

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability, probability_scaling
        )
        self.preference_estimate = PreferenceEstimate(num_arms, confidence_radius)
        self._current_arms = list(range(num_arms))
        self._best_arm = pop_random(self._current_arms, self.random_state)[0]

    def explore(self) -> None:
        """Explore arms by advancing the sorting algorithm."""
        rival_arm = pop_random(self._current_arms, self.random_state)[0]

        while (
            self.preference_estimate.get_lower_estimate(self._best_arm, rival_arm)
            <= 1 / 2
            and self.preference_estimate.get_upper_estimate(self._best_arm, rival_arm)
            >= 1 / 2
        ):
            result = self.feedback_mechanism.duel(self._best_arm, rival_arm)
            self.preference_estimate.enter_sample(self._best_arm, rival_arm, result)
            if self.is_finished():
                return
        if (
            self.preference_estimate.get_upper_estimate(self._best_arm, rival_arm)
            < 1 / 2
        ):
            self._best_arm = rival_arm

    def get_condorcet_winner(self) -> Optional[int]:
        """Get the arm with the highest probability of being the first in a ranking of the arms.

        Returns
        -------
        Optional[int]
            The best arm, None if has not been calculated yet.
        """
        self._best_arm
        # if len(self._current_arms) == 0:
        #     return self._best_arm
        # else:
        #     return None

    def get_winner(self):
        return self.get_condorcet_winner()

    def exploration_finished(self) -> bool:
        """Determine whether the best arm has been found."""
        return len(self._current_arms) == 0

    def exploit(self) -> None:
        """Run one step of exploitation."""
        winner = self.get_condorcet_winner()
        assert winner is not None
        self.feedback_mechanism.duel(winner, winner)

    def step(self) -> None:
        """Run one step of the algorithm."""
        if not self.exploration_finished():
            self.explore()
        else:
            self.exploit()


class MallowsMPR(CopelandRankingProducer, PacAlgorithm):
    r"""Implementation of Mallows Most Probable Ranking Algorithm.

    This algorithm computes a :term:`Copeland ranking` with a given error probability.

    It is assumed that the arms are sampled from a :term:`Mallows distribution`, which is stricter than the :term:`total order` assumption. See :cite:`busa2014preference` for details on this distribution.

    The amount of pairwise arm comparisons is bound by :math:`\mathcal{O}\left(\frac{N \log_2(N)}{\rho^2}\log\frac{N \log_2(N)}{\delta\rho}\right)`, where :math:`N` is the number of arms, :math:`\delta` is the given error probability. The parameter :math:`\rho` is dependent on the Mallows distribution parameter :math:`\phi` as follows: :math:`\rho=\frac{1-\phi}{1+\phi}`.

    This algorithm recursively builds a :term:`Copeland ranking` over the arms by sorting them using either :class:`Mergesort<duelpy.util.sorting.MergeSort>` or :class:`Quicksort<duelpy.util.sorting.Quicksort>`.
    Arms are compared repeatedly until sufficient confidence is obtained, this confidence is based on the :term:`Mallows distribution`.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Used for the random pivot selection in the Quicksort mode.
    failure_probability
        An upper bound on the acceptable probability to fail, called :math:`\delta` in :cite:`busa2014preference`.
    sorting_mode
        Determines which sort algorithm should be used, ``'merge'`` for Mergesort or ``'quick'`` for Quicksort.


    Attributes
    ----------
    feedback_mechanism
    random_state
    failure_probability
    sorting_algorithm
    preference_estimate
        Estimates for arm preferences.

    Examples
    --------
    Find the Condorcet winner in this example:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3, 0.2, 0.2],
    ...     [0.9, 0.7, 0.5, 0.8, 0.9],
    ...     [0.9, 0.8, 0.2, 0.5, 0.2],
    ...     [0.9, 0.8, 0.1, 0.8, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(3)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix,
    ...                                       random_state=random_state)
    >>> mallows = MallowsMPR(feedback_mechanism, random_state=random_state, failure_probability=0.9)
    >>> mallows.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> ranking = mallows.get_ranking()
    >>> ranking, comparisons
    ([2, 4, 3, 1, 0], 571)
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        failure_probability: float = 0.05,
        sorting_algorithm: Optional[Type[SortingAlgorithm]] = None,
    ):
        super().__init__(feedback_mechanism, time_horizon)

        num_arms = self.feedback_mechanism.get_num_arms()

        self.failure_probability = failure_probability

        arms = list(range(num_arms))
        if sorting_algorithm is None:
            sorting_algorithm = MergeSort
        self._sorting_algorithm = sorting_algorithm(
            arms, self._determine_better_arm, random_state
        )

        self._ranking: Optional[List[int]] = None

        def probability_scaling(num_samples: int) -> float:
            return (
                4
                * self._sorting_algorithm.get_comparison_bound(num_arms)
                * num_samples ** 2
            )

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability, probability_scaling
        )

        self.preference_estimate = PreferenceEstimate(num_arms, confidence_radius)

        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

    def _determine_better_arm(self, arm_1: int, arm_2: int) -> int:
        """Determine whether the first arm is preferred to the second with the given confidence.

        Parameters
        ----------
        arm_1
            The first arm.
        arm_2
            The second arm.

        Raises
        ------
        AlgorithmFinishedException
            If the comparison budget is reached.

        Returns
        -------
        int
            1 if the first arm is better, -1 if the second arm is better, 0 if not sure yet.
        """
        if self.is_finished():
            raise AlgorithmFinishedException()
        first_arm_won = self.feedback_mechanism.duel(arm_1, arm_2)
        self.preference_estimate.enter_sample(arm_1, arm_2, first_arm_won)
        if self.preference_estimate.get_lower_estimate(arm_1, arm_2) > 0.5:
            return 1
        elif self.preference_estimate.get_upper_estimate(arm_1, arm_2) < 0.5:
            return -1
        else:
            return 0

    def explore(self) -> None:
        """Explore arms by advancing the sorting algorithm."""
        try:
            self._sorting_algorithm.step()
        except AlgorithmFinishedException:
            pass
        if self._sorting_algorithm.is_finished():
            self._ranking = self._sorting_algorithm.get_result()

    def exploration_finished(self) -> bool:
        """Determine whether the ranking has been found."""
        return self._ranking is not None

    def get_ranking(self) -> Optional[List[int]]:
        """Get the computed ranking.

        Returns
        -------
        Optional[List[int]]
            The ranking, None if it has not been calculated yet.
        """
        return self._ranking
