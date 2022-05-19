"""Implementation of the 'Winner Stays' algorithm variants for weak and strong regret settings."""

from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.algorithm import Algorithm
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.util.utility_functions import argmax_set


class WinnerStaysWeakRegret(Algorithm):
    r"""Implements the weak regret version of the `Winner Stays` algorithm :cite:`chen2017dueling`.

    The goal of this algorithm is to find the :term:`Condorcet winner` while minimizing the weak regret suffered in the process.

    The algorithm assumes at the very least that a :term:`Condorcet winner` exists, but the expected regret improves if a :term:`total order` over the arms exists.

    The incurred weak regret is constant in time and only depends on the number of arms :math:`N`: :math:`\mathcal{O}(N^2)`. If a :term:`total order` over the arms exists, this is improved to :math:`\mathcal{O}(N \log(N))`.

    The algorithm is tournament-based. It stores the difference between won and lost duels for each arm. The next arms are then selected from the set of arms with the highest difference. If one of the actions from the previous round is still in this argmax set, it is chosen again. In the first round and if the actions of the previous round are not part of the argmax set, the actions are chosen uniformly at random from it. The two chosen actions are guaranteed to be not identical.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        How many comparisons the algorithm should make. This does not impact the
        decision of the algorithm, only for how many times ``step`` executes.
        May be ``None`` to indicate a unknown or infinite time horizon.
    random_state
        Optional, used for random choices in the algorithm.

    Attributes
    ----------
    win_deltas
        Stores the difference between won and lost rounds for each arm. Corresponds to the :math:`C(t,i)` values in :cite:`chen2017dueling`.
    feedback_mechanism
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> from duelnlg.duelpy.stats.metrics import WeakRegret
    >>> from duelnlg.duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.4, 0.4],
    ...     [0.6, 0.5, 0.3],
    ...     [0.6, 0.7, 0.5],
    ... ])

    >>> random_state = np.random.RandomState(3)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"weak_regret": WeakRegret(preference_matrix)}
    ... )
    >>> ws_wr = WinnerStaysWeakRegret(feedback_mechanism, random_state=random_state)
    >>> for t in range(100):
    ...    ws_wr.step()
    >>> np.round(np.sum(feedback_mechanism.results["weak_regret"]), 2)
    0.6
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: np.random.RandomState = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.arm_count = self.feedback_mechanism.get_num_arms()
        self.win_deltas = np.zeros((self.arm_count,))
        self._last_arm_i = -1
        self._last_arm_j = -1

    def step(self) -> None:
        """Execute one round of the algorithm.

        Run through one round of the algorithm. First, the arms with the largest wins-losses difference are selected. Then the feedback is used to update the win-loss statistic.
        """
        # selection of arms i,j
        # arm i
        arm_i_candidates = argmax_set(self.win_deltas)

        if self._last_arm_i in arm_i_candidates:
            arm_i = self._last_arm_i
        elif self._last_arm_j in arm_i_candidates:
            arm_i = self._last_arm_j
        else:
            arm_i = self.random_state.choice(arm_i_candidates)

        # arm j
        # make sure we find the argmax without i and remove i if it is still in the argmax
        win_deltas_without_i = np.delete(self.win_deltas, arm_i)
        win_deltas_without_i_max = np.max(win_deltas_without_i)
        all_arms = np.arange(self.arm_count)
        mask = np.argwhere(
            np.logical_or(self.win_deltas < win_deltas_without_i_max, all_arms == arm_i)
        )
        arm_j_candidates = np.delete(all_arms, mask)

        if self._last_arm_i in arm_j_candidates:
            arm_j = self._last_arm_i
        elif self._last_arm_j in arm_j_candidates:
            arm_j = self._last_arm_j
        else:
            arm_j = self.random_state.choice(arm_j_candidates)

        self._last_arm_i = arm_i
        self._last_arm_j = arm_j

        # updating win-lose differences for the chosen arms
        if self.feedback_mechanism.duel(arm_i, arm_j):
            winner, loser = arm_i, arm_j
        else:
            winner, loser = arm_j, arm_i

        self.win_deltas[winner] += 1
        self.win_deltas[loser] -= 1

    def get_condorcet_winner(self) -> int:
        """Get the index of the arm currently believed to be the Condorcet winner.

        Returns
        -------
        int
            The Condorcet winner
        """
        # index as tie breaker, choose the smallest
        return argmax_set(self.win_deltas)[0]


class WinnerStaysStrongRegret(Algorithm):
    r"""Implements the strong regret version of the `Winner Stays` algorithm.

    The goal of this algorithm is to find the :term:`Condorcet winner` while minimizing the strong regret suffered in the process.

    The algorithm assumes at the very least that a :term:`Condorcet winner` exists, but the expected regret improves if a :term:`total order` over the arms exists.

    The incurred strong regret is dependent on the duels made :math:`T` and on the number of arms :math:`N`: :math:`\mathcal{O}(N^2 + N \log(T))`. If a :term:`total order` over the arms exists, this is improved to :math:`\mathcal{O}(N \log(T) + N \log(N))`.

    This algorithm is based on the weak regret version. It interleaves the weak regret `Winner Stays` algorithm with exponentially increasing periods of pure exploitation (pulling the currently believed-to-be-best arm twice).
    As soon as we have found the best arm, the strong regret in the exploitation phase will be :math:`0`. Since the duration is exponentially increasing, this leads to a strong regret of :math:`0` per round in the limit. For details see :cite:`chen2017dueling`.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    exploitation_factor
        Determines the length of rounds, i.e. how often the best arm should be pulled in each round.
        It should be larger than :math:`1`.
        The default is ``2``, which results in a doubling of the round length.
        This parameter is called :math:`\beta` in :cite:`chen2017dueling`.
    time_horizon
        How many comparisons the algorithm should do. This does not impact the
        decision of the algorithm, only for how many times ``step`` executes.
        May be ``None`` to indicate a unknown or infinite time horizon.
    random_state
        Optional, used for random choices in the algorithm.

    Attributes
    ----------
    feedback_mechanism
    exploitation_factor
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> from duelnlg.duelpy.stats.metrics import StrongRegret
    >>> from duelnlg.duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5],
    ... ])
    >>> random_state = np.random.RandomState(1)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"strong_regret": StrongRegret(preference_matrix)}
    ... )
    >>> ws_wr = WinnerStaysStrongRegret(feedback_mechanism, random_state=random_state)
    >>> for t in range(100):
    ...     ws_wr.step()
    >>> np.round(np.sum(feedback_mechanism.results["strong_regret"]), 2)
    1.8
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        exploitation_factor: float = 2,
        random_state: np.random.RandomState = np.random.RandomState(),
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        if exploitation_factor < 1:
            raise ValueError(
                "The exploitation_factor parameter needs to be larger than 1."
            )
        self._exploitation_factor = exploitation_factor
        # time_horizon is None since we control the execution manually
        self._ws = WinnerStaysWeakRegret(
            feedback_mechanism, time_horizon=None, random_state=random_state
        )
        self._round_index = 0
        self._round_length = 0
        self._current_round_iteration = 0
        self._best_arm = 0

    def step(self) -> None:
        """Execute one iteration of the algorithm.

        Step through one iteration of the algorithm. In the first iteration of each round, the weak regret winner stays is consulted, then the best arm is chosen.
        """
        if self._round_length == self._current_round_iteration:
            self._round_index += 1
            self._current_round_iteration = 0
            self._round_length = np.floor(
                self._exploitation_factor ** self._round_index
            )
            self._ws.step()
            self._best_arm = self._ws.get_condorcet_winner()
        else:
            self.feedback_mechanism.duel(self._best_arm, self._best_arm)
            self._current_round_iteration += 1

    def get_condorcet_winner(self) -> int:
        """Get the index of the arm currently believed to be the Condorcet winner.

        Returns
        -------
        int
            The Condorcet winner
        """
        return self._best_arm
