"""Find the Condorcet winner in a PB-MAB problem using the 'Beat the Mean Bandit' algorithm."""
from typing import Optional
from typing import Tuple

import numpy as np

from duelnlg.duelpy.algorithms.algorithm import Algorithm
from duelnlg.duelpy.algorithms.interfaces import CondorcetProducer
from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats.confidence_radius import ConfidenceRadius
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.util.utility_functions import argmax_set
from duelnlg.duelpy.util.utility_functions import argmin_set


class BeatTheMeanBandit(CondorcetProducer, PacAlgorithm):
    r"""Implements the Beat the Mean Bandit algorithm.

    The goal of this algorithm is to find the :term:`Condorcet winner`.

    It is assumed that a :term:`total order` over the arms exists. Additionally :term:`relaxed stochastic transitivity` and the
    :term:`stochastic triangle inequality` are assumed.

    The online version (if a time horizon is supplied) has a high probability cumulative regret bound of :math:`\mathcal{O}\left(\frac{\gamma^7 N}{\epsilon_\ast} \log T\right)`.
    The :math:`\gamma` is part of the :math:`\gamma`-:term:`relaxed stochastic transitivity` assumption. :math:`N` is the number of
    arms. :math:`\epsilon_\ast` is the winning probability of the best arm against the second best arm minus :math:`0.5`.

    This is an explore-then-exploit algorithm. The Beat the Mean algorithm, as described in :cite:`yue2011beat`,
    proceeds in a sequence of rounds and maintains a working set of active arms during each round. For each active
    arm, an empirical estimate is maintained for how likely an arm is to beat the mean bandit of the working set. In
    each iteration, an arm with the fewest recorded comparisons is selected for comparison. We then enter an exploit
    phase by repeatedly choosing ``best_arm`` until reaching :math:`T` total comparisons. The algorithm terminates only
    when one active arm remains, or when time horizon is reached. If a time horizon is given, this algorithm matches the
    "Online" variant in :cite:`yue2011beat`.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        The total number of rounds.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    gamma
        The :term:`relaxed stochastic transitivity` (corresponds to :math:`\gamma` in :cite:`yue2011beat`) that the
        algorithm should assume for the given problem setting. The value must be greater than :math:`0`.
        A higher value corresponds to a stronger assumption, where :math:`1` corresponds to :term:`strong stochastic
        transitivity`. In theory it is not possible to assume more than a gamma of :math:`1`, but in practice you can
        still specify higher values. This will lead to tighter confidence intervals and possibly better results,
        but the theoretical guarantees do not hold in that case.


    Attributes
    ----------
    comparison_history
        A ``ComparisonHistory`` object which stores the history of the comparisons between the arms.
    random_state
    feedback_mechanism
    time_horizon

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> from duelnlg.duelpy.stats.metrics import AverageRegret
    >>> from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix
    >>> from duelnlg.duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5],
    ... ]))
    >>> random_state = np.random.RandomState(43)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"average_regret": AverageRegret(preference_matrix)}
    ... )
    >>> time_horizon = 10000  # time horizon greater than or equal to number of arms.
    >>> btm = BeatTheMeanBandit(feedback_mechanism=feedback_mechanism, time_horizon=time_horizon, random_state=random_state, gamma=1.0)
    >>> btm.run()
    >>> best_arm = btm.get_condorcet_winner()
    >>> best_arm
    2
    >>> cumul_regret = np.sum(feedback_mechanism.results["average_regret"])
    >>> np.round(cumul_regret, 2)
    903.1
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.RandomState = None,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            np.random.RandomState() if random_state is None else random_state
        )
        # time_horizon is initialized as an int in __init__ and can never be None. This
        # assertion is necessary for mypy since time_horizon is defined as
        # Optional[int] in the superclass.
        assert self.time_horizon is not None
        # Allowed failure-probability (corresponds to :math:`\delta` in :cite:`yue2011beat`), i.e. probability
        # that the actual value lies outside of the computed confidence interval. Derived from the Hoeffding bound.
        self.failure_probability = 1 / (
            2 * self.time_horizon * self.feedback_mechanism.get_num_arms()
        )
        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability=self.failure_probability, factor=9 * (gamma ** 4) * 2,
        )
        self.comparison_history = ComparisonHistory(
            number_of_arms=self.feedback_mechanism.get_num_arms(),
            confidence_radius=confidence_radius,
        )

    def explore(self) -> None:
        """Run one step of exploration."""
        arm1, arm2 = self.comparison_history.get_dueling_arms(
            random_state=self.random_state
        )
        first_won = self.feedback_mechanism.duel(arm_i_index=arm1, arm_j_index=arm2)
        self.comparison_history.enter_sample(arm1=arm1, arm2=arm2, first_won=first_won)
        if (
            self.comparison_history.get_lower_bound()
            >= self.comparison_history.get_upper_bound()
        ):
            # Check if the empirically worst bandit is separated from the empirically best one by a sufficient
            # confidence margin.
            worst_arm = self.comparison_history.get_worst_arm()
            self.remove_from_working_set(worst_arm)

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        Returns
        -------
        bool
            Whether exploration is finished.
        """
        return self.comparison_history.size_working_set <= 1

    def remove_from_working_set(self, worst_arm: int) -> None:
        """Remove the worst arm from the working set and update the comparison statistics.

        Parameters
        ----------
        worst_arm
            The index of the empirically worst arm.
        """
        self.comparison_history.remove_arm_history(worst_arm=worst_arm)

    def get_condorcet_winner(self) -> int:
        """Return the arm with highest empirical estimate.

        Returns
        -------
        int
            The arm with the highest empirical estimate.
        """
        return argmax_set(array=self.comparison_history.probability_estimate)[0]

    def get_winner(self):
        return self.get_condorcet_winner()


class ComparisonHistory:
    r"""Store the comparison history of the working set.

    Parameters
    ----------
    number_of_arms
        The number of arms in the estimated preference matrix.
    confidence_radius
        The confidence radius to use when computing confidence intervals.

    Attributes
    ----------
    working_set
        Stores the set of active arms. Corresponds to :math:`W_l` in :cite:`yue2011beat`.
    size_working_set
        Stores the amount of active arms in ``working_set``. Using this instead of summing over ``working_set`` improves performance.
    comparisons
        Stores the total number of comparisons of a specific arm. Corresponds to :math:`n_b` in :cite:`yue2011beat`.
    wins
        Stores the number of wins of each arm. Corresponds to :math:`w_b` in :cite:`yue2011beat`.
    probability_estimate
        Stores the empirical estimate of arms versus the mean bandit. Corresponds to :math:`\hat{P}_b` in :cite:`yue2011beat`.
    confidence_radius
    """

    def __init__(
        self, number_of_arms: int, confidence_radius: ConfidenceRadius
    ) -> None:
        self.confidence_radius = confidence_radius
        self.working_set = np.full(number_of_arms, True, dtype=bool)
        self.size_working_set = number_of_arms
        self.comparisons = np.zeros((number_of_arms, number_of_arms), dtype=int)
        self.wins = np.zeros((number_of_arms, number_of_arms), dtype=int)
        self.probability_estimate = np.full(number_of_arms, 0.5)
        self._comparison_count_cache = np.zeros(number_of_arms, dtype=int)

    def get_min_comparison(self) -> int:
        """Return the minimum value in comparisons.

        Returns
        -------
        int
            The minimum value in comparisons.
        """
        return min(self._comparison_count_cache[self.working_set])

    def get_dueling_arms(self, random_state: np.random.RandomState) -> Tuple[int, int]:
        """Return the least sampled arm and a random challenger.

        Parameters
        ----------
        random_state
            A numpy random state. Defaults to an unseeded state when not specified.

        Returns
        -------
        arm1
            The index of challenger arm from the ``working_set``.
        arm2
            The index of arm to compare against.
        """
        arm1 = random_state.choice(
            argmin_set(
                array=np.ma.array(self._comparison_count_cache, mask=~self.working_set)
            )
        )  # break ties randomly within the working set

        arm1_mask = np.full(self.working_set.shape, True, dtype=np.bool)
        arm1_mask[arm1] = False
        arm2 = random_state.choice(
            np.argwhere(self.working_set & arm1_mask).flatten()
        )  # select arm2 from the current working_set at random
        return arm1, arm2

    def enter_sample(self, arm1: int, arm2: int, first_won: bool) -> None:
        """Enter the result of a sampled duel.

        Parameters
        ----------
        arm1
            The index of the first arm of the duel.
        arm2
            The index of the second arm of the duel.
        first_won
            Whether the first arm won the duel.
        """
        if first_won:
            self.wins[arm1][arm2] += 1
        self.comparisons[arm1][arm2] += 1
        self._comparison_count_cache[arm1] += 1

        self.update_probability_estimate()

    def update_probability_estimate(self) -> None:
        """Set the estimate of the win probability of the arm."""
        wins = np.sum(self.wins, axis=1)
        self.probability_estimate = np.divide(
            wins,
            self._comparison_count_cache,
            out=np.full(wins.shape, 0.5),
            where=self._comparison_count_cache != 0,
        )

    def remove_arm_history(self, worst_arm: int) -> None:
        """Remove the worst arm and the associated comparison history.

        Remove the empirically worst arm from the working set and update the comparisons, wins and probability
        estimate of each arm in the current working set.

        Parameters
        ----------
        worst_arm
            The index of the empirically worst arm.
        """
        self._comparison_count_cache -= self.comparisons[worst_arm, :]
        if self.working_set[worst_arm]:
            self.size_working_set -= 1
            self.working_set[worst_arm] = False
        self.update_probability_estimate()  # update the probability_estimate

    def get_worst_arm(self) -> int:
        """Get the empirically worst arm from the current ``working_set``.

        Returns
        -------
        int
            The index of the empirically worst arm in the current ``working_set``.
        """
        return argmin_set(
            array=np.ma.array(self.probability_estimate, mask=~self.working_set)
        )[0]

    def get_upper_bound(self) -> float:
        """Get the upper bound for empirically worst arm.

        Returns
        -------
        float
            The upper bound for empirically worst arm.
        """
        return min(
            self.probability_estimate[self.working_set]
        ) + self.confidence_radius(self.get_min_comparison())

    def get_lower_bound(self) -> float:
        """Get the lower bound for empirically best arm.

        Returns
        -------
        float
            The lower bound for empirically best arm.
        """
        return max(self.probability_estimate) - self.confidence_radius(
            self.get_min_comparison()
        )


class BeatTheMeanBanditPAC(BeatTheMeanBandit):
    r"""The PAC variant of the Beat the Mean Bandit algorithm.

    The goal of this algorithm is to find the :term:`Condorcet winner`.

    It is assumed that a :term:`total order` over the arms exists. Additionally :math:`\gamma`-:term:`RST<Relaxed Stochastic Transitivity>` and the :term:`STI<Stochastic Triangle Inequality>` are assumed.

    In the :term:`PAC` setting (no time horizon given), the sample complexity is bound by :math:`O\left(\frac{N \gamma^6}{\epsilon^2} \log\frac{Nc}{\delta}\right)`.
    The constant :math:`c` is given as :math:`\left\lceil \frac{36}{\gamma^6\epsilon^2}\log \frac{N}{\delta}\right\rceil`.

    The :term:`PAC` setting for Beat the Mean Bandit algorithm takes an 'explore then exploit' approach. In :term:`PAC` setting,
    the exploration conditions for Beat the Mean Bandit algorithm is different from the Online setting. There are two
    termination cases for :term:`PAC` exploration, the first case is when the active set has been reduced to a single bandit.
    The second case is when the number of comparisons recorded for each remaining bandit is at least ``opt_n``
    (Corresponds to :math:`N'` in section 3.1.2 in :cite:`busa2018preference`). We do not use
    the time horizon in Beat-the-Mean :term:`PAC` setting (i.e., we set ``time_horizon = None``); it is used only in the online setting.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment. This parameter has been taken from the parent class.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    gamma
        The :math:`\gamma`-:term:`RST<Relaxed Stochastic Transitivity>` (corresponds to :math:`\gamma` in :cite:`yue2011beat`) that the
        algorithm should assume for the given problem setting. The value must be greater than :math:`0`.
        A higher value corresponds to a stronger assumption, where :math:`1` corresponds to :term:`STI<stochastic triangle inequality>`. In theory it is not possible to assume more than a gamma of :math:`1`, but in practice you can
        still specify higher values. This will lead to tighter confidence intervals and possibly better results,
        but the theoretical guarantees do not hold in that case. This parameter has been taken from the parent class.
    epsilon
        :math:`\epsilon` in (:math:`\epsilon, \delta`) :term:`PAC` algorithms, given by the user.
    failure_probability
        Allowed failure probability (corresponds to :math:`\delta` in Algorithm 2 in :cite:`yue2011beat`),
        i.e. probability that the actual value lies outside of the computed confidence interval. Derived from the
        Hoeffding bound.
    time_horizon
        Determines how many duels are executed in the online setting.

    Attributes
    ----------
    opt_n
        Corresponds to :math:`N'` in section 3.1.2 in :cite:`busa2018preference`.
    comparison_history
        A ComparisonHistory object which stores the history of the comparisons between the arms.
    random_state


    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5],
    ... ])
    >>> random_state = np.random.RandomState(43)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=random_state)
    >>> btm = BeatTheMeanBanditPAC(feedback_mechanism=feedback_mechanism, random_state=random_state, epsilon=0.001)
    >>> btm.run()
    >>> best_arm = btm.get_condorcet_winner()
    >>> best_arm
    2
    """

    # Disabling pylint errors because we are reimplementing the initialization since the superclass expects a time
    # horizon while it is optional for this class. For reference, take a look at
    # https://gitlab.com/duelpy/duelpy/-/merge_requests/77#note_448073174
    # pylint: disable=non-parent-init-called,super-init-not-called
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: np.random.RandomState = None,
        gamma: float = 1.0,
        epsilon: float = 0.01,
        failure_probability: float = 0.1,
    ):
        Algorithm.__init__(
            self, feedback_mechanism=feedback_mechanism, time_horizon=time_horizon
        )
        self.random_state = (
            np.random.RandomState() if random_state is None else random_state
        )
        # Corresponds to `N'` in section 3.1.2 in :cite:`busa2018preference`
        self.opt_n = np.ceil(
            36
            / (gamma ** 6 * epsilon ** 2)
            * np.log(self.feedback_mechanism.get_num_arms() / failure_probability)
        )

        prob_scaling = (self.feedback_mechanism.get_num_arms() ** 3) * self.opt_n

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability=failure_probability,
            factor=9 * (gamma ** 4) * 2,
            probability_scaling_factor=lambda x: prob_scaling,
        )
        self.comparison_history = ComparisonHistory(
            number_of_arms=self.feedback_mechanism.get_num_arms(),
            confidence_radius=confidence_radius,
        )

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        Returns
        -------
        bool
            Whether the algorithm is finished.
        """
        return (
            self.comparison_history.size_working_set <= 1
            or self.comparison_history.get_min_comparison() >= self.opt_n
        )
