"""An implementation of the Relative Upper Confidence Bound (RCB) algorithm."""

from typing import Optional
from typing import Set

import numpy as np

from duelnlg.duelpy.algorithms.algorithm import Algorithm
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.util.utility_functions import argmax_set


class RelativeUCB(Algorithm):
    r"""Implementation of the Relative Upper Confidence Bound algorithm.

    The goal of this algorithm is to find the :term:`Condorcet winner` while incurring minimal regret.

    It is assumed that a :term:`Condorcet winner` exists.

    The regret bounds of this algorithm can be found in :cite:`zoghi2014dueling`.

    The algorithm is presented in :cite:`zoghi2014dueling`.
    RUCB is an extension of the Upper Confidence Bound (UCB) algorithm for regular multi-armed bandits. It is motivated by learning from the relative feedback rather than real valued feedback between two arms. It works for both finite as well as for infinite time horizons. The major goals of this algorithm are to minimize cumulative regret over time for the K-armed dueling bandit problem and also return a :term:`Condorcet winner`.

    In each time-step RUCB executes three sub-parts sequentially:

    - Initially, assume all arms as a potential champion. All arms are compared in pairwise optimistically fashion using upper confidence bound. If the upper confidence bound of an arm against any other arm is less than :math:`0.5`, then that "loser" is removed from the potential champions. This process keeps on and when we are left with only one arm in the pool then that arm is assigned as the hypothesized best arm. There is always at most one hypothesized best arm. This hypothesized best arm :math:`B` is demoted from its status as soon as it loses to another arm and from the remaining potential champions arm, a potential champion arm :math:`arm_c` is chosen in two ways: if :math:`B` is not present, we sample an arm uniformly randomly. If :math:`B` is present, the probability of picking the arm :math:`B` is set to :math:`\frac{1}{2}` and the remaining arms are given equal probability for being chosen.

    - Regular UCB is performed using potential champion :math:`arm_c` as a benchmark. Now, we select challenger arm :math:`arm_d` (distinct from :math:`arm_c`) whose upper confidence bound is maximal with reference to the potential champion :math:`arm_c`.

    - Now the potential champion :math:`arm_c` and challenger arm :math:`arm_d` are compared. Based on the comparison, the winner arm is decided and the win count is updated. At last, the :term:`Condorcet winner` is returned as the arm whose winning count is maximum.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        How many comparisons the algorithm should do. This does not impact the
        decision of the algorithm, only for how many steps ``run`` executes.
        May be ``None`` to indicate a unknown or infinite time horizon.
    exploratory_constant
        Optional, The confidence radius grows proportional to the square root of this value. Corresponds to :math:`\alpha` in
        :cite:`zoghi2014dueling`. The value of ``exploratory_constant`` must be greater than :math:`0.5`. The default value is ``0.51``.
    random_state
        Optional, used for random choices in the algorithm.

    Attributes
    ----------
    preference_estimate
        Estimation of a preference matrix based on samples.
    feedback_mechanism
    exploratory_constant
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> from duelnlg.duelpy.stats.metrics import WeakRegret
    >>> from duelnlg.duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(43)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"weak_regret": WeakRegret(preference_matrix)}
    ... )
    >>> test_object = RelativeUCB(
    ...     feedback_mechanism=feedback_mechanism,
    ...     time_horizon=100,
    ...     exploratory_constant=0.6,
    ...     random_state=random_state,
    ... )
    >>> test_object.run()
    >>> test_object.get_champion()
    2
    >>> np.round(np.sum(feedback_mechanism.results["weak_regret"]), 2)
    5.6
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        exploratory_constant: float = 0.51,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.exploratory_constant = exploratory_constant
        self.time_step = 0
        # Each step in the code refers to lines of RUCB algorithm presented in the paper.
        # Step 1: Initialization of count of wins between each arm
        self.preference_estimate = PreferenceEstimate(
            num_arms=feedback_mechanism.get_num_arms()
        )
        # Step 2: Initialization hypothesized best arm
        self.hypothesized_arm: Optional[int] = None
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

    # pylint: disable=duplicate-code
    def _update_confidence_radius(self) -> None:
        r"""Update the confidence radius using latest failure probability.

        Failure probability for the upper confidence bound is :math:`1/t^(2 * \alpha)`
        where :math:`t` is the current round of the algorithm and :math:`\alpha` is the
        exploratory constant.

        Refer :cite:`zoghi2014dueling` for further details.
        """
        failure_probability = 1 / (self.time_step ** (2 * self.exploratory_constant))
        confidence_radius = HoeffdingConfidenceRadius(failure_probability)
        self.preference_estimate.set_confidence_radius(confidence_radius)

    def _select_potential_winner(self, potential_arms: Set[int]) -> int:
        """Select a potential winner form the set of potential winners.

        Parameters
        ----------
        potential_arms
           Set of potential winner arms.

        Returns
        -------
        int
            Selected arm from the given set of potential winners.
        """
        if len(potential_arms) == 1:
            self.hypothesized_arm = list(potential_arms)[0]
            # arm_c be the unique element of potential arm set
            arm_c = self.hypothesized_arm

        # Step 10: if more than one potential arm
        if len(potential_arms) > 1:
            # probability distribution for list of arms
            probability_distribution = dict()
            if self.hypothesized_arm is not None:
                probability_distribution[self.hypothesized_arm] = 0.5
                potential_arms = potential_arms - {self.hypothesized_arm}

            for potential_arm in potential_arms:
                # distribute probability equal to other arms
                probability_distribution[potential_arm] = 1 / (
                    np.power(2, 0 if self.hypothesized_arm is None else 1)
                    * len(potential_arms - {self.hypothesized_arm})
                )
            #   Step 11:sample arm_c from probability distribution
            arm_c = self.random_state.choice(
                list(probability_distribution.keys()),
                p=list(probability_distribution.values()),
            )
        return arm_c

    def get_champion(self) -> int:
        """Get the champion arm that has won more often than any other arm.

        Returns
        -------
        int
            Champion arm at time-step :math:`T`.
        """
        arms_win_count = np.zeros(self.feedback_mechanism.get_num_arms())
        # calculate number of expected wins for each arm
        for arm_i in range(self.feedback_mechanism.get_num_arms()):
            for arm_j in range(self.feedback_mechanism.get_num_arms()):
                if self.preference_estimate.get_mean_estimate(arm_i, arm_j) > 0.5:
                    arms_win_count[arm_i] += self.preference_estimate.wins[arm_i, arm_j]

        winner_arm = int(np.argmax(arms_win_count))
        return winner_arm

    def get_winner(self) -> Optional[int]:

        return list(
            self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )[0]

    def step(self) -> None:
        """Run one round of an algorithm."""
        self.time_step += 1
        self._update_confidence_radius()
        # Step 4: computation for upper confidence bound matrix U[i,j]
        #   µ[i,j]
        upper_confidence_bound_matrix = (
            self.preference_estimate.get_upper_estimate_matrix()
        )
        # Step 6: compute potential champion of arms
        potential_arms = set()
        for arm_i in range(self.feedback_mechanism.get_num_arms()):
            could_be_winner = True  # selected arm wins all other arm or not
            for arm_j in range(self.feedback_mechanism.get_num_arms()):
                if upper_confidence_bound_matrix[arm_i][arm_j] < 0.5:
                    could_be_winner = False
            if could_be_winner:
                potential_arms.add(arm_i)

        # Step 7: potential set is empty
        if len(potential_arms) == 0:
            arm_c = self.random_state.choice(
                range(self.feedback_mechanism.get_num_arms())
            )
            potential_arms.add(arm_c)

        # Step 8: hypothesized arm is no longer a potential winner
        if (
            self.hypothesized_arm is not None
            and self.hypothesized_arm not in potential_arms
        ):
            self.hypothesized_arm = None

        # Step 9-12: select an potential champion arm(arm_c) from potential arms
        arm_c = self._select_potential_winner(potential_arms)

        # Step 13: selection of challenger arm(arm_d) where potential champion and challenger should not be same
        # Select the upper confidences of all the arm with respect to potential champion(arm_c)
        upper_confidences = upper_confidence_bound_matrix[:, arm_c]
        # Select challenger arm - arm_d(other than arm_c) whose upper confidence bound is maximum with reference to arm_c.
        arm_d = self.random_state.choice(argmax_set(upper_confidences, [arm_c]))

        # Step 14: Compare potential champion(arm_c) and challenger(arm_d).
        self.preference_estimate.enter_sample(
            arm_c, arm_d, self.feedback_mechanism.duel(arm_c, arm_d)
        )
