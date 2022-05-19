"""Implementation of the knockout tournament algorithm."""

from itertools import combinations
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import CondorcetProducer
from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException


class KnockoutTournament(CondorcetProducer, PacAlgorithm):
    r"""Implementation of the knockout tournament algorithm.

    The goal of this algorithm is to find the :math:`\epsilon`-:term:`Condorcet winner` while minimizing the number of comparisons.

    The algorithm assumes a :term:`total order` over the existing arms and that :term:`strong stochastic transitivity`, :term:`stochastic triangle inequality` and :term:`relaxed stochastic transitivity` hold.

    The amount of pairwise comparisons made by the algorithm is bound  by :math:`\mathcal{O}\left(\frac{N}{\epsilon^2}\left(1+\log\frac{1}{\delta}\right)\right)`, where :math:`N` is the number of arms, :math:`\epsilon` the maximal deviation from the solution and :math:`\delta` is the error probability.

    The algorithm was originally introduced in :cite:`falahatgar2017maximum`. It is an :math:`\epsilon`-:math:`\delta`-:term:`PAC` algorithm. It takes the set of arms as an input and compares them in rounds. At the end of each round, the size of the input is halved.
    The winning arm for a round is decided based on the allowed sub-optimality :math:`\epsilon` and with a confidence interval based on the failure probability.
    The algorithm runs in rounds, where in each round it randomly pairs the arms into group and the winners are proceeded into the next round. For example that we have four arms [A, B, C, D]. It will first group the arms in pairs like [A, B] as the first pair and [C, D] as the second pair. After grouping them in pairs, the algorithm pulls out the winner from each pair, and the winners move to the next round.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
         The number of steps that the algorithm is supposed to be run.
    epsilon
        The optimality of the winning arm. Corresponds to :math:`\epsilon` in :cite:`falahatgar2017maximum`. Default value is ``0.05``
        which has been used in the experiments in :cite:`falahatgar2017maximum`.
    failure_probability
        The probability that the result is not an :math:`\epsilon`-:term:`Condorcet winner`. Corresponds to :math:`\delta` in :cite:`falahatgar2017maximum`.
        Default value is ``0.1`` which has been used in the experiments in :cite:`falahatgar2017maximum`.
    stochasticity
        The assumed stochastic transitivity parameter. Corresponds to :math:`\gamma` in :cite:`falahatgar2017maximum`. Default value is
        ``0.6`` which has been used in the experiments in :cite:`falahatgar2017maximum`.

    Attributes
    ----------
    feedback_mechanism
    tournament_arms
        The arms that are still in the tournament.
    epsilon
    failure_probability
    stochasticity
    time_step
         Number of rounds the algorithm has executed.

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
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=np.random.RandomState(100))
    >>> knockout_tournament = KnockoutTournament(feedback_mechanism, epsilon=0.1, failure_probability=0.3, stochasticity=0.6, time_horizon=300)
    >>> knockout_tournament.run()
    >>> best_arm = knockout_tournament.get_condorcet_winner()
    >>> best_arm
    2
    >>> feedback_mechanism.get_num_duels()
    300

    In this example the :math:`epsilon`-Condorcet winner is the arm with index 2.
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        epsilon: float = 0.05,
        failure_probability: float = 0.1,
        stochasticity: float = 0.6,
        base: float = 2.0,
        random_state: np.random.RandomState = None,
    ):
        self.tournament_arms = np.arange(feedback_mechanism.get_num_arms())
        self.stochasticity = stochasticity
        super().__init__(feedback_mechanism, time_horizon)
        self.time_step = 1
        self.epsilon = epsilon
        self.base = base
        self.failure_probability = failure_probability
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        # Confidence radius is derived using the Hoeffding's inequality and the Union bound, It's used
        # in selecting a challenger.
        # Refer page 20 of :cite:`falahatgar2017maximum` for confidence radius derivation.

        def probability_scaling(num_samples: int) -> float:
            return 4 * num_samples ** 2

        self.preference_estimate = PreferenceEstimate(
            num_arms=feedback_mechanism.get_num_arms(),
            confidence_radius=HoeffdingConfidenceRadius(
                failure_probability=self.failure_probability,
                probability_scaling_factor=probability_scaling,
            ),
        )

    def explore(self) -> None:
        """Execute one round of the algorithm.

        Arms are paired into groups, and each pairs of arms is compared repeatedly.

        """
        winning_arms = []

        current_epsilon = ((np.power(2, 1 / 3) - 1) * self.epsilon) / (
            self.stochasticity * np.power(self.base, self.time_step / 3)
        )

        current_failure_probability = self.failure_probability / np.power(
            self.base, self.time_step
        )

        self.random_state.shuffle(self.tournament_arms)
        num_arms = len(self.tournament_arms)
        try:
            for i in range(num_arms // 2):
                arm_i = self.tournament_arms[2 * i]
                arm_j = self.tournament_arms[2 * i + 1]
                winning_arms.append(
                    self._determine_winner(
                        arm_i, arm_j, current_epsilon, current_failure_probability
                    )
                )
            if (num_arms) % 2 == 1:
                winning_arms.append(self.tournament_arms[num_arms - 1])
        except AlgorithmFinishedException:
            return

        self.tournament_arms = winning_arms
        self.time_step += 1

    def get_condorcet_winner(self):
        return list(self.tournament_arms)[0]

    def get_winner(self):
        return self.get_condorcet_winner()

    def exploration_finished(self) -> bool:
        """Determine if the exploration is finished.

        The execution is finished when the time horizon is reached or when no time horizon was given and the :term:`Condorcet winner` has been found.

        Returns
        -------
        bool
            Whether the algorithm is finished.
        """
        return len(self.tournament_arms) == 1

    def _determine_winner(
        self,
        arm_i: int,
        arm_j: int,
        current_epsilon: float,
        current_failure_probability: float,
    ) -> int:
        """Determine the preferred arm by repeated comparison.

        Compare function output the winning arm, and the preferred arm proceed to the next round.

        Parameters
        ----------
        arm_i
            index of the first arm
        arm_j
            index of the second arm

        Raises
        ------
        AlgorithmFinishedException
            If the comparison budget is reached.


        Returns
        -------
        int
            The index of the winning arm
        """

        def probability_scaling(num_samples: int) -> float:
            return 4 * num_samples ** 2

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability=self.failure_probability,
            probability_scaling_factor=probability_scaling,
        )
        estimate_probability_arm_i = 0.5

        # In order to gain more confidence about the winning arm in each COMPARE, we repeat each
        # comparison several times. Comparison stops when it reaches the comparison budget and output the arm with more wins.
        comparison_budget = (1 / (2 * np.power(current_epsilon, 2))) * np.log(
            2 / current_failure_probability
        )
        rounds = 0
        # print ("current_epsilon:",  current_epsilon, "comparison_budget: ", comparison_budget)
        while (
            np.abs(estimate_probability_arm_i - 0.5)
            <= confidence_radius(rounds) - current_epsilon
            and rounds <= comparison_budget
        ):
            # update information about the preferred and eliminated arms
            self.preference_estimate.enter_sample(
                arm_j, arm_i, self.feedback_mechanism.duel(arm_i, arm_j)
            )
            rounds += 1
            estimate_probability_arm_i = self.preference_estimate.get_mean_estimate(
                arm_j, arm_i
            )
            if self.is_finished():
                raise AlgorithmFinishedException()

        return arm_j if estimate_probability_arm_i <= 0.5 else arm_i
