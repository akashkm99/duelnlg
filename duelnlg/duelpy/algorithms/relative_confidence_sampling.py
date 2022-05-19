"""An implementation of the Relative Confidence Sampling algorithm."""
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import CondorcetProducer
from duelnlg.duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius


class RelativeConfidenceSampling(CondorcetProducer):
    r"""Implementation of the Relative Confidence Sampling algorithm.

    The goal of this algorithm is to find a :term:`Condorcet winner` while incurring minimal regret.

    It is assumed that a :term:`Condorcet winner` exists.

    No regret or sample complexity bounds were established in the source paper :cite:`zoghi2014ranker`.

    The algorithm proceeds by conducting duels among the candidate arms and eliminating the sub-optimal
    arms based on the results of the prior duels. After conducting sufficient rounds, the
    algorithm would always choose the :term:`Condorcet winner` to conduct a duel with itself. This
    would result in no more regret and thus the goal would be achieved.

    Relative Confidence Sampling works continuously in the following 3 steps:

    1. A simulated tournament is conducted among all the arms to obtain a champion.
    The tournament is based on sampling from a beta distribution. The beta distribution
    is parameterized on the results of the prior duels among the competing arms.
    The :term:`Condorcet winner` of this simulated tournament is selected as the current champion.
    If no :term:`Condorcet winner` exists, the arm that has been selected as the champion the least
    number of times is the new champion.

    2. A challenger which has the highest chance of winning against the current champion
    is obtained using the upper confidence bound. The upper confidence bound is based on the
    probability estimates of winning against the champion. These probability estimates can be
    derived for every arm using the results of the prior duels against the champion.

    3. A duel is conducted among the champion and the challenger and the results are stored.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    time_horizon
        How many comparisons the algorithm should do. This does not impact the
        decision of the algorithm, only for how many times ``step`` executes.
        May be ``None`` to indicate a unknown or infinite time horizon.
    exploratory_constant
        A parameter which is used in calculating the upper confidence bounds. The confidence
        radius grows proportional to the square root of this value. A higher upper confidence
        bound results in more exploration.
        Corresponds to :math:`\alpha` in :cite:`zoghi2014ranker`. The value of ``exploratory_constant`` must be greater than :math:`0.5`.
        Default value is ``0.501`` which has been used in the experiments related to RCS in :cite:`zoghi2014ranker`.

    Attributes
    ----------
    feedback_mechanism
    exploratory_constant
    random_state
    time_step
        Number of rounds the algorithm has executed.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> from duelnlg.duelpy.stats.metrics import WeakRegret
    >>> from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix
    >>> from duelnlg.duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> arms = list(range(len(preference_matrix)))
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"weak_regret": WeakRegret(preference_matrix)}
    ... )
    >>> rcs = RelativeConfidenceSampling(feedback_mechanism=feedback_mechanism, time_horizon=100, exploratory_constant=0.6, random_state=random_state)
    >>> rcs.run()

    The best arm in this case is the last arm (index 2)

    >>> np.round(np.sum(feedback_mechanism.results["weak_regret"]), 2)
    0.8
    >>> rcs.get_condorcet_winner()
    2
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: Optional[np.random.RandomState] = None,
        exploratory_constant: float = 0.501,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        if exploratory_constant <= 0.5:
            raise ValueError("Value of exploratory constant must be greater than 0.5")
        self.exploratory_constant = exploratory_constant
        self.time_step = 0
        self.preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms()
        )
        # Number of times each arm was already chosen as the champion
        self._champion_chosen_frequency = np.zeros(
            self.feedback_mechanism.get_num_arms(), dtype=int
        )

    def step(self) -> None:
        """Run one round of the algorithm."""
        self.time_step += 1
        # Update and set the new confidence radius in `preference_estimate` as per
        # the current `time_step`
        self._update_confidence_radius()

        champion = self._run_simulated_tournament()
        challenger = self._select_challenger_for(champion)
        arm1, arm2, score = self.feedback_mechanism.get_duel(champion, challenger)
        # Enter the duel result
        self.preference_estimate.enter_sample(arm1, arm2, score)

    def get_condorcet_winner(self) -> Optional[int]:
        """Determine a Condorcet winner using RCS algorithm.

        Returns
        -------
        Optional[int]
            The index of a Condorcet winner, if existent, among the given arms.
        """
        return (
            self.preference_estimate.get_mean_estimate_matrix().get_condorcet_winner()
        )

    def get_winner(self) -> Optional[int]:

        return list(
            self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )[0]

    def _update_confidence_radius(self) -> None:
        r"""Update the confidence radius using latest failure probability.

        Failure probability for the upper confidence bound is :math:`1/t^(2 * \alpha)`
        where :math:`t` is the current round of the algorithm and :math:`\alpha` is the
        exploratory constant.

        Refer https://arxiv.org/pdf/1312.3393.pdf for further details.
        """
        failure_probability = 1 / (self.time_step ** (2 * self.exploratory_constant))
        confidence_radius = HoeffdingConfidenceRadius(failure_probability)
        self.preference_estimate.set_confidence_radius(confidence_radius)

    def _run_simulated_tournament(self) -> int:
        """Run a simulated tournament among all arms to pick the champion.

        Returns the Condorcet winner of the tournament, if existent, as the
        champion. A Condorcet winner is an arm whose probability of beating
        every other arm in the tournament is at least 0.5.
        If no Condorcet winner exists, then the arm which was least chosen
        as a champion is returned as the current champion.

        Returns
        -------
        int
            The champion of the tournament.
        """
        sampled_preference_matrix = self.preference_estimate.sample_preference_matrix(
            self.random_state
        )
        condorcet_winner = sampled_preference_matrix.get_condorcet_winner()

        champion = (
            condorcet_winner
            if condorcet_winner is not None
            else np.argmin(self._champion_chosen_frequency)
        )
        self._champion_chosen_frequency[champion] += 1

        return champion

    def _select_challenger_for(self, champion: int) -> int:
        """Pick the appropriate challenger for the selected champion.

        Determine the challenger that has the highest probability of winning against
        the champion based on the upper confidence bound on the probability estimates.

        Parameters
        ----------
        champion
            The arm which is selected as the champion.

        Returns
        -------
        int
            The arm which is selected as the challenger to the champion.
        """
        upper_confidence_bounds = [
            self.preference_estimate.get_upper_estimate(arm_j, champion)
            for arm_j in range(self.feedback_mechanism.get_num_arms())
        ]
        challenger = np.argmax(upper_confidence_bounds)
        return challenger
