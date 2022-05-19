"""An implementation of KL-divergence based PAC algorithm."""
import math
from typing import Optional

import numpy as np
from scipy.special import rel_entr

from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.algorithms.interfaces import SingleCopelandProducer
from duelnlg.duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.util.utility_functions import newton_raphson
from duelnlg.duelpy.util.utility_functions import pop_random


class KLDivergenceBasedPAC(SingleCopelandProducer, PacAlgorithm):
    r"""Implement the KL-divergence based PAC algorithm.

    The goal of the algorithm is to find a :term:`Copeland winner` in a
    :term:`PAC` setting i.e. the algorithm finds an :math:`\epsilon`-Copeland
    winner with probability at least :math:`1 - \delta`.

    It is assumed that there are no ties between arms, i.e. the
    probability of any arm winning against another arm is never :math:`\frac{1}{2}`.

    The bound on the expected regret is given as
    :math:`\mathcal{O}\left( \ln(\frac{N}{\delta\Delta_i\epsilon} \frac{1 - \mu_i}{(\Delta_i^\epsilon)^2} \right)`. :math:`N` is the number of arms,
    :math:`\epsilon` is the error parameter and :math:`\delta` is the
    failure probability.
    :math:`\mu_i` is the expected reward of arm :math:`a_i`. Let
    :math:`a_1` be the arm which generates the maximum reward
    :math:`\mu_1`.
    :math:`\Delta_i = \max(\mathit{cpld}(a_1)-\mathit{cpld}(a_i), \frac{1}{N-1})`
    where :math:`\mathit{cpld}(a_i)` is the Copeland score of arm
    :math:`a_i`.
    :math:`\Delta_i^\epsilon = \max(\Delta_i, \epsilon(1 - \mathit{cpld}(a_1)))`

    This was originally introduced as a component of the :class:`ScalableCopelandBandits<duelpy.algorithms.scalable_copeland_bandits.ScalableCopelandBandits>` algorithm described in :cite:`zoghi2015copeland`.
    This implementation is based on `Algorithm 2` (which further uses
    `Algorithm 4`) from the same paper. This algorithm uses
    KL-Divergence in the process of finding an approximate :term:`Copeland winner`.
    An additional condition is used to terminate the exploration
    phase. This additional condition checks whether there is only one :term:`Copeland winner`
    candidate left. If yes, then the exploration phase is stopped.
    This is done to ensure that the KL divergence based algorithm begins with
    the exploitation phase (i.e. dueling a :term:`PAC`-:term:`Copeland winner` against itself)
    earlier in some cases. Otherwise, it might continue to duel the one remaining
    candidate against random opponents for a while, which could incur a higher
    regret in the primary algorithm :class:`ScalableCopelandBandits<duelpy.algorithms.scalable_copeland_bandits.ScalableCopelandBandits>`.

    The algorithm finds a :term:`Copeland winner` based on the smallest and greatest
    probability distribution that has a low KL-Divergence (less than or
    equal to the value of :math:`\ln\left(\frac{4tN}{\delta}\right) + 2 \ln \ln(t)` in `Algorithm 4`
    in the paper) to the estimated preference probabilities.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        Number of time steps to execute for. Corresponds to
        :math:`T` in :cite:`zoghi2015copeland`. Defaults to
        ``None`` when not specified.
    epsilon
        The optimality of the winning arm. Corresponds to :math:`\epsilon`
        in :cite:`zoghi2015copeland`.
    failure_probability
        Upper bound on the probability of failure. Corresponds to
        :math:`\delta` in :cite:`zoghi2015copeland`.
    random_state
        A numpy random state. Defaults to an unseeded state when not
        specified.
    preference_estimate
        A ``PreferenceEstimate`` object is needed if this algorithm is used
        as a subroutine and the result is required to be stored in further
        rounds. The confidence radius of the given preference estimate will be
        overridden. Pass ``None`` if the algorithm should start from a new
        preference estimate. Defaults to ``None``.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> arms = list(range(len(preference_matrix)))
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, arms, random_state)
    >>> kldpac = KLDivergenceBasedPAC(feedback_mechanism=feedback_mechanism,
    ...     epsilon=0,failure_probability=0.05,random_state=random_state)
    >>> kldpac.run()

    The best arm in this case is the last arm (index 2)

    >>> kldpac.feedback_mechanism.get_num_duels()
    89
    >>> kldpac.get_copeland_winner()
    2
    """

    # Disabling pylint errors as the attributes are required
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        epsilon: float = 0.51,
        failure_probability: float = 0.05,
        preference_estimate: Optional[PreferenceEstimate] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.epsilon = epsilon
        self.failure_probability = failure_probability
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.preference_estimate = (
            preference_estimate
            if preference_estimate is not None
            else PreferenceEstimate(self.feedback_mechanism.get_num_arms())
        )
        confidence_radius = HoeffdingConfidenceRadius(
            self.failure_probability / (self.feedback_mechanism.get_num_arms() ** 2)
        )
        self.preference_estimate.set_confidence_radius(confidence_radius)

        self.copeland_winner = None

        # Refer "Algorithm 4" in :cite:`zoghi2015copeland` for details.
        self.copeland_winner_candidates = np.arange(
            self.feedback_mechanism.get_num_arms()
        )
        self.rewards_for_candidates = np.zeros(
            self.feedback_mechanism.get_num_arms(), dtype=int
        )
        self.no_of_times_rewards_determined_for_candidates = np.full(
            self.feedback_mechanism.get_num_arms(), 2, dtype=int
        )

        # Refer "Algorithm 4" in :cite:`zoghi2015copeland` for details.
        # Initialize interval boundary values for each of the Copeland winner
        # candidates. Assuming the initial boundary values as [1.48e-8, 1 - 1.48e-8]
        # instead of [0,1] to overcome undefined values in further calculations.
        # The value ``1.48e-8`` is taken from the parameter ``tol`` of the ``newton``
        # function of ``scipy``. Refer the following link for details:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
        self.left_boundary_for_candidates = np.full(
            self.feedback_mechanism.get_num_arms(), (1.48e-8), dtype=float
        )
        self.right_boundary_for_candidates = np.full(
            self.feedback_mechanism.get_num_arms(), (1 - 1.48e-8), dtype=float
        )

    def get_copeland_winner(self) -> Optional[int]:
        """Get the Copeland winner.

        Returns
        -------
        Optional[int]
            The index of the estimated Copeland winner.
            ``None``, if a winner could not be determined.
        """
        return self.copeland_winner if self.exploration_finished() else None

    def exploration_finished(self) -> bool:
        """Determine whether exploration is finished.

        This termination condition is based on the condition that is used in
        Algorithm 4 in :cite:`zoghi2015copeland`. In addition to that, it is
        also checked whether there exists only one :term:`Copeland winner` candidate
        in order to end the exploration phase and start with the exploitation
        phase as per the explore-then-exploit principle.

        Returns
        -------
        bool
            ``True`` if the exploration should stop/has been stopped,
            ``False`` otherwise.
        """
        try:
            return (1 - np.amax(self.left_boundary_for_candidates)) / (
                1 - np.amax(self.right_boundary_for_candidates)
            ) <= (1 + self.epsilon) or self.copeland_winner_candidates.size == 1
        except ZeroDivisionError:
            pass

        return False

    def explore(self) -> None:
        """Explore the given set of arms to obtain a Copeland winner."""
        self._determine_rewards_for_copeland_winner_candidates()
        self._calculate_interval_boundaries_for_copeland_winner_candidates()
        self._remove_candidates()
        self.no_of_times_rewards_determined_for_candidates += 1

        self.copeland_winner = self.copeland_winner_candidates[
            np.argmax(self.left_boundary_for_candidates)
        ]

    def _determine_rewards_for_copeland_winner_candidates(self) -> None:
        """Determine rewards (0 or 1) for Copeland winner candidates."""
        # Calculate and add reward for each arm
        for index in range(np.size(self.copeland_winner_candidates)):
            # Check whether further reward generation is necessary
            if self.is_finished():
                return
            self.rewards_for_candidates[index] += self._reward_for(
                self.copeland_winner_candidates[index]
            )

    def _reward_for(self, arm: int) -> int:
        r"""Generate a reward for the given arm.

        A random opponent is chosen against the given arm for dueling.
        Then duels are conducted until either the lower bound of the
        preference probability of the arm against the opponent is above
        0.5 or the upper bound of the same is below 0.5. Finally, a
        reward of 0 or 1 is generated which is based upon the preference
        probability of the arm against the opponent.

        Parameters
        ----------
        arm
            Arm whose reward is to be generated.

        Returns
        -------
        int
            0 or 1 depending upon the chance of the arm winning against
            a random opponent in a duel.
        """
        # Sample a random opponent from a uniform distribution on the participating
        # arms. Opponent must be different than the provided arm.
        arm_candidates = self.feedback_mechanism.get_arms()
        arm_candidates.remove(arm)
        opponent = pop_random(arm_candidates, self.random_state)[0]

        while not (
            self.preference_estimate.get_lower_estimate(arm, opponent) > 0.5
            or self.preference_estimate.get_upper_estimate(arm, opponent) < 0.5
            or self.is_finished()
        ):
            arm_won = self.feedback_mechanism.duel(arm, opponent)
            self.preference_estimate.enter_sample(arm, opponent, arm_won)

        arm_win_estimate = self.preference_estimate.get_mean_estimate(arm, opponent)
        if arm_win_estimate > 0.5:
            return 1

        return 0

    def _calculate_interval_boundaries_for_copeland_winner_candidates(self) -> None:
        r"""Calculate interval boundaries for the Copeland winner candidates.

        New interval boundaries are calculated for the Copeland winner candidates.
        This is done using Newton-Raphson method of finding the approximate roots
        of a function. Refer to "https://en.wikipedia.org/wiki/Newton's_method" for
        details. The ``function_to_be_solved`` for which roots need to be found
        consists of the Chernoff’s inequality stated w.r.t the KL-divergence of two
        random variables. For two Bernoulli random variables with parameters :math:`p`,
        :math:`q` the KL-divergence from :math:`q` to :math:`p` is defined as
        :math:`d(p, q) = (1 − p) \ln(\frac{(1 − p)}{(1 − q)}) + p \ln(\frac{p}{q}) `
        with 0 \ln(0) = 0.
        """
        no_of_times_arm_played: int = 0
        avg_reward: float = 0.0

        for index in range(np.size(self.copeland_winner_candidates)):
            no_of_times_arm_played = self.no_of_times_rewards_determined_for_candidates[
                index
            ]
            avg_reward = self.rewards_for_candidates[index] / no_of_times_arm_played

            def function_to_be_solved(point: float) -> float:
                if point <= 0 or point >= 1:
                    return math.inf
                kl_divergences = rel_entr(1 - avg_reward, 1 - point) + rel_entr(
                    avg_reward, point
                )

                return (
                    kl_divergences
                    - (
                        np.log(
                            4
                            * np.sum(self.no_of_times_rewards_determined_for_candidates)
                            * self.feedback_mechanism.get_num_arms()
                            / self.failure_probability
                        )
                        + (
                            2
                            * np.log(
                                np.log(
                                    np.sum(
                                        self.no_of_times_rewards_determined_for_candidates
                                    )
                                )
                            )
                        )
                    )
                    / no_of_times_arm_played
                )

            def derivative_of_function_to_be_solved(point: float) -> float:
                if point <= 0 or point >= 1:
                    return math.inf
                return (point - avg_reward) / (point * (1 - point))

            left_boundary = newton_raphson(
                self.left_boundary_for_candidates[index],
                function_to_be_solved,
                derivative_of_function_to_be_solved,
            )
            if 0 < left_boundary < avg_reward:
                self.left_boundary_for_candidates[index] = left_boundary

            right_boundary = newton_raphson(
                self.right_boundary_for_candidates[index],
                function_to_be_solved,
                derivative_of_function_to_be_solved,
            )
            if avg_reward < right_boundary < 1:
                self.right_boundary_for_candidates[index] = right_boundary

    def _remove_candidates(self) -> None:
        """Drop the arms which cannot become Copeland winners."""
        candidates_to_be_removed = np.argwhere(
            self.right_boundary_for_candidates
            < np.amax(self.left_boundary_for_candidates)
        )

        self.copeland_winner_candidates = np.delete(
            self.copeland_winner_candidates, candidates_to_be_removed
        )
        self.rewards_for_candidates = np.delete(
            self.rewards_for_candidates, candidates_to_be_removed
        )
        self.no_of_times_rewards_determined_for_candidates = np.delete(
            self.no_of_times_rewards_determined_for_candidates, candidates_to_be_removed
        )
        self.left_boundary_for_candidates = np.delete(
            self.left_boundary_for_candidates, candidates_to_be_removed
        )
        self.right_boundary_for_candidates = np.delete(
            self.right_boundary_for_candidates, candidates_to_be_removed
        )
