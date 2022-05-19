"""PAC best arm selection with the SAVAGE algorithm."""


from typing import Optional
from typing import Tuple

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.algorithms.interfaces import SingleCopelandProducer
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius


class Savage(SingleCopelandProducer, PacAlgorithm):
    r"""Determine the PAC-best arm with the SAVAGE algorithm.

    This algorithm makes no assumptions about the environment.

    The sample complexity is bounded by :math:`\sum_{i=1}^N \mathcal{O}\left(\frac{\log\left(\frac{N}{\delta\Delta_i}\right)}{\Delta_i^2}\right)` if the time horizon :math:`T` is finite and :math:`\sum_{i=1}^N \mathcal{O}\left(\frac{\log\left(\frac{NT}{\delta}\right)}{\Delta_i^2}\right)` otherwise.

    SAVAGE is a general algorithm that can infer some information about an
    environment from samples. It works by repeatedly sampling possible
    environments (in the case of PB-MAB, an environment is specified by a
    preference matrix) and eliminating

    - those environment candidates that fall outside of the current confidence
      interval (for example the sets of preference matrices that would make our
      previous samples too unlikely) and
    - those environment variables (preference matrix entries) that are no
      longer relevant on the current environment candidates (for example the
      arms that cannot be the :term:`Copeland winner`). See Figure 1 in :cite:`urvoy2013generic` for an
      illustration. In this case :math:`\mu` is the preference matrix while
      :math:`x_1` and :math:`x_2` are two entries of the matrix (without loss
      of generality it is sufficient to estimate the upper-right triangle of
      the matrix). If we already know that arm i is strictly better than arm j,
      it is no longer necessary to test arm i and we can stop trying to improve
      our estimate on :math:`q_{ik}`.

    Environment parameters in the PB-MAB case are the upper triangle of the preference matrix.
    The goal goal is to design a sequence of pairwise experiments (samples of
    random variables) / duels to find the best arm (according to ranking
    procedure). This is called `voting bandits` since we use a pairwise election
    criterion to find the best bandit, meaning `beating` for :term:`Copeland<Copeland winner>`, or `better
    expectation` for a :term:`Borda<Borda winner>`.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    failure_probability
        Upper bound on the probability of failure (the :math:`\delta` in
        (:math:`\epsilon`,:math:`\delta`)-PAC).
    time_horizon
        The number of steps that the algorithm is supposed to be run. Specify
        ``None`` for an infinite time horizon.

    Attributes
    ----------
    feedback_mechanism
    failure_probability
    preference_estimate
        The current estimate of the preference matrix.

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
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=np.random.RandomState(42))

    Obviously, the last arm (index 2) is expected to win against the most other
    arms. That makes it the Copeland winner, as SAVAGE is correctly able to
    determine:

    >>> algorithm = Savage(feedback_mechanism)
    >>> algorithm.run()
    >>> algorithm.get_copeland_winner()
    2
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        failure_probability: float = 0.1,
        time_horizon: Optional[int] = None,
        winner_type="copland",
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.failure_probability = failure_probability
        self.winner_type = winner_type

        # The number of random variables that we attempt to estimate
        # (corresponds to the upper triangle of the preference matrix).
        num_arms = self.feedback_mechanism.get_num_arms()
        num_random_variables = num_arms * (num_arms - 1) / 2

        # The failure probability of each individual confidence interval must be
        # scaled appropriately so that the probability that *any* estimate fails is
        # sufficiently low (as set by failure_probability). That is achieved by a
        # naive Union bound. See page 10 of
        # http://proceedings.mlr.press/v28/urvoy13-supp.pdf for a detailed
        # derivation of this bound. Intuitively, we scale the allowed failure
        # probability down proportional the number of random variables we are
        # estimating. If the time horizon is unknown (infinite-horizon case) we
        # additionally scale by the square of the current sample to make sure
        # the infinite sum converges. Otherwise scale by the time horizon.
        def union_bound_scaling_factor(num_samples: int) -> float:
            if self.time_horizon is None:
                return (np.pi ** 2 * num_random_variables * num_samples ** 2) / 3
            else:
                return 2 * num_random_variables * self.time_horizon

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability, probability_scaling_factor=union_bound_scaling_factor
        )

        # Estimate the preference matrix based on past samples. Keeps track of the
        # t_i and \hat\mu_i variables in the paper.
        self.preference_estimate = PreferenceEstimate(
            feedback_mechanism.get_num_arms(), confidence_radius
        )

        # Initialize with all possible arm pairings, without loss of generality the
        # first arm has the lower index. This maintains a list of all pairwise win
        # probabilities we are not sufficiently sure about yet, i.e. which may
        # still influence our result. Corresponds to the W set array in the
        # reference paper.
        self._relevant_arm_combinations = {
            (i, i + j) for i in range(num_arms) for j in range(1, num_arms - i)
        }
        self.num_arms = num_arms

    def copeland_independence_test(self, arm_pair: Tuple[int, int]) -> bool:
        """Test if the result of a duel can still influence our estimate of the Copeland winner.

        This corresponds to the "IndepTest" in the paper.

        Parameters
        ----------
        arm_pair
            The pair of arms in question.

        Returns
        -------
        bool
            False if more information about the arm pair is still needed. True
            if the Copeland estimation is not dependant on further information.
        """
        most_certain_wins = np.max(
            self.preference_estimate.get_pessimistic_copeland_score_estimates()
        )
        # Set of viable hypotheses is represented implicitly by a set of confidence
        # intervals.
        (lower_bound, upper_bound) = self.preference_estimate.get_confidence_interval(
            *arm_pair
        )
        if lower_bound > 1 / 2 or upper_bound < 1 / 2:
            # We already know which arm is expected to win. How probable
            # its win is is not important for the Copeland score.
            return True

        possible_wins = (
            self.preference_estimate.get_optimistic_copeland_score_estimates()
        )
        # Compute optimistic estimates for the arm pair.
        for arm in arm_pair:
            if possible_wins[arm] > most_certain_wins:
                return False

        return True

    def condorcet_independence_test(self, arm_pair: Tuple[int, int]) -> bool:

        """Test if the result of a duel can still influence our estimate of the Condorcet winner.

        This corresponds to the "IndepTest" in the paper.

        Parameters
        ----------
        arm_pair
            The pair of arms in question.

        Returns
        -------
        bool
            False if more information about the arm pair is still needed. True
            if the Condorcet estimation is not dependant on further information.
        """
        # Set of viable hypotheses is represented implicitly by a set of confidence
        # intervals.
        (lower_bound, upper_bound) = self.preference_estimate.get_confidence_interval(
            *arm_pair
        )
        if lower_bound > 1 / 2 or upper_bound < 1 / 2:
            # We already know which arm is expected to win. How probable
            # its win is is not important for the Copeland score.
            return True

        possible_wins = (
            self.preference_estimate.get_optimistic_copeland_score_estimates()
        )
        # Compute optimistic estimates for the arm pair.
        if (possible_wins[arm_pair[0]] < self.num_arms - 1) and (
            possible_wins[arm_pair[1]] < self.num_arms - 1
        ):
            return True

        return False

    def independence_test(self, arm_pair: Tuple[int, int]) -> bool:

        if self.winner_type == "copland":
            return self.copeland_independence_test(arm_pair)

        elif self.winner_type == "condorcet":
            return self.condorcet_independence_test(arm_pair)

        else:
            raise NotImplementedError(
                "Winner types of copland and condorcet winner is only implemented"
            )

    def explore(self) -> None:
        """Run one step of exploration."""
        # Find the next arm to sample. This could probably be optimized by choosing
        # a better data structure, but I'm trying to keep it simple and relatively
        # close to the paper for now.
        next_sample = None
        current_lowest_sample_count = np.infty
        arms_to_remove = set()
        for arm_pair in self._relevant_arm_combinations:
            if (
                self.preference_estimate.get_num_samples(*arm_pair)
                < current_lowest_sample_count
            ):
                if not self.copeland_independence_test(arm_pair):
                    next_sample = arm_pair
                    current_lowest_sample_count = self.preference_estimate.get_num_samples(
                        *arm_pair
                    )
                else:
                    arms_to_remove.add(arm_pair)

        if next_sample is not None:
            # Sample a duel and keep track of the results.
            self.preference_estimate.enter_sample(
                *next_sample, self.feedback_mechanism.duel(*next_sample)
            )

        # According to the algorithm in the paper, we should always check *all*
        # remaining candidate pairs after making a sample and prune the list of
        # remaining candidates.

        # We do it slightly differently here: we only do the "independence
        # test" and remove arms when they would otherwise have been a candidate
        # for exploration in this step. That leads to the same order of arm
        # exploration, but it reduces the number of necessary checks and
        # spreads the computation cost more evenly among the time steps.
        self._relevant_arm_combinations.difference_update(arms_to_remove)

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        If no time horizon is provided, this coincides with is_finished. Once
        this function returns ``True``, the algorithm will have finished
        computing a :term:`PAC` :term:`Copeland winner`.
        """
        # When making the Condorcet assumption, the termination condition could be
        # replaced by one allowing for an epsilon-approximation. See Section 4.1.2
        # in the reference paper.
        return len(self._relevant_arm_combinations) == 0

    def get_copeland_winner(self) -> Optional[int]:
        r"""Find a Copeland winner with the SAVAGE algorithm.

        Note that only the correctness of any one of the :term:`Copeland winners<Copeland winner>` is
        covered by the failure probability. The probability that all arms in
        the set are actually :term:`Copeland winners<Copeland winner>` is lower. We still return the
        full set of arms for convenience.

        Returns
        -------
        Set[int]
            The indices of the :math:`\delta`-PAC best (Copeland) arms. The :math:`\delta`
            failure probability refers to any individual arm, but not all arms
            together.
        """
        # if not self.exploration_finished():
        #     return None
        return list(
            self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )[0]

    def get_winner(self) -> Optional[int]:

        if self.winner_type == "copland" or self.winner_type == "condorcet":
            return self.get_copeland_winner()
        else:
            raise NotImplementedError(
                "Currently only copland and condorcet winner is implemented"
            )
