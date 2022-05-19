"""Find the Condorcet winner in a PB-MAB problem with Interleaved Filtering."""

from typing import List
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import CondorcetProducer
from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.stats.preference_estimate import PreferenceEstimate


class InterleavedFiltering(CondorcetProducer, PacAlgorithm):
    r"""Implements the Interleaved Filtering algorithm.

    This algorithm finds the :term:`Condorcet winner`.

    A :term:`total order` over arms, :term:`strong stochastic transitivity` and the :term:`stochastic triangle inequality` are assumed.

    If the :term:`Condorcet winner` is not eliminated, which happens with low probability, the expected regret is bound by :math:`\mathcal{O}\left(\frac{N}{\epsilon_\ast \log(T)}\right)`. :math:`\epsilon_\ast` refers to the win probability of the best arm winning against the second best arm minus :math:`\frac{1}{2}`.

    The algorithm is explained in :cite:`yue2012bandits`.

    Exploration:

    Interleaved Filtering follows a sequential elimination approach in the exploration phase and thereby
    finds the best arm with a probability of at least :math:`1-\frac{1}{T}`, where :math:`T` is the time horizon. In each time step, the algorithm
    selects a candidate arm and compares it with all the other arms in a one-versus-all manner.
    If the algorithm selects an arm :math:`a` (candidate arm), then it compares all the other arms with :math:`a`. If there exists any arm, :math:`b`
    such that the upper confidence bound of :math:`a` beating :math:`b` is less than :math:`\frac{1}{2}`, then arm :math:`a` is eliminated and arm :math:`b` becomes
    the candidate arm and is compared to all other active arms. It applies a pruning technique to
    eliminate arm :math:`b` if the lower confidence bound of :math:`a` beating :math:`b` is greater than :math:`\frac{1}{2}`,
    as it cannot be considered as the best arm with high probability. After the exploration, the candidate arm and the total number of comparisons are given as output.

    Exploitation:

    If the total number of comparisons is less than the given time horizon then the algorithm enters into the exploitation phase.
    In the exploitation phase, only the best arm from the exploration phase is pulled and compared to itself, assuming that the exploration found the best arm.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        For how many time steps the algorithm should run, must be greater or equal to the number
        of arms.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.

    Attributes
    ----------
    failure_probability
        Allowed failure-probability (:math:`\delta`), i.e. probability that the actual value lies outside of the computed confidence interval.
        Derived from the Hoeffding bound.
    candidate_arm
        Randomly selected arm (corresponds to :math:`\hat{b}` in :cite:`yue2012bandits`) from the list of arms.
    arms_without_candidate
        The remaining set of arms (corresponds to :math:`W` in :cite:`yue2012bandits`) after removing the candidate arm.
    preference_estimate
        Estimation of a preference matrix based on samples.
    feedback_mechanism
    time_horizon

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
    >>> random_state=np.random.RandomState(3)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)
    >>> time_horizon = 1500
    >>> interleaved_filtering = InterleavedFiltering(feedback_mechanism, time_horizon, random_state=random_state)
    >>> interleaved_filtering.run()
    >>> condorcet_winner = interleaved_filtering.get_condorcet_winner()
    >>> condorcet_winner
    2
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.RandomState = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        assert self.time_horizon is not None  # for mypy
        self.failure_probability = 1 / (
            self.time_horizon * (self.feedback_mechanism.get_num_arms() ** 2)
        )
        random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.candidate_arm = random_state.choice(self.feedback_mechanism.get_arms())
        self.arms_without_candidate = self.feedback_mechanism.get_arms().copy()
        self.arms_without_candidate.remove(self.candidate_arm)
        # See the proof of Lemma 4 in the paper. The factor `8` corresponds to `m` in the proof.
        self.preference_estimate = PreferenceEstimate(
            feedback_mechanism.get_num_arms(),
            confidence_radius=HoeffdingConfidenceRadius(
                self.failure_probability, factor=8,
            ),
        )

    def get_condorcet_winner(self) -> Optional[int]:
        """Return the estimated Condorcet winner, assuming the algorithm has already run.

        Returns
        -------
        candidate_arm
           The condorcet winner in the set of arms given to the algorithm.
        """
        return self.candidate_arm  # if self.exploration_finished() else None

    def get_winner(self):
        return self.get_condorcet_winner()

    def explore(self) -> None:
        r"""Execute one round of exploration."""
        for arm in self.arms_without_candidate:
            self.preference_estimate.enter_sample(
                self.candidate_arm,
                arm,
                self.feedback_mechanism.duel(self.candidate_arm, arm),
            )
            # Terminate explore
            if self.feedback_mechanism.get_num_duels() == self.time_horizon:
                break
        updated_arms_without_candidate = self._prune_arms()
        (self.arms_without_candidate) = self._find_candidate_arm(
            updated_arms_without_candidate
        )

    def _prune_arms(self) -> list:
        """Eliminate arms that cannot be expected to win against the candidate within the confidence interval.

        Returns
        -------
        updated_arms_without_candidate
           The remaining set of arms after eliminating all the arms which, do not satisfy the condition.
        """
        # A duplicate list of arms without candidate arm, in order to avoid the index out of bounds error while
        # removing an arm from the arms_without_candidate.
        duplicate_arms_without_candidate = np.copy(self.arms_without_candidate)
        for arm in duplicate_arms_without_candidate:
            # check whether probability_estimate is greater than 1/2 AND 1/2 is not in the confidence_bounds.
            if (
                self.preference_estimate.get_lower_estimate(self.candidate_arm, arm)
                > 1 / 2
            ):
                self.arms_without_candidate.remove(arm)
        updated_arms_without_candidate = self.arms_without_candidate
        return updated_arms_without_candidate

    def _find_candidate_arm(
        self, updated_arms_without_candidate: List[int]
    ) -> List[int]:
        """Find the candidate arm and remove the new candidate arm from the list of updated arms without candidate arm.

        Parameters
        ----------
        updated_arms_without_candidate
            The remaining set of arms after eliminating all the arms whose, lower confidence bound of candidate arm and
            each arm in the set of all arms except candidate arm is > 1/2.

        Returns
        -------
        updated_arms_without_candidate
            The updated list of arms after removing the new candidate arm.
        """
        candidate_found = False
        for arm in updated_arms_without_candidate:
            # check whether, if there is any arm whose probability_estimate is less than 1/2 AND 1/2 is not in
            # the confidence_bounds.
            if (
                self.preference_estimate.get_upper_estimate(self.candidate_arm, arm)
                < 1 / 2
            ):
                self.candidate_arm = arm
                candidate_found = True
                break
        if candidate_found:
            self.arms_without_candidate = updated_arms_without_candidate
            self.arms_without_candidate.remove(self.candidate_arm)
        return updated_arms_without_candidate

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        If no time horizon is provided, this coincides with is_finished. Once
        this function returns ``True``, the algorithm will have finished
        computing a PAC Copeland winner.
        """
        return len(self.arms_without_candidate) == 0
