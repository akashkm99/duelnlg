"""An implementation of the Copeland Confidence Bound algorithm."""
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import SingleCopelandProducer
from duelnlg.duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.util.utility_functions import argmax_set


class CopelandConfidenceBound(SingleCopelandProducer):
    r"""Implement the Copeland Confidence Bound(CCB) algorithm.

    The goal of the algorithm is to minimize the :term:`Copeland regret`.

    It is assumed that there are no ties between arms, i.e. the probability of any arm winning against another is never :math:`\frac{1}{2}`.

    The bound on the expected regret is given as :math:`\mathcal{O}\left(\frac{N^2+(C+L_C)N \ln(T)}{\Delta^2}\right)`. :math:`N` is the number of arms, :math:`T` is the time horizon. :math:`C` is the amount of :term:`Copeland winners<Copeland winner>` and :math:`L_C` the amount of arms a :term:`Copeland winner` will lose agains in expectation. For any :term:`Copeland winner` and any non-Copeland winner, :math:`\Delta` is the smallest absolute distance to :math:`\frac{1}{2}` in the probability of these arms dueling against each other. Note that the paper uses a different definition of :term:`Copeland regret`, in this library the value is half of that in the paper.

    The Copeland Confidence Bound algorithm is based on :cite:`zoghi2015copeland`.
    The performance of CCB degrades at about :math:`136` arms in experiments, for details
    refer to :cite:`zoghi2015copeland`.
    It proceeds by conducting duels which are most informative about the precedence of the
    participating arms in terms of their Copeland scores. The confirmed non-Copeland
    winners are eliminated based on the results of the prior duels. After conducting
    sufficient rounds, the set of possible :term:`Copeland winners<Copeland winner>` will converge which will
    result in minimal increment in :term:`Copeland regret` and thus the goal would be achieved.

    CCB runs continuously and in each time step it follows these steps:

    1. Optimistic and Pessimistic estimates (namely `U` and `L` respectively) of the
    Preference matrix are calculated.

    2. A :term:`Copeland winner` candidate :math:`a_c` is chosen using the optimistic estimate
    `U` such that it has a chance of being a true :term:`Copeland winner`. :math:`a_c` is chosen
    from a set of top scorers from `U`, especially those which are present in a
    list :math:`B_t`. :math:`B_t` contains the arms which have a higher chance of
    being a :term:`Copeland winner`.

    3. A suitable opponent :math:`a_d` is chosen using the pessimistic estimate `L` such
    that it can beat the notion of :math:`a_c` being the true :term:`Copeland winner`. By
    definition, `U` and `L` define a confidence interval around the original preference
    matrix. Using these confidence intervals, :math:`a_d` is chosen such that a duel
    between :math:`a_c` and :math:`a_d` provides maximum information about their
    precedence over each other in terms of their Copeland scores. The historically proven
    strong opponents to :math:`a_c` are maintained in a shortlist :math:`B_t^i`. The arms
    in this list are preferred while selecting an opponent. Such a list is maintained for
    every participating arm. The :math:`B_t^i` lists for non-\ :term:`Copeland winners<Copeland winner>` will contain
    a large number of opponents and thus help in their quick elimination from the list of
    possible winners.

    4. Finally, a duel is conducted between :math:`a_c` and :math:`a_d` and the result is
    recorded.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        Number of time steps to execute for.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    exploratory_constant
        A parameter which is used in calculating the upper confidence bounds.
        The confidence radius grows proportional to the square root of this value.
        A higher upper confidence bound results in more exploration.
        Corresponds to :math:`\alpha` in :cite:`zoghi2015copeland`. The value of
        ``exploratory_constant`` must be greater than :math:`0.5`.
        The default value is ``0.501`` which has been used in the experiments for calculating the
        confidence bounds in :cite:`zoghi2014ranker`.

    Attributes
    ----------
    copeland_winner_candidates
        The arms which have a higher possibilty of becoming a :term:`Copeland winner`.
        Corresponds to :math:`B_t` in :cite:`zoghi2015copeland`.
    max_allowed_losses
        Maximum allowed losses for a :term:`Copeland winner`.
        Corresponds to :math:`L_C` in :cite:`zoghi2015copeland`.
    respective_opponents
        A dictionary which has every arm as a key. Each value in this dictionary
        is a list. This list consists of the arms which are strong opponents to the
        arm present as the key.
        Corresponds to :math:`B_t^i` in :cite:`zoghi2015copeland`.
    time_step
        Number of rounds the algorithm has executed.
        Corresponds to :math:`t` in :cite:`zoghi2015copeland`.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> from duelnlg.duelpy.stats.metrics import AverageCopelandRegret
    >>> from duelnlg.duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> arms = list(range(len(preference_matrix)))
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, arms, random_state=random_state),
    ...     metrics={"copeland_regret": AverageCopelandRegret(preference_matrix)}
    ... )
    >>> ccb = CopelandConfidenceBound(feedback_mechanism=feedback_mechanism, exploratory_constant=0.6, time_horizon=100, random_state=random_state)
    >>> ccb.run()

    The best arm in this case is the last arm (index 2)

    >>> np.round(np.sum(feedback_mechanism.results["copeland_regret"]), 2)
    47.25
    >>> ccb.get_copeland_winner()
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
        self.time_step: int = 0
        if exploratory_constant <= 0.5:
            raise ValueError("Value of exploratory constant must be greater than 0.5")
        self.exploratory_constant = exploratory_constant
        self.copeland_winner_candidates: List[int] = list()
        self.respective_opponents: Dict[int, List[int]] = dict()
        # loss-threshold above which an arm can no longer be a Copeland winner
        self.max_allowed_losses: int = 0

        self.preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms()
        )
        # Initialize the Copeland winner candidates, their respective opponents and
        # the maximum number of losses allowed for a Copeland winner
        self._reset_copeland_attributes()

    def step(self) -> None:
        """Run one round of the algorithm."""
        self.time_step += 1

        # Update and set the new confidence radius in `preference_estimate`
        # as per the current `time_step`
        self._update_confidence_radius()

        # Update the Copeland winner candidates and opponents for each arm
        # using the following three methods.
        self._reset_disproven_hypotheses()
        self._remove_non_copeland_winners()
        self._add_copeland_winner_candidates()

        # Find best duel candidates
        (
            copeland_winner_candidate,
            suitable_opponent,
        ) = self._find_best_duel_candidates()
        copeland_winner_candidate_won = self.feedback_mechanism.duel(
            copeland_winner_candidate, suitable_opponent
        )
        self.preference_estimate.enter_sample(
            copeland_winner_candidate, suitable_opponent, copeland_winner_candidate_won
        )

    def get_copeland_winner(self) -> Optional[int]:
        """Determine a Copeland winner using CCB algorithm.

        Returns
        -------
        Optional[int]
            The first Copeland winner in order among all the estimated Copeland winners.
        """
        assert self.time_horizon is not None
        if self.time_step >= self.time_horizon:
            copeland_winners = list(
                self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
            )
            return copeland_winners[0]
        return None

    def _reset_copeland_attributes(self) -> None:
        """Reset the attributes for finding a Copeland winner."""
        self._reset_copeland_winner_candidates()
        self._reset_respective_opponents()
        self._reset_max_allowed_losses()

    def _update_confidence_radius(self) -> None:
        r"""Update the confidence radius using latest failure probability.

        Failure probability for the upper confidence bound is :math:`1/t^(2 * \alpha)`
        where :math:`t` is the current round of the algorithm and :math:`\alpha` is the
        exploratory constant.

        Refer Appendix D in https://arxiv.org/pdf/1506.00312.pdf for further
        details.
        """
        failure_probability = 1 / (self.time_step ** (2 * self.exploratory_constant))
        confidence_radius = HoeffdingConfidenceRadius(failure_probability)
        self.preference_estimate.set_confidence_radius(confidence_radius)

    def _reset_copeland_winner_candidates(self) -> None:
        """Reset the winner candidates list to include all the arms."""
        self.copeland_winner_candidates = list(
            range(self.feedback_mechanism.get_num_arms())
        )

    def _reset_respective_opponents(self) -> None:
        """Clear the opponents list for all the arms."""
        for arm in range(self.feedback_mechanism.get_num_arms()):
            self.respective_opponents[arm] = list()

    def _reset_max_allowed_losses(self) -> None:
        """Set maximum allowed losses for a Copeland winner."""
        self.max_allowed_losses = self.feedback_mechanism.get_num_arms()

    def _reset_disproven_hypotheses(self) -> None:
        """Check if the Copeland winner attributes need a reset.

        Reset the Copeland attributes if the lower estimate of an arm winning
        against any of its corresponding opponents is in favour of the arm.
        """
        reset = False
        pessimistic_matrix = self.preference_estimate.get_lower_estimate_matrix()
        for arm, opponents in self.respective_opponents.items():
            for opponent in opponents:
                if pessimistic_matrix[arm][opponent] > 0.5:
                    reset = True
                    break
            if reset:
                break

        if reset:
            self._reset_copeland_attributes()

    def _remove_non_copeland_winners(self) -> None:
        """Drop the candidate arms which cannot become Copeland winner."""
        non_copeland_winners = list()
        optimistic_matrix = self.preference_estimate.get_upper_estimate_matrix()
        pessimistic_copeland_scores = (
            self.preference_estimate.get_lower_estimate_matrix().get_copeland_scores()
        )
        optimistic_copeland_scores = (
            self.preference_estimate.get_upper_estimate_matrix().get_copeland_scores()
        )

        for candidate in self.copeland_winner_candidates:
            for score in pessimistic_copeland_scores:
                if optimistic_copeland_scores[candidate] < score:
                    non_copeland_winners.append(candidate)
                    break
            if len(self.respective_opponents[candidate]) != self.max_allowed_losses + 1:
                new_opponents = list()
                for arm in range(self.feedback_mechanism.get_num_arms()):
                    if optimistic_matrix[candidate][arm] < 0.5:
                        new_opponents.append(arm)
                self.respective_opponents[candidate] = new_opponents

        for candidate in non_copeland_winners:
            self.copeland_winner_candidates.remove(candidate)

        if not self.copeland_winner_candidates:
            self._reset_copeland_attributes()

    def _add_copeland_winner_candidates(self) -> None:
        """Add Copeland winner candidate arms."""
        optimistic_copeland_winners = list(
            self.preference_estimate.get_upper_estimate_matrix().get_copeland_winners()
        )
        pessimistic_copeland_scores = (
            self.preference_estimate.get_lower_estimate_matrix().get_copeland_scores()
        )
        optimistic_copeland_scores = (
            self.preference_estimate.get_upper_estimate_matrix().get_copeland_scores()
        )

        for arm in optimistic_copeland_winners:
            if optimistic_copeland_scores[arm] == pessimistic_copeland_scores[arm]:
                self.copeland_winner_candidates.append(arm)
                self.respective_opponents[arm].clear()
                self.max_allowed_losses = (
                    self.feedback_mechanism.get_num_arms()
                    - 1
                    - optimistic_copeland_scores[arm]
                )
                for other_arm in range(self.feedback_mechanism.get_num_arms()):
                    if other_arm != arm:
                        if (
                            len(self.respective_opponents[other_arm])
                            < self.max_allowed_losses + 1
                        ):
                            self.respective_opponents[other_arm].clear()
                        elif (
                            len(self.respective_opponents[other_arm])
                            > self.max_allowed_losses + 1
                        ):
                            self.respective_opponents[
                                other_arm
                            ] = self.random_state.choice(
                                self.respective_opponents[other_arm],
                                (self.max_allowed_losses + 1),
                                replace=False,
                            ).tolist()

    def _find_best_duel_candidates(self) -> Tuple[int, int]:
        """Find the arms whose duel is the most informative.

        Returns
        -------
        Tuple[int, int]
            A pair of arms where the first arm is a Copeland winner candidate
            and the second one is a suitable opponent.
        """
        if self.random_state.random() < 0.25:
            duel_pair_list = self._obtain_duel_pair_list()
            if duel_pair_list:
                random_index = self.random_state.choice(len(duel_pair_list))
                return duel_pair_list[random_index]

        copeland_winner_candidate = self._obtain_copeland_winner_candidate()
        opponent = self._obtain_suitable_opponent_for(copeland_winner_candidate)
        return copeland_winner_candidate, opponent

    def _obtain_duel_pair_list(self) -> List[Tuple[int, int]]:
        """Calculate and return the pairs of arms for a duel.

        Returns
        -------
        List[Tuple[int, int]]
            A list of pairs of arms which can be good dueling candidates.
        """
        pessimistic_matrix = self.preference_estimate.get_lower_estimate_matrix()
        optimistic_matrix = self.preference_estimate.get_upper_estimate_matrix()
        duel_pair_list: List[Tuple[int, int]] = list()

        for arm, opponents in self.respective_opponents.items():
            for opponent in opponents:
                if (
                    pessimistic_matrix[arm][opponent]
                    <= 0.5
                    <= optimistic_matrix[arm][opponent]
                ):
                    duel_pair_list.append((arm, opponent))
        return duel_pair_list

    def get_winner(self) -> Optional[int]:

        return list(
            self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )[0]

    def _obtain_copeland_winner_candidate(self) -> int:
        """Obtain a good Copeland winner candidate.

        Returns
        -------
        int
            A Copeland winner candidate
        """
        optimistic_copeland_winners = list(
            self.preference_estimate.get_upper_estimate_matrix().get_copeland_winners()
        )
        common_candidates = list(
            set(self.copeland_winner_candidates) & set(optimistic_copeland_winners)
        )
        # If the list of `common_candidates` is non-empty then with a probability of 2/3,
        # set `optimistic_copeland_winners` equal to `common_candidates`
        if len(common_candidates) > 0:
            if self.random_state.random() < 2 / 3:
                optimistic_copeland_winners = common_candidates

        return self.random_state.choice(optimistic_copeland_winners)

    def _obtain_suitable_opponent_for(self, copeland_winner_candidate: int) -> int:
        """Obtain a suitable opponent to the Copeland winner candidate.

        Parameters
        ----------
        copeland_winner_candidate
            The arm for which a suitable opponent arm for a duel is required.

        Returns
        -------
        int
            An appropriate opponent for the Copeland winner candidate.
        """
        pessimistic_matrix = self.preference_estimate.get_lower_estimate_matrix()
        optimistic_matrix = self.preference_estimate.get_upper_estimate_matrix()
        opponent_list = list()

        if self.random_state.random() < 0.5:
            opponent_list = self.respective_opponents[copeland_winner_candidate]
        else:
            opponent_list = list(range(self.feedback_mechanism.get_num_arms()))

        candidate_opponents = np.zeros(self.feedback_mechanism.get_num_arms())
        for j in opponent_list:
            if pessimistic_matrix[j][copeland_winner_candidate] <= 0.5:
                candidate_opponents[j] = optimistic_matrix[j][copeland_winner_candidate]

        suitable_opponents = argmax_set(
            candidate_opponents, exclude_indexes=[copeland_winner_candidate]
        )
        return suitable_opponents[0]
