"""Algorithms based on the Plackett-Luce class of permutation distributions."""

from typing import List
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import AllApproximateCondorcetProducer
from duelnlg.duelpy.algorithms.interfaces import CopelandRankingProducer
from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException
from duelnlg.duelpy.util.sorting import Quicksort


def determine_better_arm(
    feedback_mechanism: FeedbackMechanism,
    time_horizon: Optional[int],
    arm_1: int,
    arm_2: int,
) -> int:
    """Duel the given arms once and determine the winner, but avoid violating the time horizon.

    Parameters
    ----------
    feedback_mechanism
        The ``FeedbackMechanism`` object used for dueling
    time_horizon
        The time horizon bound
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
        1 if the first arm won, -1 if the second arm won.
    """
    if time_horizon is not None and time_horizon <= feedback_mechanism.get_num_duels():
        raise AlgorithmFinishedException()
    if feedback_mechanism.duel(arm_1, arm_2):
        return 1
    else:
        return -1


class PlackettLucePACItem(AllApproximateCondorcetProducer, PacAlgorithm):
    r"""Implementation of the Plackett-Luce PAC-Item algorithm.

    This algorithm finds for a given confidence all arms that are :math:`\epsilon`-close to the :term:`Condorcet winner`.

    It assumes arms are distributed according to the :term:`Plackett-Luce distribution`, which assigns a utility to each arm.
    The utilities of two arms determine the probability of either arm winning against the other. For details on this distribution see :cite:`szorenyi2015online`.

    The sample complexity of the algorithm is bound by :math:`\mathcal{O}\left(\frac{\max_{i\neq i^\ast}}{\Delta_i^2} \log\left(\frac{N}{\Delta_i \delta}\right)\right)`. Here, :math:`N` is the number of arms and :math:`\Delta_i=\frac{\max\{\epsilon,p_{i^\ast,i}-\frac{1}{2}\}}{2}`, where :math:`p_{i^\ast,i}` is the probability of the best arm winning against arm i.

    The algorithm repeatedly sorts arms with a comparison budget-constrained :class:`QuickSort<duelpy.util.sorting.QuickSort>` algorithm and eliminates inferior arms.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Used for the random pivot selection in the Quicksort mode.
    failure_probability
        An upper bound on the acceptable probability to fail, called :math:`\delta` in :cite:`szorenyi2015online`.
    epsilon
        Acceptable difference to optimum, also called :math:`\epsilon` in :cite:`szorenyi2015online`.

    Attributes
    ----------
    random_state
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
    >>> plackett_luce = PlackettLucePACItem(feedback_mechanism, random_state=random_state, failure_probability=0.1, epsilon=0.01)
    >>> plackett_luce.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> arm = plackett_luce.get_approximate_condorcet_winners()
    >>> arm, comparisons
    ([2], 476)
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        failure_probability: float = 0.05,
        epsilon: float = 0.1,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        num_arms = self.feedback_mechanism.get_num_arms()

        def probability_scaling(num_samples: int) -> float:
            return 4 * (num_arms * num_samples) ** 2

        self.preference_estimate = PreferenceEstimate(
            num_arms,
            HoeffdingConfidenceRadius(failure_probability, probability_scaling),
        )

        self._epsilon = epsilon

        self._candidates = self.feedback_mechanism.get_arms()
        self._condorcet_winners: Optional[List[int]] = None

    def _sort_step(self) -> List[List[int]]:
        """Execute budgeted Quicksort.

        Returns
        -------
        List[List[int]]
            Ordered buckets of arms.
        """
        quicksort = Quicksort(
            self._candidates,
            lambda a1, a2: determine_better_arm(
                self.feedback_mechanism, self.time_horizon, a1, a2
            ),
            self.random_state,
        )
        # execute |arms|-1 sorting steps
        steps = len(self._candidates) - 1
        try:
            for _ in range(steps):
                quicksort.step()
        except AlgorithmFinishedException:
            pass

        return quicksort.get_intermediate_result()

    def _rank_breaking(self, ranking: List[List[int]]) -> None:
        """Transform a partial ranking into pairwise duels.

        This is an implementation of the updating described in line 7 of the PLPAC pseudo code in :cite:`szorenyi2015online`.
        See :cite:`azari2013generalized` for details on rank breaking.

        Parameters
        ----------
        ranking
            The arms in ordered buckets, the order inside the buckets is ignored.
        """
        for r_index, rank in enumerate(ranking):
            for arm in rank:
                for higher_rank in ranking[r_index + 1 :]:
                    for worse_arm in higher_rank:
                        self.preference_estimate.enter_sample(arm, worse_arm, True)

    def _prune_arms(self) -> None:
        """Remove arms which are inferior with high probability."""
        new_arms = self._candidates.copy()
        for index, arm_1 in enumerate(self._candidates):
            for arm_2 in self._candidates[index + 1 :]:
                if arm_1 not in new_arms or arm_2 not in new_arms:
                    continue
                if self.preference_estimate.get_lower_estimate(arm_1, arm_2) > 1 / 2:
                    new_arms.remove(arm_2)
                if self.preference_estimate.get_upper_estimate(arm_1, arm_2) < 1 / 2:
                    new_arms.remove(arm_1)
        self._candidates = new_arms

    def explore(self) -> None:
        """Execute one exploration step."""
        ranking = self._sort_step()
        # use intermediate result
        self._rank_breaking(ranking)

        # discard inferior arms
        self._prune_arms()

        # determine possible condorcet winners
        condorcet_winners = self.preference_estimate.get_lower_estimate_matrix().get_epsilon_condorcet_winners(
            self._epsilon
        )

        if len(condorcet_winners) > 0:
            self._condorcet_winners = list(condorcet_winners)

    def exploit(self) -> None:
        """Exploit knowledge by uniformly random selection two of the epsilon-delta Condorcet winners."""
        self.feedback_mechanism.duel(
            self.random_state.choice(self._condorcet_winners),
            self.random_state.choice(self._condorcet_winners),
        )

    def exploration_finished(self) -> bool:
        """Determine whether the best arm has been found."""
        return self._condorcet_winners is not None

    def get_winner(self) -> Optional[int]:

        return list(
            self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )[0]

    def step(self) -> None:
        """Run one step of the algorithm."""
        if not self.exploration_finished():
            self.explore()
        else:
            self.exploit()

    def get_approximate_condorcet_winners(self) -> Optional[List[int]]:
        """Get the arm with the highest probability of being the first in a ranking of the arms.

        Returns
        -------
        Optional[int]
            The best arm, None if has not been calculated yet.
        """
        return self._condorcet_winners


class PlackettLuceAMPR(CopelandRankingProducer, PacAlgorithm):
    r"""Implementation of the Plackett-Luce Approximate Most Probable Ranking algorithm, which computes a ranking over the arms.

    This algorithm assumes the arms are sampled from the :term:`Plackett-Luce distribution`. This distribution assigns utilities to arms, from which win probabilities can be inferred. For more information, see :cite:`szorenyi2015online`.

    To compute a ranking, the algorithm proceeds by repeating the following steps. First, the arms are divided into connected components, that is groups, for which the confidence intervals around the estimated ranking positions overlap. Arms in the same group can not be ordered confidently. That is, the best guess is that they are of equal rank.
    The order of arms in different groups is known, so each group can be analyzed in isolation. The arms in each group are compared by executing the Budgeted :class:`QuickSort<duelpy.util.sorting.QuickSort>` algorithm.
    This leads to shrinking confidence intervals. Two conditions allow the algorithm to terminate. Either all groups only contain one arm, at which point the ranking is known, or, some groups exist, whose arms are :math:`\epsilon`-close to each other. In the second case, the arms are assumed to be equal, the ties are broken an arbitarily.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Used for the random pivot selection in the Quicksort mode.
    failure_probability
        An upper bound on the acceptable probability to fail, called :math:`\delta` in :cite:`szorenyi2015online`.
    epsilon
        Acceptable difference to optimum, also called :math:`\epsilon` in :cite:`szorenyi2015online`.

    Attributes
    ----------
    random_state
    preference_estimate
        Estimates for arm preferences.

    Examples
    --------
    Find the Ranking in this example:

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
    >>> pl = PlackettLuceAMPR(feedback_mechanism, random_state=random_state, failure_probability=0.9, epsilon=0.1)
    >>> pl.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> ranking = pl.get_ranking()
    >>> ranking, comparisons
    ([2, 4, 3, 1, 0], 1097)
    """

    class Bounds:
        """Helper class to keep track of lower and upper rank bounds."""

        def __init__(self, lower_bound: int, upper_bound: int):
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        def union(self, other: "PlackettLuceAMPR.Bounds") -> None:
            """Combine two bounds objects by union."""
            self.lower_bound = min(self.lower_bound, other.lower_bound)
            self.upper_bound = max(self.upper_bound, other.upper_bound)

        def intersection_test(self, other: "PlackettLuceAMPR.Bounds") -> bool:
            """Check whether two bounds objects intersect."""
            return (
                self.lower_bound <= other.lower_bound <= self.upper_bound
                or self.lower_bound <= other.upper_bound <= self.upper_bound
            )

    class Component:
        """Helper class to keep track of arm components, i.e. arm subgroups."""

        def __init__(self, arms: List[int], bounds: "PlackettLuceAMPR.Bounds"):
            self.bounds = bounds
            self.arms = arms

        def merge(self, other: "PlackettLuceAMPR.Component") -> None:
            """Merge two Component objects and update the resulting bounds."""
            self.bounds.union(other.bounds)
            self.arms += other.arms

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        failure_probability: float = 0.05,
        epsilon: float = 0.1,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        num_arms = self.feedback_mechanism.get_num_arms()

        def probability_scaling(num_samples: int) -> float:
            return 4 * (num_arms * num_samples) ** 2

        self.preference_estimate = PreferenceEstimate(
            num_arms,
            HoeffdingConfidenceRadius(failure_probability, probability_scaling),
        )

        self._epsilon = epsilon

        self._arm_bounds = [
            PlackettLuceAMPR.Bounds(0, num_arms - 1) for _ in range(num_arms)
        ]

        self._ranking: Optional[List[int]] = None

    def _find_components(self) -> List["PlackettLuceAMPR.Component"]:
        """Divide arms into connected components.

        These components are groups of arms, which can currently not be ordered.
        """
        arms = self.feedback_mechanism.get_arms()

        # find connected components, arms with intersecting bounds
        components = [
            PlackettLuceAMPR.Component([i], self._arm_bounds[i])
            for i in range(len(arms))
        ]

        merged_components = []
        while len(components) > 0:
            component_1 = components[0]
            for i, component_2 in enumerate(components[1:]):
                i += 1  # we start with i=0 at index 1, so it needs to be adapted
                if component_1.bounds.intersection_test(component_2.bounds):
                    component_1.merge(component_2)
                    components.pop(i)
                    break
            else:
                merged_components.append(components.pop(0))
        return merged_components

    def _is_order_known(self, components: List["PlackettLuceAMPR.Component"]) -> bool:
        """Check the components for the termination condition."""
        for component in components:
            for i, arm_1 in enumerate(component.arms):
                for arm_2 in component.arms[i + 1 :]:
                    if 1 / 2 - self._epsilon >= self.preference_estimate.get_lower_estimate(
                        arm_1, arm_2
                    ) or 1 / 2 + self._epsilon <= self.preference_estimate.get_upper_estimate(
                        arm_1, arm_2
                    ):
                        return False
        return True

    def _compare_component(self, component: "PlackettLuceAMPR.Component") -> None:
        """Compare arms inside one component using Budgeted Quicksort."""
        component_size = len(component.arms)
        if component_size > 1:
            comparison_bound = int(3 * (component_size + 1) * np.log(component_size))
            quicksort = Quicksort(
                self.feedback_mechanism.get_arms().copy(),
                lambda a1, a2: determine_better_arm(
                    self.feedback_mechanism, self.time_horizon, a1, a2
                ),
                self.random_state,
            )
            try:
                for _ in range(comparison_bound):
                    quicksort.step()
            except AlgorithmFinishedException:
                pass
            ranking = quicksort.get_intermediate_result()
            for rank_index, rank in enumerate(ranking):
                for arm in rank:
                    for higher_rank in ranking[rank_index + 1 :]:
                        for worse_arm in higher_rank:
                            self.preference_estimate.enter_sample(arm, worse_arm, True)

    def _update_arm_bounds(self) -> None:
        """Recalculate arm bounds based on preference estimates."""
        arms = self.feedback_mechanism.get_arms()
        for i in arms:
            # update bounds
            self._arm_bounds[i].lower_bound = len(
                [
                    arm
                    for arm in arms
                    if arm != i
                    and self.preference_estimate.get_lower_estimate(i, arm) > 1 / 2
                ]
            )
            self._arm_bounds[i].upper_bound = self._arm_bounds[i].lower_bound + len(
                [
                    arm
                    for arm in arms
                    if arm != i
                    and self.preference_estimate.get_lower_estimate(i, arm) <= 1 / 2
                    and self.preference_estimate.get_upper_estimate(i, arm) > 1 / 2
                ]
            )

    def explore(self) -> None:
        """Explore arms by advancing the sorting algorithm."""
        components = self._find_components()

        terminated = self._is_order_known(components)

        if terminated:
            arms = self.feedback_mechanism.get_arms()
            # ties are broken randomly
            tie_breaker = np.random.permutation(len(arms))
            # the second element of a tuple is used as a tie breaker by sorted
            self._ranking = sorted(
                arms, key=lambda x: (-self._arm_bounds[x].lower_bound, tie_breaker[x])
            )

        for component in components:
            self._compare_component(component)

        self._update_arm_bounds()

    def step(self) -> None:
        """Execute one step of the algorithm."""
        if self._ranking is None:
            self.explore()
        else:
            self.exploit()

    def get_ranking(self) -> Optional[List[int]]:
        """Get the computed ranking.

        Returns
        -------
        Optional[List[int]]
            The ranking, None if it has not been calculated yet.
        """
        return self._ranking

    def exploration_finished(self) -> bool:
        """Determine whether the best arm has been found."""
        return self._ranking is not None
