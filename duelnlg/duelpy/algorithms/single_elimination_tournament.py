"""An implementation of the Single-Elimination Tournament algorithms."""
from typing import List
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import CondorcetProducer
from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.algorithms.interfaces import PartialRankingProducer
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats.preference_estimate import PreferenceEstimate
from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException
from duelnlg.duelpy.util.feedback_decorators import BudgetedFeedbackMechanism
from duelnlg.duelpy.util.heap import Heap


def _compute_binary_comparisons(
    arms: List[int], preference_separation: float, epsilon: float
) -> int:
    r"""Calculate the number of binary comparisons.

    Parameters
    ----------
    arms
        Arms given by the algorithm.
    preference_separation
        The assumed preference separation between the best arm and all other arms.
    epsilon
        :math:`\epsilon` in :math:`(\epsilon, \delta)`-PAC algorithms, given by the user.

    Returns
    -------
    int
        Number of times each comparison is repeated before coming to a conclusion. Corresponds to :math:`m` in :cite:`mohajer2017active`.
    """
    # From the formula 16 in :cite:`mohajer2017active`
    probability_scaling_factor = np.log(1 / epsilon) / np.log(np.log(len(arms)))
    # Apply ceiling function, to the formula 16 in :cite:`mohajer2017active` in order to avoid the decimals as binary comparisons.
    return int(
        np.ceil(
            (1 + probability_scaling_factor)
            * np.log2(2)
            / 2
            * np.log(np.log(len(arms)))
            / preference_separation
        )
    )


class SingleEliminationTop1Select(CondorcetProducer, PacAlgorithm):
    r"""The Top-1 Selection part of Single-Elimination Tournament.

    The goal of this algorithm is to find the top (Rank = 1) arm while minimizing the sample complexity.

    A :term:`total order` over arms, :term:`strong stochastic transitivity` and the :term:`stochastic triangle inequality` are assumed.

    The amount of pairwise comparisons made by the algorithm is bound by :math:`O\left(\frac{ N \log\log N}{\Delta_{1}}\right)`, where :math:`N` is the number of arms, and :math:`\Delta` is the preference separation.

    The algorithm was originally introduced in :cite:`mohajer2017active`. It contains many layers. In every layer the arms are paired in a random manner.
    One arm from each pair is selected with the help of pairwise comparisons between the  two arms, while the other arm is eliminated.
    As the duel between the arms is from a random observation, the duel is repeated :math:`m` (in :cite:`mohajer2017active`) number of times,
    thus establishing a probability distribution to the duel. The algorithm gives the top-1 arm with adequately large
    number of binary comparisons (larger :math:`m` in :cite:`mohajer2017active`).

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    preference_separation
        The assumed preference separation between the best arm and all other arms. Assumed to be ``0.01`` if neither ``preference_separation`` nor ``duels_per_comparison`` is specified. Corresponds to :math:`\Delta_{1,S}` in :cite:`mohajer2017active`.
    duels_per_comparison
        Number of times each comparison is repeated before coming to a conclusion. If this is not specified, an optimal value that guarantees the assumptions in the paper is computed from ``preference_separation``. See :cite:`mohajer2017active` for more details. Corresponds to :math:`m` in :cite:`mohajer2017active`.
    epsilon
        :math:`\epsilon` in :math:`(\epsilon, \delta)`-PAC algorithms, given by the user.
    arms_subset
        The set of arms given to the algorithm by other algorithms otherwise the amrs from ``feedback_mechanism`` will be taken.
    preference_estimate
        A ``PreferenceEstimate`` object is needed if this algorithm is used as a subroutine and the result is required to be stored in further rounds. The default value is ``None``.

    Attributes
    ----------
    preference_estimate
        Estimation of a preference matrix based on samples.
    condorcet_winner
        Top-1 arm in the given set of arms.
    duels_per_comparison
    feedback_mechanism

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
    >>> single_elimination_tournament = SingleEliminationTop1Select(feedback_mechanism)
    >>> single_elimination_tournament.run()
    >>> condorcet_winner = single_elimination_tournament.get_condorcet_winner()
    >>> condorcet_winner
    2
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        preference_separation: Optional[float] = None,
        duels_per_comparison: Optional[int] = None,
        arms_subset: Optional[List[int]] = None,
        preference_estimate: Optional[PreferenceEstimate] = None,
        random_state: np.random.RandomState = None,
        epsilon: float = 0.01,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.feedback_mechanism = feedback_mechanism
        self.arms = (
            self.feedback_mechanism.get_arms() if arms_subset is None else arms_subset
        )

        self.arms_duel_limit = duels_per_comparison
        if self.arms_duel_limit is not None and preference_separation is not None:
            raise Exception(
                "The `duels_per_comparison` and `preference_separation` parameters are mutually exclusive. Please refer "
                "to the documentation. "
            )
        if preference_separation is None:
            preference_separation = (
                0.01  # default value, mentioned in the documentation
            )
        if self.arms_duel_limit is None and len(self.arms) > 1:
            self.arms_duel_limit = _compute_binary_comparisons(
                arms=self.arms,
                preference_separation=preference_separation,
                epsilon=epsilon,
            )
            self.condorcet_winner = None

        self.exploration_steps = 0

        self.preference_estimate = (
            preference_estimate
            if preference_estimate is not None
            else PreferenceEstimate(self.feedback_mechanism.get_num_arms())
        )
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.random_state.shuffle(self.arms)

    def explore(self) -> None:
        """Run one step of exploration."""
        for arm_index in range(
            int(np.ceil(len(self.arms) / np.power(2, self.exploration_steps + 1)))
        ):
            # check whether the index refers to a valid arm
            if len(self.arms) == 2 * arm_index + 1:
                self.arms[arm_index] = self.arms[2 * arm_index]
                break
            # select two arms for comparison
            arm_1 = self.arms[2 * arm_index]
            arm_2 = self.arms[2 * arm_index + 1]
            # this is necessary for mypy
            # this must always be true since duels_per_comparison is initialized with a not-None value in `__init__´.
            assert self.arms_duel_limit is not None
            for _ in range(self.arms_duel_limit):
                self.preference_estimate.enter_sample(
                    arm_1, arm_2, self.feedback_mechanism.duel(arm_1, arm_2)
                )

                if self.is_finished():
                    # time horizon reached before exploration was finished

                    if self.preference_estimate.get_mean_estimate(arm_1, arm_2) > 1 / 2:
                        self.arms[arm_index] = self.arms[2 * arm_index]
                    else:
                        self.arms[arm_index] = self.arms[2 * arm_index + 1]
                    return

            # The winner is moved to the next step, that is to the first half of the currently investigated part of the arm list
            if self.preference_estimate.get_mean_estimate(arm_1, arm_2) > 1 / 2:
                self.arms[arm_index] = self.arms[2 * arm_index]
                # print (arm_1, "beat", arm_2)
            else:
                self.arms[arm_index] = self.arms[2 * arm_index + 1]
                # print (arm_2, "beat", arm_1)

        self.condorcet_winner = self.arms[0]
        self.exploration_steps = self.exploration_steps + 1

    def get_winner(self):
        return self.condorcet_winner

    def exploit(self) -> None:
        """Run one step of exploitation."""
        winner = self.get_condorcet_winner()
        assert winner is not None
        self.feedback_mechanism.duel(winner, winner)

    def step(self) -> None:
        """Run one step of the algorithm."""
        if not self.exploration_finished():
            self.explore()
        else:
            self.exploit()

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        Once this function returns ``True``, the algorithm will have finished
        computing a :term:`PAC` :term:`Condorcet winner`.
        """
        return self.exploration_steps > int(np.ceil(np.log(len(self.arms))))

    def get_condorcet_winner(self) -> Optional[int]:
        """Return the estimated PAC-Condorcet winner.

        Returns
        -------
        int
           The Condorcet winner in the set of arms given to the algorithm.
        """
        if not self.exploration_finished():
            return None
        return self.condorcet_winner


class SingleEliminationTopKSorting(PartialRankingProducer, PacAlgorithm):
    r"""Implements the top-k sorting algorithm in the Single-Elimination Tournament.

    The goal of this algorithm is to find the top-k arms while minimizing the sample complexity.

    The algorithm assumes a :term:`total order` over the arms.

    The algorithm has sample complexity of :math:`\mathcal{O}\left(\frac{(N+k \log k) \max \{\log k, \log \log N\}}{\Delta_{k}}\right)` where :math:`\Delta_{k}=\min _{i \in[k]} \min _{j: j \geq i} \Delta_{i, j}^{2}` in the case of top-k ranking
    and :math:`\Delta_{k}=\Delta_{k, k+1}^{2}` in the case of top-k identification. :math:`N` is the number of arms.

    The algorithm divides the dataset or the set of arms into :math:`k` sub-groups each of size :math:`\frac{N}{k}`. From every sub-group a top arm is selected by using the :class:`TopOneSelection<duelpy.algorithms.single_elimination_tournament.SingleEliminationTop1Select>` algorithm and short list all the winners.
    A (max-) heap data structure is built from the short list,there by getting the top arm from the obtained heap, which will be the root element of the heap.
    Then the top arm is removed from the short list. In order to find the second best arm, again the home sub-group from which the previous top arm is taken, is accessed
    and the second best arm is identified and added to the short list. This process of identifying and removing is repeated for :math:`k - 1` times, untill all the top-k arms are identified.
    See Algorithm 2 in :cite:`mohajer2017active` for more details.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    preference_separation
        The assumed preference separation between the best arm and all other arms. Assumed to be ``0.01`` if neither ``preference_separation`` nor ``duels_per_comparison`` is specified. Corresponds to :math:`\Delta_{1,S}` in :cite:`mohajer2017active`.
    duels_per_comparison
        Number of times each comparison is repeated before coming to a conclusion. If this is not specified, an optimal value that guarantees the assumptions in the paper is computed from ``preference_separation``. See :cite:`mohajer2017active` for more details. Corresponds to :math:`m` in :cite:`mohajer2017active`.
    k_top_ranked
        The desired number of top arms in the given set of arms. If this is not specified it is taken as ``2``.
    epsilon
        :math:`\epsilon` in :math:`(\epsilon, \delta)`-:term:`PAC` algorithms, given by the user.

    Attributes
    ----------
    budgeted_feedback_mechanism
        A ``BudgetedFeedbackMechanism`` object describing the environment.
    preference_estimate
        Estimation of a preference matrix based on samples.
    top_k_arms
        List of top k arms given by the algorithm.
    sub_groups
        Set of arms divided into :math:`k` sub-groups each of size :math:`\frac{N}{k}`.
    sub_group_index
        Index of the sub group.
    top_1_selection_class
        Storing an instance of SingleEliminationTop1Select class.
    short_list
        From every sub-group a top arm is selected and a short list of all the winners is created.
    heap
        Storing an instance of ``Heap`` class.
    algorithm_stage
        The stage of the algorithm.
    rank_index
        Present rank index.
    heap_updated
        Check whether the heap is updated or not.
    feedback_mechanism
    duels_per_comparison

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
    >>> random_state=np.random.RandomState(100)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)
    >>> single_elimination_tournament = SingleEliminationTopKSorting(feedback_mechanism)
    >>> single_elimination_tournament.run()
    >>> top_k_arms = single_elimination_tournament.get_partial_ranking()
    >>> top_k_arms
    [2, 1]
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        k_top_ranked: Optional[int] = None,
        time_horizon: Optional[int] = None,
        preference_separation: Optional[float] = None,
        duels_per_comparison: Optional[int] = None,
        epsilon: float = 0.01,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.feedback_mechanism = feedback_mechanism
        if k_top_ranked is None:
            self.k_top_ranked = 2
        else:
            self.k_top_ranked = k_top_ranked
        self.time_horizon = time_horizon
        self.budgeted_feedback_mechanism = feedback_mechanism
        if self.time_horizon is not None:
            self.budgeted_feedback_mechanism = BudgetedFeedbackMechanism(
                self.feedback_mechanism,
                max_duels=self.time_horizon - self.feedback_mechanism.get_num_duels(),
            )
        self.preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms(),
        )
        if duels_per_comparison is not None and preference_separation is not None:
            raise Exception(
                "The `duels_per_comparison` and `preference_separation` parameters are mutually exclusive. Please refer "
                "to the documentation. "
            )

        if preference_separation is None:
            preference_separation = (
                0.01  # default value, mentioned in the documentation
            )
        if duels_per_comparison is None:
            duels_per_comparison = _compute_binary_comparisons(
                arms=feedback_mechanism.get_arms(),
                preference_separation=preference_separation,
                epsilon=epsilon,
            )

        self.preference_estimate = PreferenceEstimate(
            feedback_mechanism.get_num_arms(),
        )

        def compare_repeatedly(arm_1: int, arm_2: int) -> int:
            if self.is_finished():
                raise AlgorithmFinishedException
            self.preference_estimate.enter_sample(
                arm_1, arm_2, self.feedback_mechanism.duel(arm_1, arm_2)
            )
            # In order to avoid incompatible type error, before the algorithm gives the winner.This
            # assertion is necessary for mypy since copeland_winner is defined as
            # Optional[int] in the TopOneSelection.
            assert duels_per_comparison is not None
            if (
                self.preference_estimate.get_num_samples(arm_1, arm_2)
                >= duels_per_comparison
            ):
                return (
                    1
                    if self.preference_estimate.get_mean_estimate(arm_1, arm_2) > 1 / 2
                    else -1
                )
            else:
                return 0

        self.sub_groups = [
            self.feedback_mechanism.get_arms()[i :: self.k_top_ranked]
            for i in range(self.k_top_ranked)
        ]
        self.sub_group_index = 0
        self.top_1_selection_class = SingleEliminationTop1Select

        self.top_1_selection_instance: Optional[
            SingleEliminationTop1Select
        ] = SingleEliminationTop1Select(
            feedback_mechanism=self.budgeted_feedback_mechanism,
            arms_subset=self.sub_groups[self.sub_group_index].copy(),
            preference_estimate=self.preference_estimate,
            time_horizon=self.time_horizon,
        )
        self.short_list: List[int] = list()
        self.heap = Heap(compare_fn=compare_repeatedly)
        self.algorithm_stage = 0
        self.rank_index = 0
        self.top_k_arms: List[int] = list()
        self.heap_updated = False

    def _create_short_list(self) -> None:
        """Short list is created with the winners from given set of arms."""
        assert self.top_1_selection_instance is not None
        if self.top_1_selection_instance.is_finished():
            # In order to avoid incompatible type error, before the algorithm gives the winner.
            # This assertion is necessary for mypy since copeland_winner is defined as
            # Optional[int] in the TopOneSelection.

            winner = self.top_1_selection_instance.get_condorcet_winner()
            assert winner is not None

            self.short_list.append(winner)
            self.sub_group_index += 1

            if self.sub_group_index >= len(self.sub_groups):
                self.algorithm_stage = 1
                return

            self.top_1_selection_instance = self.top_1_selection_class(
                feedback_mechanism=self.budgeted_feedback_mechanism,
                arms_subset=self.sub_groups[self.sub_group_index].copy(),
                preference_estimate=self.preference_estimate,
                time_horizon=self.time_horizon,
            )
        self.top_1_selection_instance.step()

    def _create_heap(self) -> None:
        """Create a heap data structure from the given short list of arms."""
        for i in range(self.k_top_ranked):
            self.heap.insert(self.short_list[i], self.sub_groups[i])
        self.algorithm_stage = 2
        self.top_1_selection_instance = None  # type: ignore

    def _top_k_sorting(self) -> None:
        """Top-k arms is identified in the given set of arms. For instance top-2 arms in the given example."""
        if not self.heap.is_finished():
            self.heap.step()
            return

        if self.top_1_selection_instance is None:
            min_node = self.heap.get_min()
            assert min_node is not None
            self.top_1_selection_instance = SingleEliminationTop1Select(
                feedback_mechanism=self.budgeted_feedback_mechanism,
                arms_subset=min_node[1].copy(),
                preference_estimate=self.preference_estimate,
                time_horizon=self.time_horizon,
            )
        else:
            if (
                self.top_1_selection_instance.is_finished()
                and self.rank_index < self.k_top_ranked
            ):
                if not self.heap_updated:
                    top_1_arm = self.top_1_selection_instance.get_condorcet_winner()
                    assert top_1_arm is not None
                    self.heap.update_min_key(top_1_arm)
                    self.heap_updated = True
                    return
                self.heap_updated = False
                min_node = self.heap.get_min()
                assert min_node is not None

                self.top_k_arms.append(min_node[0])

                min_heap: list = min_node[1]
                min_heap.remove(min_node[0])
                if len(min_node[1]) > 0:
                    self.top_1_selection_instance = SingleEliminationTop1Select(
                        feedback_mechanism=self.budgeted_feedback_mechanism,
                        arms_subset=min_node[1].copy(),
                        preference_estimate=self.preference_estimate,
                    )

                else:
                    self.heap.delete(min_node[0])
                self.rank_index += 1
        self.top_1_selection_instance.step()

    def explore(self) -> None:
        """Run exploration phase of the algorithm."""
        if self.k_top_ranked == 1:
            top_1_arm = self.top_1_selection_instance
            assert top_1_arm is not None
            top_1_arm.step()
            # In order to avoid incompatible type error, before the algorithm gives the winner.This
            # assertion is necessary for mypy since copeland_winner is defined as
            # Optional[int] in the TopOneSelection.
            winner = top_1_arm.get_condorcet_winner()
            assert winner is not None
            self.top_k_arms.append(winner)

        elif self.k_top_ranked > 1 and self.k_top_ranked == round(self.k_top_ranked):

            if self.algorithm_stage == 0:
                self._create_short_list()
            elif self.algorithm_stage == 1:

                self._create_heap()
            else:
                self._top_k_sorting()
        else:
            raise Exception("Invalid value for K")

    def step(self) -> None:
        """Execute one step of the algorithm."""
        if not self.exploration_finished():
            try:
                self.explore()
            except AlgorithmFinishedException:
                pass
        else:
            self.exploit()

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        Once this function returns ``True``, the algorithm will have finished
        computing a :term:`PAC` :term:`Copeland winner`.
        """
        return len(self.top_k_arms) == self.k_top_ranked

    def get_partial_ranking(self) -> Optional[List[int]]:
        """Return the Copeland winner given by the algorithm.

        Returns
        -------
        list
           The top-k Copeland winners in the set of arms given to the algorithm.
        """
        if not self.exploration_finished():
            return None
        return self.top_k_arms
