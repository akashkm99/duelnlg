"""An implementation of the Merge Relative Upper Confidence Bound (Merge-RUCB) algorithm."""
from typing import List
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import CondorcetProducer
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.util.utility_functions import argmax_set


class MergeRUCB(CondorcetProducer):
    r"""Implementation of the Merge Relative Upper Confidence Bound algorithm.

    The goal of the algorithm is to find the :term:`Condorcet winner` while incurring minimum regret with minimum comparisons
    between the arms.

    It is assumed that the :term:`Condorcet winner` exists.

    The regret is bounded by :math:`\mathcal{O}\left(N \log T\right)` where :math:`N` is the number of arms and :math:`T` is the time horizon.

    The algorithm described in the paper :cite:`zoghi2015mergerucb`. Dueling bandit algorithms have to learn
    something about the preference relation by pairwise comparison. Thus, they often have a worst-case sample
    complexity that scales with the square of the arms. MergeRUCB avoids this with a divide-and-conquer strategy: It
    divides the set of arms (batch) into multiple sub-sets (small batches), "solves" these smaller problems,
    and then merges the results. The batch of arms is divided into multiple small batches based on the partition
    size (:math:`P`), which was decided to be greater than or equal to 4. Then these small batches are dueled, and the weak-arms are
    dropped from the small batches. After pruning of arms from the small-batch, all the batches are sorted and merged
    so that the new set of small batches lie in the range either :math:`\frac{P}{2}` or :math:`\frac{3P}{2}`.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    exploratory_constant
        The confidence radius grows proportional to the square root of this value.
    partition_size
        Initial size of the batches.
    failure_probability
        Probability of failure.
    random_state
        Optional, used for random choices in the algorithm.

    Attributes
    ----------
    feedback_mechanism
    partition_size
    random_state
    preference_estimate
        Stores estimates of arm preferences

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> from duelnlg.duelpy.stats.metrics import WeakRegret
    >>> from duelnlg.duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3, 0.2, 0.2],
    ...     [0.9, 0.7, 0.5, 0.8, 0.9],
    ...     [0.9, 0.8, 0.2, 0.5, 0.2],
    ...     [0.9, 0.8, 0.1, 0.8, 0.5]
    ... ])
    >>> random_state=np.random.RandomState(43)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"weak_regret": WeakRegret(preference_matrix)}
    ... )
    >>> test_object = MergeRUCB(
    ...  feedback_mechanism=feedback_mechanism,
    ...  exploratory_constant=1.01,
    ...  time_horizon=200,
    ...  random_state=random_state,
    ...  failure_probability=0.01)
    >>> test_object.run()
    >>> test_object.get_condorcet_winner()
    2
    >>> np.round(np.sum(feedback_mechanism.results["weak_regret"]), 2)
    28.8
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        partition_size: int = 4,
        failure_probability: float = 0.01,
        exploratory_constant: float = 1.01,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.partition_size = partition_size
        self.exploratory_constant = exploratory_constant
        self.time_step = 0
        self.stage = 0

        self.preference_estimate = PreferenceEstimate(
            num_arms=feedback_mechanism.get_num_arms()
        )
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        self.confidence = np.ceil(
            (
                (4 * self.exploratory_constant - 1)
                * self.feedback_mechanism.get_num_arms() ** 2
                / ((2 * self.exploratory_constant - 1) * failure_probability)
            )
            ** (1.0 / (2 * self.exploratory_constant - 1))
        )

        # partition the arm into the batches.
        self.arm_batches: List[List] = []
        self._set_arm_batches()
        # confirms the batch size is between p/2 or 3p/2
        self._merge()

    def _set_arm_batches(self) -> None:
        """Divide the arms into the batches."""
        num_of_batches = np.floor(
            self.feedback_mechanism.get_num_arms() / self.partition_size
        )
        arms_shuffled_list = list(self.feedback_mechanism.get_arms())
        self.random_state.shuffle(arms_shuffled_list)
        self.arm_batches = np.array_split(arms_shuffled_list, num_of_batches + 1)

    def _update_confidence_radius(self) -> None:
        """Update the confidence radius using latest failure probability.

        Failure probability for the upper confidence bound is :math:`1/t+c^(2 * alpha)`
        where :math:`t` is the current round of the algorithm and :math:`alpha` is the
        exploratory constant.

        Refer :cite:`Zoghi2015bMergeRUCB` for further details.
        """
        failure_probability = 1 / (
            (self.time_step + self.confidence) ** (2 * self.exploratory_constant)
        )
        confidence_radius = HoeffdingConfidenceRadius(failure_probability)
        self.preference_estimate.set_confidence_radius(confidence_radius)

    def _prune_arm(
        self, batch_index: int, upper_confidence_bound_matrix: np.array
    ) -> None:
        """Remove the arm which has the least potential to win.

        Parameters
        ----------
        batch_index
            The index of the batch.
        upper_confidence_bound_matrix
            The upper confidence bound value of all the arms in matrix representation.
        """
        batch = self.arm_batches[batch_index]
        final_batch = set(batch.copy())
        if len(batch) == 0:
            return
        for arm_k in batch:
            for arm_l in batch:
                if upper_confidence_bound_matrix[arm_k][arm_l] < 0.5:
                    final_batch.remove(arm_k)
        self.arm_batches[batch_index] = list(final_batch)

    def _num_arms_batches(self) -> int:
        """Calculate the number of arms in all the batches."""
        number_of_arms = [len(each_batch) for each_batch in self.arm_batches]
        return sum(number_of_arms)

    def step(self) -> None:
        """Run one round of an algorithm."""
        self.time_step += 1
        # no more stages
        if self._num_arms_batches() == 1:
            return
        self.stage += 1
        self._update_confidence_radius()
        # number of batches present in the current stage
        num_of_batches = len(self.arm_batches)
        batch_index = np.mod(self.time_step, num_of_batches)

        upper_confidence_bound_matrix = (
            self.preference_estimate.get_upper_estimate_matrix().preferences
        )
        # pruning of the arms from the batches
        self._prune_arm(batch_index, upper_confidence_bound_matrix)

        # The particular arm batch should contain at least two different arms to duel.
        if len(self.arm_batches[batch_index]) != 1:
            # choose potential champion arm randomly
            arm_c = self.random_state.choice(self.arm_batches[batch_index])

            upper_confidences = upper_confidence_bound_matrix[:, arm_c]
            # Select challenger arm i.e. arm_d (other than arm_c) whose upper confidence bound is maximum with
            # reference to arm_c.
            non_challenger_arm = set(self.feedback_mechanism.get_arms()) - set(
                self.arm_batches[batch_index]
            )
            non_challenger_arm.add(arm_c)
            arm_d = self.random_state.choice(
                argmax_set(upper_confidences, list(non_challenger_arm))
            )

            # Compare potential champion (arm_c) and challenger (arm_d).
            self.preference_estimate.enter_sample(
                arm_c, arm_d, self.feedback_mechanism.duel(arm_c, arm_d)
            )

        # merging logic
        self._merge()

    def _merge(self) -> None:
        """Merge the two batches to get size between p/2 or 3p/2."""
        if (
            self._num_arms_batches()
            <= self.feedback_mechanism.get_num_arms() / (2 ** self.stage)
            and len(self.arm_batches) > 1
        ):
            self._merge_batches()
            if min([len(a) for a in self.arm_batches]) <= 0.5 * self.partition_size:
                self._merge_batches()
            self.stage += 1

    def _merge_batches(self) -> None:
        """Merge the two batches whose sizes is less than p/2 or 3p/2."""
        old_batches = self.arm_batches[:]
        old_batches.sort(key=lambda x: (len(x), self.random_state.uniform()))
        self.arm_batches = []
        i = 0
        j = len(old_batches) - 1
        while i <= j:
            if i == j:
                self.arm_batches.append(old_batches[i])
                return
            elif len(old_batches[i]) + len(old_batches[j]) >= self.partition_size * 1.5:
                self.arm_batches.append(old_batches[j])
                j -= 1
            elif len(old_batches[i]) + len(old_batches[j]) >= self.partition_size * 0.5:
                self.arm_batches.append(
                    np.asarray(list(old_batches[i]) + list(old_batches[j]))
                )
                i += 1
                j -= 1
            else:
                old_batches[j].append(old_batches[i])
                i += 1

    def get_condorcet_winner(self) -> Optional[int]:
        """Determine a Condorcet winner using Merge_RUCB algorithm.

        Returns
        -------
        Optional[int]
            The index of a Condorcet winner, if existent, among the given arms.
        """
        return (
            self.preference_estimate.get_mean_estimate_matrix().get_condorcet_winner()
        )
