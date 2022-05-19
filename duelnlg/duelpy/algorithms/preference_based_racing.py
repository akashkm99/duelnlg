"""Implementations of preference based racing algorithms."""
from typing import List
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import TopKArmsProducer
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius


class PreferenceBasedRacing(TopKArmsProducer):
    r"""Preference based racing algorithm superclass.

    This algorithm is taken from :cite:`busa2013top`, given there as `Algorithm 1`.

    The goal of this algorithm is to find the top-k arms while minimizing the exact sample complexity.

    No assumptions about the arms are necessary.

    The algorithm keeps track of which arm pairs are active. In each round all of these are queried once, then a sampling function determines which arms are selected and how the active arm pairs are updated.
    It terminates if no more arm pairs are active, or the queries to any arm pair exceeds the maximum comparison parameter.


    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Optional, used for random choices in the algorithm.
    num_top_arms
        How many arms should be returned. The :math:`k` of top-:math:`k` arms.
    max_comparisons
        An upper bound on the comparisons for each arm pair, see :math:`n_{max}` in the paper.
    failure_probability
        An upper bound on the acceptable probability to fail, also called :math:`\delta` in :cite:`busa2013top`.


    Attributes
    ----------
    feedback_mechanism
    failure_probability
    random_state
    preference_estimate
        Estimates for arm preferences.
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        num_top_arms: int = 1,
        max_comparisons: int = 1000,
        failure_probability: float = 0.1,
    ):
        super().__init__(
            feedback_mechanism=feedback_mechanism, time_horizon=time_horizon
        )
        self.failure_probability = failure_probability

        self.num_top_arms = num_top_arms

        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        num_arms = self.feedback_mechanism.get_num_arms()

        self.max_comparisons = max_comparisons

        # pylint: disable=unused-argument
        def probability_scaling(num_samples: int) -> float:
            return 2 * num_arms ** 2 * self.max_comparisons

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability, probability_scaling
        )
        self.preference_estimate = PreferenceEstimate(num_arms, confidence_radius)
        self._selected_arms: List[int] = []
        self._discarded_arms: List[int] = []
        self._active_arm_pairs = [
            (i, j) for i in range(num_arms) for j in range(num_arms) if i != j
        ]

    def sampling(self) -> None:
        """Update the selected arms.

        This should be overriden by subclasses.
        """
        raise NotImplementedError

    def explore(self) -> None:
        """Explore arms by advancing the sorting algorithm."""
        for arm_1, arm_2 in self._active_arm_pairs:
            self.preference_estimate.enter_sample(
                arm_1, arm_2, self.feedback_mechanism.duel(arm_1, arm_2)
            )

        self.sampling()

    def exploit(self) -> None:
        """Exploit by choosing two top arms randomly."""
        self.feedback_mechanism.duel(
            self.random_state.choice(self._selected_arms),
            self.random_state.choice(self._selected_arms),
        )

    def step(self) -> None:
        """Execute one step of the algorithm."""
        if self.exploration_finished():
            self.exploit()
        else:
            self.explore()

    def get_top_k(self) -> Optional[List[int]]:
        """Get the arm with the highest probability of being the first in a ranking of the arms.

        Returns
        -------
        Optional[int]
            The best arm, None if has not been calculated yet.
        """
        if self.exploration_finished():
            return self._selected_arms
        else:
            return None

    def exploration_finished(self) -> bool:
        """Determine whether the best arm has been found."""
        return (
            len(self._active_arm_pairs) == 0
            or np.max(self.preference_estimate.wins + self.preference_estimate.wins.T)
            >= self.max_comparisons
        )

    def is_finished(self) -> bool:
        """Determine if the algorithm is finished.

        Returns
        -------
        bool
            Whether the algorithm is finished.
        """
        if self.time_horizon is None:
            return self.exploration_finished()
        else:
            return super().is_finished()


class CopelandPBR(PreferenceBasedRacing):
    r"""Implementation of Copeland Ranking Preference Based Racing algorithm.

    The goal of this algorithm is to find the best :math:`k` arms with respect to a :term:`Copeland ranking`.

    It makes no assumptions about the arms, except that they can be compared.

    The expected number of comparisons is bounded by :math:`\sum_{i \neq j}\left\lceil \frac{1}{2 \Delta_{i,j}^2} \log \frac{2N^2 n_max}{\delta}\right\rceil`. :math:`N` is the number of arms, the sum iterates over all pairs of arms without self comparisons. :math:`\Delta_{i,j}+1/2` is the probability of arm :math:`i` winning against arm :math:`j` and :math:`\delta` is the failure probability.

    The algorithm keeps track of all pairwise probabilities and stops sampling them if the better arm can be determined with confidence. The :math:`k` arms with the highest estimated :term:`Copeland score`.
    See :cite:`busa2013top` for more details.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Optional, used for random choices in the algorithm.
    num_top_arms
        How many arms should be returned. The :math:`k` of top-:math:`k` arms.
    max_comparisons
        An upper bound on the comparisons for each arm pair, see :math:`n_{max}` in the paper.
    failure_probability
        An upper bound on the acceptable probability to fail, called :math:`\delta` in :cite:`busa2013top`.


    Attributes
    ----------
    feedback_mechanism
    random_state
    failure_probability
    sorting_algorithm
    preference_estimate
        Estimates for arm preferences.

    Examples
    --------
    Find the best 2 arms in this example:

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
    >>> test_object = CopelandPBR(feedback_mechanism, random_state=random_state, num_top_arms=2, failure_probability=0.1)
    >>> test_object.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> top_k = test_object.get_top_k()
    >>> top_k, comparisons
    ([4, 2], 614)
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        num_top_arms: int = 1,
        max_comparisons: int = 1000,
        failure_probability: float = 0.1,
    ):
        super().__init__(
            feedback_mechanism=feedback_mechanism,
            time_horizon=time_horizon,
            random_state=random_state,
            num_top_arms=num_top_arms,
            max_comparisons=max_comparisons,
            failure_probability=failure_probability,
        )

    def sampling(self) -> None:
        """Update the selected arms.

        This should be overriden by subclasses.
        """
        lower_estimate = (
            self.preference_estimate.get_lower_estimate_matrix().preferences
        )
        upper_estimate = (
            self.preference_estimate.get_upper_estimate_matrix().preferences
        )
        num_arms = self.feedback_mechanism.get_num_arms()
        cpld_lower_bound = num_arms - np.sum(
            upper_estimate < 1 / 2, axis=0
        )  # beaten by
        cpld_upper_bound = np.sum(lower_estimate > 1 / 2, axis=0)  # beats

        score = cpld_lower_bound[None, :] < cpld_upper_bound[:, None]
        selected_arms = num_arms - self.num_top_arms < np.sum(score, axis=0)
        discarded_arms = self.num_top_arms < np.sum(score, axis=1)
        selected_or_discarded = np.logical_or(selected_arms, discarded_arms)
        new_active_pairs = []
        for arm_1, arm_2 in self._active_arm_pairs:
            if (
                not (selected_or_discarded[arm_1] and selected_or_discarded[arm_2])
                and lower_estimate[arm_1, arm_2] < 1 / 2 < upper_estimate[arm_1, arm_2]
            ):
                new_active_pairs.append((arm_1, arm_2))
        self._active_arm_pairs = new_active_pairs
        self._selected_arms = list(
            np.argpartition(
                self.preference_estimate.get_mean_estimate_matrix().get_copeland_scores(),
                -self.num_top_arms,
            )[-self.num_top_arms :]
        )


class BordaPBR(PreferenceBasedRacing):
    r"""Implementation of the Borda Preference Based Racing algorithm.

    The goal of this algorithm is to find the best :math:`k` arms with respect to a :term:`Borda ranking`.

    It makes no assumptions about the arms, except that they can be compared.

    See Theorem 2 in See :cite:`busa2013top` for more details. for a bound on the pairwise comparisons.

    The paper calls this algorithm sum of expectations (SE), which is the same as the :term:`Borda score`.
    The algorithm keeps track of all pairwise probabilities and stops sampling them if the better arm can be determined with confidence. The :math:`k` arms with the highest estimated :term:`Borda score`.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Optional, used for random choices in the algorithm.
    num_top_arms
        How many arms should be returned. The :math:`k` of top-:math:`k` arms.
    max_comparisons
        An upper bound on the comparisons for each arm pair, see :math:`n_{max}` in the paper.
    failure_probability
        An upper bound on the acceptable probability to fail, called :math:`\delta` in :cite:`busa2013top`.


    Attributes
    ----------
    feedback_mechanism
    random_state
    failure_probability
    sorting_algorithm
    preference_estimate
        Estimates for arm preferences.

    Examples
    --------
    Find the best 2 arms in this example:

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
    >>> test_object = BordaPBR(feedback_mechanism, random_state=random_state, num_top_arms=2, failure_probability=0.1, max_comparisons=100)
    >>> test_object.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> top_k = test_object.get_top_k()
    >>> top_k, comparisons
    ([4, 2], 1000)
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        num_top_arms: int = 1,
        max_comparisons: int = 100,
        failure_probability: float = 0.1,
    ):
        super().__init__(
            feedback_mechanism=feedback_mechanism,
            time_horizon=time_horizon,
            random_state=random_state,
            num_top_arms=num_top_arms,
            max_comparisons=max_comparisons,
            failure_probability=failure_probability,
        )

        self._discarded_arms = np.full(self.feedback_mechanism.get_num_arms(), False)

    # pylint: disable=too-many-locals
    def sampling(self) -> None:
        """Update the selected arms."""
        num_arms = self.feedback_mechanism.get_num_arms()
        active_arms = np.full(num_arms, False)  # G in the paper
        for pair in self._active_arm_pairs:
            active_arms[pair[0]] = True
            active_arms[pair[1]] = True
        selected_arms = np.logical_not(
            np.logical_or(self._discarded_arms, active_arms)
        )  # ~B in the paper
        lower_borda = (
            self.preference_estimate.get_lower_estimate_matrix().get_borda_scores()
        )
        upper_borda = (
            self.preference_estimate.get_upper_estimate_matrix().get_borda_scores()
        )
        mean_borda = (
            self.preference_estimate.get_mean_estimate_matrix().get_borda_scores()
        )

        # reduced problem:
        num_current_arms = np.sum(active_arms)
        num_top_arms_left = self.num_top_arms - np.sum(selected_arms)

        cross_comparison = upper_borda[:, None] < lower_borda[None, :]
        active_mask = np.tile(active_arms, (num_arms, 1))
        worse_arms_num = np.sum(np.logical_and(cross_comparison, active_mask.T), axis=0)
        better_arms_num = np.sum(np.logical_and(cross_comparison, active_mask), axis=1)
        # add arm to selected if it is in the top arms, top arms means that for the arms which are currently investigated and the number of top arms which have not been found
        selected_arms = np.logical_or(
            selected_arms, (num_current_arms - num_top_arms_left) < worse_arms_num
        )
        # discard arm if more arms are better than the amount of top arms left to find
        self._discarded_arms = np.logical_or(
            self._discarded_arms, num_top_arms_left < better_arms_num
        )

        new_active_pairs = []
        for arm_1, arm_2 in self._active_arm_pairs:
            if not (selected_arms[arm_1] or self._discarded_arms[arm_1]):
                new_active_pairs.append((arm_1, arm_2))
        self._active_arm_pairs = new_active_pairs
        self._selected_arms = list(
            np.argpartition(mean_borda, -self.num_top_arms)[-self.num_top_arms :]
        )


class RandomWalkPBR(PreferenceBasedRacing):
    r"""Implementation of the Random Walk Preference Based Racing algorithm.

    The goal of this algorithm is to find the best :math:`k` arms with respect to a random walk ranking.

    It makes no assumptions about the arms, except that they can be compared.

    See Theorem 2 in :cite:`busa2013top` for a bound on the pairwise comparisons.

    This sampling strategy treats the arms as a markov chain. The state transitions are proportional to the probabilities of an arm losing to the respective other arms.
    The resulting stationary distribution is used for ranking the arms and finding the k best ones.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Optional, used for random choices in the algorithm.
    num_top_arms
        How many arms should be returned. The :math:`k` of top-:math:`k` arms.
    max_comparisons
        An upper bound on the comparisons for each arm pair, see :math:`n_{max}` in the paper.
    failure_probability
        An upper bound on the acceptable probability to fail, called :math:`\delta` in :cite:`busa2013top`.


    Attributes
    ----------
    feedback_mechanism
    random_state
    failure_probability
    sorting_algorithm
    preference_estimate
        Estimates for arm preferences.

    Examples
    --------
    Find the best 2 arms in this example:

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
    >>> test_object = RandomWalkPBR(feedback_mechanism, random_state=random_state, num_top_arms=2, failure_probability=0.1, max_comparisons=200)
    >>> test_object.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> top_k = test_object.get_top_k()
    >>> top_k, comparisons
    ([2, 4], 1102)
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        num_top_arms: int = 1,
        max_comparisons: int = 100,
        failure_probability: float = 0.1,
    ):
        super().__init__(
            feedback_mechanism=feedback_mechanism,
            time_horizon=time_horizon,
            random_state=random_state,
            num_top_arms=num_top_arms,
            max_comparisons=max_comparisons,
            failure_probability=failure_probability,
        )

    # pylint: disable=too-many-locals
    def sampling(self) -> None:
        """Update the selected arms."""
        num_arms = self.feedback_mechanism.get_num_arms()
        mean_matrix = self.preference_estimate.get_mean_estimate_matrix().preferences
        # each column is summed for normalization
        column_sum = np.sum(mean_matrix, axis=0)
        transition_matrix = (
            mean_matrix / column_sum
        )  # normalize over columns to get transposed transition matrix
        # there is an error in the paper, this confidence should be correct
        confidence_matrix = self.preference_estimate.get_radius_matrix()
        confidences = num_arms * 3 * np.max(confidence_matrix, axis=0) / column_sum

        # eigenvector v of transition matrix with (maximum) eigenvalue 1
        # ~ stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)
        principal_eigenvector = eigenvectors[
            :, np.argmax(eigenvalues)
        ]  # eigenvectors are columns, not rows here
        # calculate  A# = (I - S + 1vT)^-1 - 1vT
        ev_matrix = np.tile(principal_eigenvector, (num_arms, 1))
        temp = np.eye(num_arms) - transition_matrix.T + ev_matrix
        temp2 = np.linalg.inv(temp) - ev_matrix
        # the two a=num_top_arm th and b=num_top_arm+1 th largest elements in eigenvector
        v_sorted = np.abs(principal_eigenvector.copy())
        v_sorted.sort()
        difference = np.abs(
            v_sorted[self.num_top_arms] - v_sorted[self.num_top_arms - 1]
        )
        max_conf = 2 * np.max(np.abs(confidences)) * np.max(np.abs(temp2))

        if difference > max_conf:
            # termination condition
            self._active_arm_pairs = []
        else:
            pair_confidences = confidence_matrix / column_sum
            max_value = np.max(pair_confidences)
            self._active_arm_pairs = np.argwhere(pair_confidences == max_value)
        self._selected_arms = list(
            np.argwhere(
                np.abs(principal_eigenvector) >= v_sorted[-self.num_top_arms]
            ).flatten()[:2]
        )
