"""An implementation of the Sequential Elimination Algorithm."""

from typing import List
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.algorithms.interfaces import SingleCopelandProducer
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.stats.preference_estimate import PreferenceEstimate
from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException
import duelnlg.duelpy.util.utility_functions as utility


class SequentialElimination(SingleCopelandProducer, PacAlgorithm):
    r"""Implement the Sequential Elimination algorithm.

    The goal of this algorithm is to find an :math:`\epsilon`-maximum arm.

    The algorithm computes a :term:`PAC` estimation of the :term:`Copeland winner`.
    An arm is :math:`\epsilon`-maximum (where :math:`\epsilon = \epsilon_u-\epsilon_l`), if it is preferable to other arms with
    probability at least :math:`0.5-\epsilon`.

    If the anchor arm provided to the algorithm is a good anchor element, then there are only m elements
    for which element a is not :math:`\epsilon_l` preferable. This means, all other elements will be eliminated but
    among these :math:`m` elements, there can be at most :math:`m` changes of anchor element. Thus, there can be at most m rounds and
    hence we can bound total comparison rounds by :math:`\mathcal{O}(N + m^2)`. :math:`N` is the number of arms.

    Thus this :term:`PAC` algorithm reduces the comparisons to at most m elements which are not :math:`\epsilon_l` preferable and
    the remaining n-m elements are :math:`\epsilon_l` perferable and hence are removed with comparison complexity of
    :math:`\mathcal{O}(N)`.

     Refer to the paper :cite:`falahatgar2017maxing`.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        The number of steps that the algorithm is supposed to be run. Specify ``None`` for an infinite time horizon.
    failure_probability
        Determines the number of iterations that both arms are compared against. Corresponds to :math:`\delta` in
        :cite:`falahatgar2017maxing`. Default value is ``0.1``, as given in section 6.
    epsilon_lower
        Default value is ``0.0``. Refer to section 3.1.1 in :cite:`falahatgar2017maxing`.
    epsilon_upper
        Corresponds to :math:`\epsilon` with default value is ``0.5``, as given in section 3.1.1 in
        :cite:`falahatgar2017maxing`.
    arms_subset
        Represents the list of arms which is sent by other algorithms and is the subset from list of arms
        fetched from ``feedback_mechanism``.
    anchor_arm
        If none is provided, it is selected randomly from the list of arms provided to the algorithm.
        Otherwise, it represents the anchor arm extracted from ``feedback_mechanism.get_arms()``.
        A good anchor element is an arm for which every other arm (being :math:`\epsilon_l` preferable) is deemed worse
        and gets eliminated.

    Attributes
    ----------
    feedback_mechanism
    failure_probability
    preference_estimate

    Raises
    ------
    ValueError
        Raised when the value of upper epsilon is not greater than lower epsilon.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> import numpy as np
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.5],
    ...     [0.9, 0.5, 0.5],
    ... ])
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=np.random.RandomState())
    >>> sequential_elimination = SequentialElimination(feedback_mechanism, random_state=np.random.RandomState())
    >>> sequential_elimination.run()
    >>> preferred_arms = [1,2]
    >>> best_arm = sequential_elimination.get_copeland_winner()
    >>> best_arm in preferred_arms
    True
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        random_state: np.random.RandomState = None,
        time_horizon: Optional[int] = None,
        failure_probability: float = 0.1,
        epsilon_lower: float = 0.0,
        epsilon_upper: float = 0.05,
        arms_subset: Optional[List] = None,
        anchor_arm: Optional[int] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        if epsilon_upper <= epsilon_lower:
            raise ValueError("upper epsilon must be bigger than lower epsilon.")
        self._epsilon_upper = epsilon_upper
        self._epsilon_lower = epsilon_lower
        self.failure_probability = failure_probability
        if random_state is not None:
            self._random_state = random_state
        else:
            self._random_state = np.random.RandomState()
        self.preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms()
        )

        if arms_subset is None:
            self._remaining_arms: list = self.feedback_mechanism.get_arms()
        else:
            self._remaining_arms = arms_subset.copy()

        if anchor_arm is None:
            random_arms = utility.pop_random(
                self._remaining_arms, random_state=self._random_state
            )
            self._anchor_arm = random_arms.pop(0)

        else:
            self._anchor_arm = anchor_arm
            if self._anchor_arm in self._remaining_arms:
                self._remaining_arms.remove(self._anchor_arm)

    def explore(self) -> None:
        """Compare the current anchor arm against a randomly selected arm.

        The anchor arm is updated with the arm beating the current anchor arm and the new anchor arm is compared against
        the remaining arms step by step. Refer to section 3.1.1 in paper :cite:`falahatgar2017maxing`.
        """
        # randomly select a competing arm and after the duel remove that element from arms list.
        random_competing_arm = utility.pop_random(
            self._remaining_arms.copy(), random_state=self._random_state,
        )[0]

        try:
            comparison_result = self._is_competing_arm_better(
                competing_arm=random_competing_arm,
            )
        except AlgorithmFinishedException:
            return
        self._remaining_arms.remove(random_competing_arm)
        if comparison_result:
            # competing arm beats the anchor arm.
            self._anchor_arm = random_competing_arm

        # print ("winner: ", self._anchor_arm)

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        If no time horizon is provided, this coincides with ``is_finished``. Once
        this function returns ``True``, the algorithm will have finished
        computing a :term`PAC` :term:`Copeland winner`.
        """
        return len(self._remaining_arms) == 0

    def get_copeland_winner(self) -> Optional[int]:
        """Return the Copeland winner arm selected by the algorithm.

        Returns
        -------
        None
            If the algorithm has not concluded.
        int
            If the algorithm has found the winner.
        """

        return self._anchor_arm
        # if self.exploration_finished():
        #     return self._anchor_arm
        # else:
        #     return None

    def get_winner(self):
        return self.get_copeland_winner()

    def exploit(self) -> None:
        """Run one step of exploitation."""
        winner = self.get_winner()
        self.feedback_mechanism.duel(winner, winner)

    def step(self):
        if not self.exploration_finished():
            self.explore()
        else:
            self.exploit()

    def _is_competing_arm_better(self, competing_arm: int) -> bool:
        r"""Determine if competing arm beats anchor arm.

        The calibrated preference probability estimate (:math:`\hat{p}_{i,j}`) for competing arm against anchor arm
        refers to the probability that competing arm is preferred to anchor arm and is calculated based on the number of
        times competing arm has won divided by total number of duels between both arms. This value is updated after each
         duel between them.

        The confidence radius (:math:`\hat{c}`) is calculated such that with proof >= :math:`1-\delta`,
        :math:`\lvert \hat{p}_{i,j} - p_{i,j} \rvert < \hat{c}` after any number of comparisons. Here :math:`1-\delta`
        as mentioned in the paper :cite:`falahatgar2017maxing`, is called confidence value but we have referred it
        as the failure probability.

        The method returns True if :math:`\hat{p}_{i,j}`  :\ge math:`(\epsilon_u + \epsilon_l)/2` otherwise False is
        returned.
        For more details, please refer to appendix section Algorithm 9 in :cite:`falahatgar2017maxing`.

        Parameters
        ----------
        competing_arm
            Arm that challenges the current anchor arm.

        Raises
        ------
        AlgorithmFinishedException
            If the comparison budget is exceeded before the better arm could be
            determined.

        Returns
        -------
        bool
            Whether competing arm is better than anchor arm or not.

        """
        epsilon = (
            self._epsilon_upper - self._epsilon_lower
        )  # refer to :math:`\epsilon` in paper :cite:`falahatgar2017maxing`.
        epsilon_mean = (self._epsilon_upper + self._epsilon_lower) / 2
        confidence_radius = 0.5
        current_iteration_count = (
            0  # refer to variable 't' in paper :cite:`falahatgar2017maxing`
        )
        calibrated_preference_estimate = 0.0

        # number of rounds is selected in such a way that compare method selects the winner with :math:`1-\delta`
        # confidence. See Algorithm 4 of paper :cite:`falahatgar2017maxing`.
        rounds_for_iteration = int(
            2 * np.log(2 / self.failure_probability) / (np.power(epsilon, 2))
        )

        def prob_scaling(num_iteration: int) -> float:
            return np.square(2 * num_iteration)

        confidence_radius_fn = HoeffdingConfidenceRadius(
            self.failure_probability, prob_scaling
        )

        # compare two arms multiple times to get an estimate of their winnings. Refer to Algorithm 4 of paper
        # :cite:`falahatgar2017maxing`.
        # print ("rounds_for_iteration: ", rounds_for_iteration)
        # print ("comparing: ", competing_arm, self._anchor_arm)
        while (
            current_iteration_count < rounds_for_iteration
            and np.absolute(calibrated_preference_estimate - epsilon_mean)
            <= confidence_radius
        ):
            # print ("confidence_radius: ", confidence_radius)
            # print ("np.absolute(calibrated_preference_estimate)", np.absolute(calibrated_preference_estimate - epsilon_mean))
            if self.is_finished():
                raise AlgorithmFinishedException()
            current_iteration_count += 1
            feedback_result = self.feedback_mechanism.duel(
                competing_arm, self._anchor_arm
            )

            self.preference_estimate.enter_sample(
                competing_arm, self._anchor_arm, feedback_result
            )

            calibrated_preference_estimate = (
                self.preference_estimate.get_mean_estimate(
                    competing_arm, self._anchor_arm
                )
                - 0.5
            )
            confidence_radius = confidence_radius_fn(
                self.preference_estimate.get_num_samples(
                    competing_arm, self._anchor_arm
                )
            )
        # refer to algorithm of COMPARE of Algorithm 4 in paper :cite:`falahatgar2017maxing`.
        return calibrated_preference_estimate >= epsilon_mean
