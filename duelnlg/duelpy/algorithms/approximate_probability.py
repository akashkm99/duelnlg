"""Implementation of the Approximate probability algorithm."""

from typing import List
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.interfaces import PreferenceMatrixProducer
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix


class ApproximateProbability(PreferenceMatrixProducer):
    r"""Implementation of the Approximate probability algorithm.

    The goal is to approximate the pairwise preference matrix between all arms.

    The algorithm assumes a :term:`total order` over the existing arms and that :term:`strong stochastic
    transitivity` and :term:`stochastic triangle inequality` hold. Additionally, a :math:`\frac{\epsilon}{8}`-approximate ranking over the arms has to be provided.

    The bound on the expected regret is given as :math:`\mathcal{O}\left(\frac{N\min\left\{N,\frac{1}{\epsilon}\right\}}{\epsilon^2}\right)`,
    where :math:`N` is the number of arms and :math:`\epsilon` is the targeted
    estimation accuracy.

    The approximate probability algorithm is based on `Algorithm 5` in :cite:`falahatgar2018limits`.
    It's an (:math:`\epsilon, \delta`)-:term:`PAC` algorithm with :math:`\delta = \frac{1}{N^2}`
    where :math:`N` is the number of arms.

    The algorithm takes an ordered set of arms and approximates all pairwise probabilities to
    an accuracy of :term:`\epsilon`. This ranking could be the result of the :term:`BinarySearchRanking<duelpy.algorithms.binary_search_ranking.BinarySearchRanking>` algorithm.
    Probabilities are calculated starting with the best arm against all others and then iterating down the ranking order. The result is guaranteed to be consistent with the ranking.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    epsilon
        The optimality of the winning arm. Corresponds to :math:`\epsilon` in :cite:`falahatgar2018limits`.
        Default value is ``0.05``, which has been used in the experiments in :cite:`falahatgar2018limits`.
    order_arms
        A :math:`\frac{\epsilon}{8}` ranking over the arms.

    Attributes
    ----------
    feedback_mechanism
    tournament_arms
        The arms that are still in the tournament.
    estimate_pairwise_probability
    epsilon
    comparison_arm
        Iterate the number of comparisons of a specific arm.
    order_arms

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.9, 0.7, 0.5],
    ...     [0.9, 0.5, 0.3],
    ...     [0.5, 0.1, 0.1]
    ... ])
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=np.random.RandomState(100))
    >>> test_object = ApproximateProbability(feedback_mechanism, epsilon=0.05, order_arms=[1, 0, 2])
    >>> test_object.run()
    >>> test_object.get_preference_matrix()
    array([[0.5, 0. , 0. ],
           [0.7, 0.5, 0. ],
           [0.7, 0.5, 0.5]])
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        order_arms: List[int],
        epsilon: float = 0.05,
        time_horizon: Optional[int] = None,
    ):
        self.tournament_arms = feedback_mechanism.get_num_arms()
        self.comparison_arm = 0
        super().__init__(feedback_mechanism, time_horizon)
        self.epsilon = epsilon
        self.order_arms = order_arms
        self._estimate_pairwise_probability: list = np.zeros(
            (self.tournament_arms, self.tournament_arms)
        )

        self.preference_estimate = PreferenceEstimate(
            num_arms=feedback_mechanism.get_num_arms()
        )

    def estimate_probabilities_against_first_arm(self) -> None:
        """Run one step of comparison.

        The first ranked and the other arms are dueled repeatedly, determining their preference probabilities.
        """
        self._estimate_pairwise_probability[0][0] = 0.5
        arm_i = self.order_arms[0]
        for arm_index in range(1, self.tournament_arms):
            arm_j = self.order_arms[arm_index]
            self._estimate_pairwise_probability[arm_index][0] = self.duel_repeatedly(
                arm_i, arm_j
            )

            if (
                self._estimate_pairwise_probability[arm_index][0]
                < self._estimate_pairwise_probability[arm_index - 1][0]
            ):
                self._estimate_pairwise_probability[arm_index][
                    0
                ] = self._estimate_pairwise_probability[arm_index - 1][0]

    def estimate_pairwise_probabilities(self, rank_1: int) -> None:
        """Run second step of comparison.

        It compares arm :math:`i` and arm :math:`j` multiple times and estimates the
        pairwise probability.
        """
        self._estimate_pairwise_probability[rank_1][rank_1] = 0.5
        for rank_2 in range(rank_1 + 1, self.tournament_arms):
            if (
                self._estimate_pairwise_probability[rank_2 - 1][rank_1]
                == self._estimate_pairwise_probability[rank_2][rank_1 - 1]
            ):

                self._estimate_pairwise_probability[rank_2][
                    rank_1
                ] = self._estimate_pairwise_probability[rank_2 - 1][rank_1]
            else:
                arm_i = self.order_arms[rank_2]
                arm_j = self.order_arms[rank_1]

                self._estimate_pairwise_probability[rank_2][
                    rank_1
                ] = self.duel_repeatedly(arm_i, arm_j)

    def duel_repeatedly(self, arm_i: int, arm_j: int) -> float:
        """Determine the preferred arm by repeated comparison.

        It calculates the number of times arm :math:`i` won against other arms in the set,
        and return the estimate pairwise probability.
        """
        compare_range = (int)(
            (16 / self.epsilon ** 2) * np.log(self.tournament_arms ** 4)
        )
        number_of_win_arm_j = 0
        for _ in range(compare_range):
            win_j = self.feedback_mechanism.duel(arm_j, arm_i)
            if win_j:
                number_of_win_arm_j += 1

        # approximate_probability corresponds to \hat\tilde p and is the estimated
        # win-fraction rounded to the nearest multiple of epsilon
        return (
            np.round(number_of_win_arm_j / compare_range / self.epsilon) * self.epsilon
        )

    def step(self) -> None:
        """Take multiple samples per step in the algorithm."""
        if self.comparison_arm == 0:
            self.estimate_probabilities_against_first_arm()
        else:
            self.estimate_pairwise_probabilities(self.comparison_arm)

        self.comparison_arm += 1

    def is_finished(self) -> bool:
        """Determine if the algorithm is finished.

        If the comparison arm is greater than tournament arms then it will terminate.
        """
        return self.comparison_arm >= self.tournament_arms

    def get_preference_matrix(self) -> Optional[PreferenceMatrix]:
        """Return the computed preference matrix if it is ready.

        Returns
        -------
        Optional[PreferenceMatrix]
            The estimated pairwise preference matrix or ``None`` if the result
            is not ready.
        """
        return (
            PreferenceMatrix(self._estimate_pairwise_probability)
            if self.is_finished()
            else None
        )
