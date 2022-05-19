"""PAC Borda winner with Successive Elimination with Comparison Sparsity algorithm."""
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from numpy.random.mtrand import RandomState

from duelnlg.duelpy.algorithms.interfaces import BordaProducer
from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException
from duelnlg.duelpy.util.feedback_decorators import BudgetedFeedbackMechanism


class SuccessiveElimination(BordaProducer, PacAlgorithm):
    r"""Implements the Successive Elimination with Comparison Sparsity algorithm.

    The goal of the algorithm is to find a :math:`\delta`-:term:`PAC` :term:`Borda winner`.

    The algorithm has a sample complexity of :math:`\mathcal{O}(n \log(n))` and succeeds with probability at least
    :math:`1-\delta`. Here, :math:`n` is the total number of arms and :math:`\delta` is the failure probability.

    It is assumed that a unique :term:`Borda winner` exists and that there are no ties between
    arms, i.e. the probability of any arm winning against another is never :math:`\frac{1}{2}`.

    The algorithm presented in :cite:`jamieson2015sparse` works on the exploration problem of finding the :term:`Borda winner` from noisy
    comparisons in a sparsity model. A sparsity model assumes a small set of top candidates that are similar to each
    other, and a large set of irrelevant candidates that would always lose in a pairwise comparison with one of the
    top candidates. The algorithm exploits this sparsity model to find the :term:`Borda winner` using fewer samples than
    standard algorithms. The algorithm implements the successive elimination strategy given in :cite:`even2006action`
    with the Borda reduction and an additional elimination criterion that exploits sparsity. The algorithm maintains
    an active set of arms (:math:`A_t` in :cite:`jamieson2015sparse`) of potential Borda winners. At
    each round :math:`t`, the algorithm chooses an arm uniformly at random (:math:`I_t` in
    :cite:`jamieson2015sparse`) from the set and compares it with all the arms in the active set. A parameter
    ``time_gate`` (:math:`T_0` in :cite:`jamieson2015sparse`) is specified to guarantee that all arms with sufficiently
    large Borda score gaps :math:`s_1 - s_i` are eliminated by round :math:`T_0`. This condition is fulfilled by
    `condition 2` in :cite:`jamieson2015sparse`. Once :math:`t>T_0`, i.e., ``round`` > ``time_gate``,
    `condition 1` also becomes active and the algorithm starts removing the arms with large partial Borda gaps,
    exploiting the assumption that the top arms can be distinguished by comparisons with a sparse set of
    other arms.

    The algorithm terminates when only one arm remains.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        The total number of rounds.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    sparsity_level
        Refers to :math:`k` in `Algorithm 1` in :cite:`jamieson2015sparse`. The assumed size of the sparsity set.
        This is the set of similar and nearly-optimal arms which are easy to
        differentiate from all other arms. Should be a value between :math:`1` and :math:`n - 2`, where
        :math:`n` is the size of the set of arms. Defaults to ``3`` if at least :math:`5` arms are available,
        :math:`n - 2` if only :math:`3` or :math:`4` arms are available. This corresponds to :math:`k` in
        :cite:`jamieson2015sparse`.
    failure_probability
        Refer to :math:`\delta` in :cite:`jamieson2015sparse`.
    time_gate
        It is specified to guarantee that all arms with sufficiently large Borda gaps are
        eliminated when the number of rounds becomes greater than ``time_gate``. `Theorem 2` in
        :cite:`jamieson2015sparse` specifies the requirement of ``time_gate`` parameter. Corresponds to
        :math:`T_0` in :cite:`jamieson2015sparse`. The paper :cite:`jamieson2015sparse` suggests :math:`T_0 = 0`
        based on their experiments in `section 5`.

    Attributes
    ----------
    random_state
    feedback_mechanism
    time_horizon
    failure_probability
    time_gate
    round
    sparsity_level
    borda_scores_array
        An array which stores the borda scores of each arm.
    confidence_factor
        Refer to :math:`C_t` in :cite:`jamieson2015sparse`.
    current_working_set
        The set with possible Borda winners from the set :math:`N`. Corresponds to :math:`A_t` in `Algorithm 1` in
        :cite:`jamieson2015sparse`.

    Raises
    ------
    ValueError
        Raised when the given sparsity level is invalid or the number of arms is too small to
        choose a default value. See the documentation of the ``sparsity_level`` parameter for
        details.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ... [0.5, 0.1, 0.1, 0.1, 0.1],
    ... [0.9, 0.5, 0.3, 0.2, 0.2],
    ... [0.9, 0.7, 0.5, 0.8, 0.9],
    ... [0.9, 0.8, 0.2, 0.5, 0.2],
    ... [0.9, 0.8, 0.1, 0.8, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(43)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=random_state)
    >>> secs = SuccessiveElimination(feedback_mechanism=feedback_mechanism, random_state=random_state, failure_probability=0.1)
    >>> secs.run()
    >>> borda_winner = secs.get_borda_winner()
    >>> borda_winner
    2
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: RandomState = None,
        sparsity_level: Optional[int] = None,
        failure_probability: float = 0.1,
        time_gate: int = 0,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            np.random.RandomState() if random_state is None else random_state
        )
        # Since this algorithm is a PAC algorithm, we use "BudgetedFeedbackMechanism" to avoid overflow the duels
        # w.r.t the time_horizon (if it is given).
        if self.time_horizon is not None:
            self.feedback_mechanism = BudgetedFeedbackMechanism(
                feedback_mechanism=feedback_mechanism,
                max_duels=self.time_horizon - feedback_mechanism.get_num_duels(),
            )
        else:
            self.feedback_mechanism = feedback_mechanism
        # The sparsity level, as recommended by the authors of the algorithm should be ``5`` for typical problems.
        # However, for ``n`` number of arms in the preference matrix, if n<5, this fails the condition of sparsity
        # level in [n-2]. Therefore, extra constraints are added for sparsity level. Use highest level of sparsity
        # level, i.e., sparsity_level = n - 2 when 3<= n <=6. If the user defines sparsity level value greater than
        # n-2, for n<=2, a ValueError will be raised.
        if sparsity_level is None:
            if self.feedback_mechanism.get_num_arms() > 6:
                sparsity_level = 5
            elif 3 <= self.feedback_mechanism.get_num_arms() <= 6:
                sparsity_level = self.feedback_mechanism.get_num_arms() - 2
            elif self.feedback_mechanism.get_num_arms() <= 2:
                raise ValueError(
                    "Value of sparsity level must be between 1 and (number of arms - 2)"
                )
        elif (
            sparsity_level > self.feedback_mechanism.get_num_arms() - 2
            or self.feedback_mechanism.get_num_arms() <= 2
        ):
            raise ValueError(
                "Value of sparsity level must be between 1 and (number of arms - 2)"
            )
        self.sparsity_level = sparsity_level
        self.failure_probability = failure_probability
        self.time_gate = time_gate
        self.preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms()
        )
        self.current_working_set = self.feedback_mechanism.get_arms()
        self.borda_scores_array = np.zeros(
            feedback_mechanism.get_num_arms(), dtype=float
        )
        self.confidence_factor: float = 0
        self.round: int = 1

    def explore(self) -> None:
        """Run one step of exploration."""
        # Update and set the new confidence factor
        # in each ``round``.
        try:
            self._update_confidence_factor()
            # Choose a random arm for each step.
            arm2 = self.random_state.choice(self.feedback_mechanism.get_arms())
            for arm1 in self.current_working_set:
                if arm1 != arm2:
                    first_won = self.feedback_mechanism.duel(
                        arm_i_index=arm1, arm_j_index=arm2
                    )
                else:
                    first_won = False
                self.preference_estimate.enter_sample(
                    first_arm_index=arm1, second_arm_index=arm2, first_won=first_won
                )
                # Update the borda score of ``arm1`` based on the duel outcome
                self._update_borda_score(
                    bernoulli_variable=1 if first_won else 0, arm_working_set=arm1
                )
            self._update_current_set()
            self.round += 1
        except AlgorithmFinishedException:
            pass

    def _update_confidence_factor(self) -> None:
        r"""Update Confidence factor (Corresponds to :math:`C_t` in :cite:`jamieson2015sparse`."""
        factor = (2 * self.feedback_mechanism.get_num_arms()) / self.round
        scaling_factor = (
            4 * self.feedback_mechanism.get_num_arms() ** 2 * self.round ** 2
        )
        additive_term = (2 * self.feedback_mechanism.get_num_arms()) / 3 * self.round
        self.confidence_factor = np.sqrt(
            factor * np.log(scaling_factor / self.failure_probability)
        ) + additive_term * np.log(scaling_factor / self.failure_probability)

    def _update_borda_score(
        self, bernoulli_variable: int, arm_working_set: int
    ) -> None:
        r"""Update the borda score for each sample.

        The Borda score of the arm in a round ``t`` is calculated as
        :math:`\hat s_{j, t}= \frac{t-1}{t} \hat s_{j, t-1} + \frac{n / (n - 1)}{t} Z_{j, I_{t}}`,
        where :math:`\hat s_{j, t}` is the Borda score of the arm ``j`` at round ``t`` and ``n`` is the size of the
        ``current_working_set``. The variable :math:`Z_{j, I_{t}}` is the independent Bernoulli variable of the duel
        between arm ``j`` and random arm which was chosen at round ``t``.

        Parameters
        ----------
        bernoulli_variable
            The independent Bernoulli variable of a duel whose expectation is equal to the pairwise probability.
            Corresponds to :math:`Z^{(t)}_{i,j}` in :cite:`jamieson2015sparse` , each denoting the outcome of
            "dueling" arms ``i`` and ``j`` at time ``t`` .
        arm_working_set
            The arm in the ``current_working_set`` which is being dueled with the random arm.
        """
        size = self.feedback_mechanism.get_num_arms()
        self.borda_scores_array[arm_working_set] = (
            (self.round - 1 / self.round) * self.borda_scores_array[arm_working_set]
        ) + ((size / ((size - 1) * self.round)) * bernoulli_variable)

    def _update_current_set(self) -> None:
        r"""Update the current working set by removing the arms.

        The removal of arms is based on two conditions with depends on whether the number of duels have crossed the
        ``time_gate``.
        """
        arms_to_be_removed = []
        for arm in self.current_working_set:
            if self._condition_1(arm_in_working_set=arm):
                arms_to_be_removed.append(arm)
            elif self._condition_2(arm_in_working_set=arm):
                arms_to_be_removed.append(arm)
        self.current_working_set = np.setdiff1d(
            self.current_working_set, arms_to_be_removed
        )

    def _condition_1(self, arm_in_working_set: int) -> bool:
        r"""Eliminate the arm by exploiting the sparsity condition in `Algorithm 1` in :cite:`jamieson2015sparse`.

        The condition only applies if the logical time step (Corresponding to :math:`t` in
        :cite:`jamieson2015sparse`) is greater than the time gate (Corresponding to :math:`T_0` in
        :cite:`jamieson2015sparse`). Once this condition is active, the function starts removing the arms with large
        partial Borda gaps hence exploiting the assumption that the top arms can be distinguished by comparisons with
        a sparse set of other arms.

        Parameters
        ----------
        arm_in_working_set
            The arm in the working set which can be eliminated if ``condition_1`` is true. Corresponds to arm ``j`` in :cite:`jamieson2015sparse`.

        Returns
        -------
        bool
            Whether the condition is ``True`` or ``False``.
        """
        if self.round > self.time_gate:
            try:
                sparsity_set = self.random_state.choice(
                    self.current_working_set, size=self.sparsity_level, replace=False
                )
            except ValueError:
                sparsity_set = self.random_state.choice(
                    self.current_working_set, size=self.sparsity_level, replace=True
                )
            for challenger_arm in self.current_working_set:
                # ``omega_dict`` contains the arms which yields the largest discrepancies of the estimated
                # probabilities.
                omega_dict = self._get_empirical_sum_of_discrepancy(
                    challenger_arm=challenger_arm,
                    arm_in_working_set=arm_in_working_set,
                    sparsity_set=sparsity_set,
                )
                # ``argmax_value_set`` selects the keys yielding the largest discrepancies of the estimated
                # probabilities.
                argmax_value_set = [
                    k for k, v in omega_dict.items() if v == max(omega_dict.values())
                ]
                # The threshold quantity in Algorithm 1 of paper suggests a constant of ``6``. However, in Section 5
                # of the paper, the authors have recommended to use a value of ``0.5`` instead of ``6`` because the analysis
                # that led to this constant is very loose.
                assert self.sparsity_level is not None
                if (
                    self._empirical_partial_gap(
                        challenger_arm=challenger_arm,
                        arm_in_working_set=arm_in_working_set,
                        argmax_set=argmax_value_set,
                    )
                    - 0.5 * (self.sparsity_level + 1) * self.confidence_factor
                    > 0
                ):
                    return True
        return False

    def _empirical_partial_gap(
        self, challenger_arm: int, arm_in_working_set: int, argmax_set: List[int]
    ) -> float:
        r"""Give the partial gap between the Borda scores of ``challenger_arm`` and ``arm_in_working_set``.

        Corresponds to :math:`\hat{\Delta}_{i,j,t}(\Omega)` in :cite:`jamieson2015sparse`. The partial gap is based
        on only the comparisons with the arms in the sparsity set with size :math:`k`.

        Parameters
        ----------
        challenger_arm
            Corresponds to arm ``i`` in :cite:`jamieson2015sparse`.
        arm_in_working_set
            The arm in the working set which can be eliminated. Corresponds to arm ``j`` in :cite:`jamieson2015sparse`.
        argmax_set
            The set of arms for with largest discrepancies.

        Returns
        -------
        float
            Returns the partial gap between the Borda scores of ``challenger_arm`` and ``arm_in_working_set``.
        """
        diff_mean_estimate_sum = 0.0
        for arm_in_argmax_set in argmax_set:
            if (
                challenger_arm != arm_in_working_set
                and arm_in_working_set != arm_in_argmax_set
                and challenger_arm != arm_in_argmax_set
            ):
                diff_mean_estimate_sum += self.preference_estimate.get_mean_estimate(
                    first_arm_index=challenger_arm, second_arm_index=arm_in_argmax_set
                ) - self.preference_estimate.get_mean_estimate(
                    first_arm_index=arm_in_working_set,
                    second_arm_index=arm_in_argmax_set,
                )
        return (
            2
            * (
                self.preference_estimate.get_mean_estimate(
                    first_arm_index=challenger_arm, second_arm_index=arm_in_working_set
                )
                - 0.5
            )
            + diff_mean_estimate_sum
        )

    def _get_empirical_sum_of_discrepancy(
        self, challenger_arm: int, arm_in_working_set: int, sparsity_set: np.ndarray
    ) -> dict:
        r"""Return the dictionary of empirical sums of discrepancies.

        Corresponds to :math:`\hat{\nabla}_{i,j}(\Omega)` in :cite:`jamieson2015sparse`. The ``key`` of the dictionary
        contains all the arms in the ``subset`` from the set :math:`N` and size ``sparsity_level``. The ``value`` of
        the dictionary contains all the empirical sum of the discrepancies for the arm (:math:`\lvert \hat{p}_{i,\omega}-\hat{p}_{j,\omega} \rvert`)
        in the corresponding ``key``, where :math:`\omega \in \Omega`.

        Parameters
        ----------
        challenger_arm
            Corresponds to arm ``i`` in :cite:`jamieson2015sparse`.
        arm_in_working_set
            The arm in the working set which can be eliminated. Corresponds to arm ``j`` in :cite:`jamieson2015sparse`.
        sparsity_set
            Corresponds to the set :math:`\Omega` in :cite:`jamieson2015sparse`.

        Returns
        -------
        dict
            A dictionary with arm as the ``key`` and sum of difference of the discrepancies as ``value``.
        """
        empirical_sum_dictionary: Dict = defaultdict(float)
        for arm_in_subset in sparsity_set:
            if (
                challenger_arm != arm_in_working_set
                and arm_in_working_set != arm_in_subset
                and challenger_arm != arm_in_subset
            ):
                empirical_sum_dictionary[arm_in_subset] += abs(
                    self.preference_estimate.get_mean_estimate(
                        first_arm_index=challenger_arm, second_arm_index=arm_in_subset
                    )
                    - self.preference_estimate.get_mean_estimate(
                        first_arm_index=arm_in_working_set,
                        second_arm_index=arm_in_subset,
                    )
                )
        return empirical_sum_dictionary

    def _condition_2(self, arm_in_working_set: int) -> bool:
        r"""Check if another arm has a higher Borda score with sufficient confidence.

        The ``time_gate`` (:math:`T_0`) guarantees that this elimination will follow till the logical time step is less
        than ``time_gate``.

        Parameters
        ----------
        arm_in_working_set
            The arm in the working set which can be eliminated.

        Returns
        -------
        bool
            Whether the condition is ``True`` or ``False``.
        """
        threshold = (
            (
                self.feedback_mechanism.get_num_arms()
                / self.feedback_mechanism.get_num_arms()
                - 1
            )
            * np.sqrt(
                (
                    2
                    * np.log(
                        4
                        * self.feedback_mechanism.get_num_arms()
                        * self.round ** 2
                        / self.failure_probability
                    )
                )
            )
            / self.round
        )
        for challenger_arm in self.current_working_set:
            if arm_in_working_set != challenger_arm:
                if (
                    self.borda_scores_array[challenger_arm]
                    - self.borda_scores_array[arm_in_working_set]
                    > threshold
                ):
                    return True
        return False

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        Returns
        -------
        bool
            Whether exploration is finished.
        """
        return len(self.current_working_set) <= 1

    def get_borda_winner(self) -> Optional[int]:
        """Return the computed :term:`Borda winner` if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.

        Returns
        -------
        int
            The arm with the highest Borda score.
        """
        if not self.exploration_finished():
            return None
        return self.current_working_set[0]
