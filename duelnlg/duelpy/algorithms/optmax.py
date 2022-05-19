"""An implementation of the OptMax algorithm."""

from typing import Optional

import numpy as np
from numpy.random import RandomState

from duelnlg.duelpy.algorithms.interfaces import PacAlgorithm
from duelnlg.duelpy.algorithms.interfaces import SingleCopelandProducer
from duelnlg.duelpy.algorithms.sequential_elimination import SequentialElimination
from duelnlg.duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.stats.preference_estimate import PreferenceEstimate
from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException
import duelnlg.duelpy.util.utility_functions as utility


class OptMax(SingleCopelandProducer, PacAlgorithm):
    r"""Implement the OptMax algorithm.

    The goal of this algorithm is to find an :math:`\epsilon`-maximum arm among the given arms.

    It assumes :term:`moderate stochastic transitivity`.

    As per the theorem 8 in the paper :cite:`falahatgar2018limits`, the OptMax algorithm takes
    :math:`\mathcal{O}\left(\frac{N}{\epsilon^2} \log\left(\frac{1}{\delta}\right)\right)` comparisons to find an
    :math:`\epsilon`-maximum arm. :math:`N` is the number of arms.

    This algorithm computes the :math:`\epsilon`-maximum arm by choosing one of the methods among
    ``_pick_anchor_for_lower_range`` or ``_pick_anchor_for_medium_range`` or ``_pick_anchor_for_higher_range``.
    These methods are selected based on the ``failure_probability`` :math:`\delta`, which is calculated using the number of
    arms :math:`N` given to the algorithm. Here :math:`\epsilon = \epsilon_u - \epsilon_l` and
    :math:`\epsilon_u` and :math:`\epsilon_l` are upper and lower bias respectively.

    The algorithm as presented in :cite:`falahatgar2018limits` either finds an arm which is either a
    :math:`\frac{2\epsilon}{3}`-maximum arm or uses :class:`Sequential Elimination<duelpy.algorithms.sequential_elimination.SequentialElimination>` to find an arm which is
    :math:`\epsilon`-maximum against the other arms.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    failure_probability
        Refer to :math:`\delta` in :cite:`falahatgar2018limits`.
    time_horizon
        Sets a limit to the number of comparisons that the algorithm will make.
    epsilon_range
        Refer to :math:`\epsilon` in :cite:`falahatgar2018limits`
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.

    Raises
    ------
    ValueError
        Raised when the wrong set of arms are given to the algorithm.

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
    >>> preferred_arms=[1,2]
    >>> random_state_user = RandomState()
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state_user)
    >>> opt_max = OptMax(feedback_mechanism=feedback_mechanism, failure_probability=0.1, random_state=random_state_user)
    >>> opt_max.run()
    >>> best_arm = opt_max.get_copeland_winner()
    >>> best_arm in preferred_arms
    True
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        failure_probability: float = 0.1,
        time_horizon: Optional[int] = None,
        epsilon_range: float = 0.05,
        random_state: RandomState = RandomState(),
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self._failure_probability = failure_probability
        self._epsilon_range = epsilon_range
        self._anchor_arm: Optional[int] = None
        self.preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms()
        )
        self._random_state = random_state

    def exploration_finished(self) -> bool:
        """Determine whether algorithm has completed exploration.

        Returns
        -------
        bool
            Exploration finished or not.
        """
        return self._anchor_arm is not None

    def explore(self) -> None:
        """Run one step of exploration.

        Exploration is divided into 3 parts. For more details, refer to `Algorithm 4` of :cite:`falahatgar2018limits`.
        """
        try:
            if self._failure_probability <= 1 / np.power(
                len(self.feedback_mechanism.get_arms()), 1 / 3
            ):
                self._anchor_arm = self._pick_anchor_for_low_range()
            elif self._failure_probability <= 1 / np.log(
                len(self.feedback_mechanism.get_arms())
            ):
                self._anchor_arm = self._pick_anchor_for_medium_range()
            else:
                self._anchor_arm = self._pick_anchor_for_high_range()
        except AlgorithmFinishedException:
            return

    def get_copeland_winner(self) -> Optional[int]:
        """Return the arm chosen by the algorithm as Copeland winner.

        Returns
        -------
        None
            If the algorithm has not concluded.
        int
            If the algorithm has found the winner.
        """
        return self._anchor_arm

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

    def _pick_anchor_for_low_range(self) -> Optional[int]:
        r"""Optimal for lower range of failure probability  (:math:`\delta < N ^{-\frac{1}{3}}`).

        From a random subset of size :math:`{N}^{3/4}` where :math:`N` is the number of the arms given to the
        algorithm, this algorithm finds an :math:`\epsilon`-maximum of the set using a :math:`\frac{\epsilon}{2}`-maximum arm
        which is found by using sequential elimination algorithm.
        For more details, refer to `Algorithm 3` of :cite:`falahatgar2018limits`

        Raises
        ------
        AlgorithmFinishedException
            When number of duels equals time_horizon

        Returns
        -------
        int
            Returns a Copeland winner.
        """
        size = np.power(self.feedback_mechanism.get_num_arms(), 3 / 4)
        # form a set of random elements without replacement from arms.
        random_set = utility.pop_random(
            input_list=self.feedback_mechanism.get_arms().copy(),
            amount=int(size),
            random_state=self._random_state,
        )

        picked_anchor = self._pick_anchor(
            arms=random_set,
            epsilon_range=self._epsilon_range / 2,
            failure_probability=self._failure_probability / 3,
        )
        if picked_anchor is None:
            # this implies that time_horizon has been passed.
            raise AlgorithmFinishedException

        if self.time_horizon is not None:
            _time_horizon: Optional[int] = (
                self.time_horizon - self.feedback_mechanism.get_num_duels()
            )
        else:
            _time_horizon = None
        # sequential elimination runs on all the arms.
        seq_elim = SequentialElimination(
            feedback_mechanism=self.feedback_mechanism,
            epsilon_lower=self._epsilon_range / 2,
            epsilon_upper=self._epsilon_range,
            failure_probability=self._failure_probability / 3,
            anchor_arm=picked_anchor,
            time_horizon=_time_horizon,
        )
        while not seq_elim.exploration_finished() and not seq_elim.is_finished():
            seq_elim.explore()
        return seq_elim.get_copeland_winner()

    def _pick_anchor_for_medium_range(self) -> Optional[int]:
        r"""Optimal for medium range of failure probability (:math:`\delta < \frac{1}{\log(N)}`).

        From a random subset of size as mentioned in `Algorithm 11` of :cite:`falahatgar2018limits` , a pruned subset is
        formed. Performing sequential elimination on this small sized subset, will yield the copeland winner in
        :math:`\mathcal{O}\left(\frac{N}{\epsilon^2} \log\left(\frac{1}{\delta}\right)\right)` complexity.

        Refer to `Algorithm 1` of :cite:`falahatgar2018limits` for more details.

        Raises
        ------
        AlgorithmFinishedException
            When number of duels equals ``time_horizon``

        Returns
        -------
        int
            Returns a Copeland winner.
        """
        size_random_set = int(
            self.feedback_mechanism.get_num_arms()
            / np.square(np.log(self.feedback_mechanism.get_num_arms()))
        )

        # form a set of random elements without replacement from arms.
        random_set = utility.pop_random(
            amount=size_random_set,
            input_list=self.feedback_mechanism.get_arms(),
            random_state=self._random_state,
        )

        picked_anchor = self._pick_anchor(
            arms=random_set,
            epsilon_range=self._epsilon_range / 3,
            failure_probability=self._failure_probability / 4,
        )

        selected_anchor_element: int = picked_anchor
        # Refer to Step 5 in Algorithm 11 in paper :cite:`falahatgar2018limits`.
        failure_probability_factor = (
            4
            * np.square(np.log(len(self.feedback_mechanism.get_arms())))
            * np.log(
                4 * len(self.feedback_mechanism.get_arms()) / self._failure_probability
            )
        )
        pruned_list = self._prune(
            arms=random_set,
            selected_arm=selected_anchor_element,
            epsilon_lower=self._epsilon_range / 3,
            epsilon_upper=2 * self._epsilon_range / 3,
            failure_probability=self._failure_probability / failure_probability_factor,
        )

        if self.time_horizon is not None:
            _time_horizon: Optional[int] = (
                self.time_horizon - self.feedback_mechanism.get_num_duels()
            )
        else:
            _time_horizon = None

        # sequential elimination runs on pruned_list of arms.
        seq_elim = SequentialElimination(
            feedback_mechanism=self.feedback_mechanism,
            epsilon_lower=self._epsilon_range / 3,
            epsilon_upper=self._epsilon_range,
            failure_probability=self._failure_probability / 4,
            arms_subset=pruned_list,
            anchor_arm=selected_anchor_element,
            time_horizon=_time_horizon,
        )
        while not seq_elim.exploration_finished() and not seq_elim.is_finished():
            seq_elim.explore()
        return seq_elim.get_copeland_winner()

    def _pick_anchor_for_high_range(self) -> Optional[int]:
        r"""Optimal for higher range of failure probability.

        This algorithm is run for m number of stages (refer to `Algorithm 12` of :cite:`falahatgar2018limits`). In the
        first stage, the size of pruned set is small. In every stage, this subset grows. This step reduces the epsilon and
        confidence errors. The multiple stages are needed, to ensure that final pruning will yield only those arms
        survive which can actually beat the selected anchor arm. In every stage, sequential elimination will choose an
        anchor element which is around :math:`\frac{\epsilon}{3}`-maximum arm to other arms. This anchor element is used for the
        next stage.
        Refer to `Algorithm 12` in :cite:`falahatgar2018limits` for more details.

        Raises
        ------
        AlgorithmFinishedException
            When number of duels equals time_horizon

        Returns
        -------
        int
            Returns a Copeland winner.
        """
        set_size = np.divide(
            len(self.feedback_mechanism.get_arms()),
            np.square(np.log(len(self.feedback_mechanism.get_arms()))),
        )

        selected_copeland_arm = None

        # form a set of random elements without replacement from arms as mentioned on line 3.
        random_set = utility.pop_random(
            amount=int(set_size),
            input_list=self.feedback_mechanism.get_arms().copy(),
            random_state=self._random_state,
        )

        picked_anchor = self._pick_anchor(
            arms=random_set,
            epsilon_range=self._epsilon_range / 3,
            failure_probability=self._failure_probability / 4,
        )

        # total_stages refer to ``m`` as mentioned on line 5.
        total_stages = int(
            2
            * np.log(
                np.log(len(self.feedback_mechanism.get_arms()))
                / np.log(1 / self._failure_probability)
            )
        )

        # current_stage_anchor_arm refer to :math:`a_{i+1}` as mentioned on line 4.
        current_stage_anchor_arm: int = picked_anchor
        for current_stage in range(1, total_stages + 1):
            # epsilon_current_stage_upper refer to :math:`{\epsilon_i}^\'` on line 8.
            epsilon_current_stage_upper = self._epsilon_range / 3 + (
                2 * self._epsilon_range / 3
            ) / np.power(2, (total_stages - current_stage) / 3)
            # epsilon_current_stage_lower refer to :math:`{\epsilon_i}^\'\'` on line 9.
            epsilon_current_stage_lower = self._epsilon_range / 3 + (
                2 * self._epsilon_range / 3
            ) / np.power(2, (total_stages - current_stage + 1) / 3)
            # failure_probability_current_stage refer to :math: `{\delta_i} ^ \'`  as mentioned on line 10.
            failure_probability_current_stage = np.power(
                self._failure_probability, (total_stages - current_stage + 4)
            )
            # Corresponds to Step 11.
            random_set_current_stage = utility.pop_random(
                amount=int(
                    np.maximum(
                        set_size,
                        len(self.feedback_mechanism.get_arms())
                        * np.power(
                            self._failure_probability, total_stages - current_stage
                        ),
                    )
                ),
                input_list=self.feedback_mechanism.get_arms(),
                random_state=self._random_state,
            )
            # pruned_list_current_stage refer to :math:`{Q_i}^'' on line 12.
            pruned_list_current_stage = self._prune(
                arms=random_set_current_stage.copy(),
                selected_arm=current_stage_anchor_arm,
                epsilon_lower=epsilon_current_stage_lower,
                epsilon_upper=(
                    epsilon_current_stage_lower + epsilon_current_stage_upper
                )
                / 2,
                failure_probability=np.power(failure_probability_current_stage, 5) / 3,
            )
            if self.time_horizon is not None:
                _time_horizon: Optional[int] = (
                    self.time_horizon - self.feedback_mechanism.get_num_duels()
                )
            else:
                _time_horizon = None

            # sequential elimination runs on current_stage pruned_list.
            seq_elim = SequentialElimination(
                feedback_mechanism=self.feedback_mechanism,
                epsilon_lower=epsilon_current_stage_lower,
                epsilon_upper=epsilon_current_stage_upper,
                failure_probability=failure_probability_current_stage / 3,
                arms_subset=pruned_list_current_stage,
                anchor_arm=current_stage_anchor_arm,
                time_horizon=_time_horizon,
            )
            while not seq_elim.exploration_finished() and not seq_elim.is_finished():
                seq_elim.explore()
            selected_copeland_arm = seq_elim.get_copeland_winner()

        # Refer to line 15.
        return selected_copeland_arm if not None else None

    def _arm1_beats_arm2(
        self,
        arm1: int,
        arm2: int,
        epsilon_upper: float,
        epsilon_lower: float,
        failure_probability: float,
    ) -> bool:
        r"""Determine if competing arm beats anchor arm.

        The calibrated preference probability estimate (:math:`\hat{p}_{i,j}`) for competing arm against anchor arm
        refers to the probability that competing arm is preferred to anchor arm and is calculated based on the number of
        times competing arm has won divided by total number of duels between both arms. This value is updated after each
         duel between them.

        The confidence radius (:math:`\hat{c}`) is calculated such that with proof :math:`\ge 1-\delta`,
        :math:`\lvert \hat{p}_{i,j} - p_{i,j} \rvert < \hat{c}` after any number of comparisons. Here :math:`1-\delta`
        as mentioned in the paper :cite:`falahatgar2017maxing`, is called confidence value but we have referred it
        as the failure probability.

        The method returns True if :math:`\hat{p}_{i,j} \ge (\epsilon_u + \epsilon_l)/2` otherwise False is
        returned.
        For more details, please refer to appendix section `Algorithm 9` in :cite:`falahatgar2018limits`.

        Parameters
        ----------
        arm1
            Corresponds to :math:`i` in `Algorithm 9` in the paper :cite:`falahatgar2018limits`.
        arm2
            Corresponds to :math:`j` in `Algorithm 9` in the paper :cite:`falahatgar2018limits`.
        epsilon_lower
            Corresponds to :math:`\epsilon_l` in the paper :cite:`falahatgar2018limits`.
        epsilon_upper
            Corresponds to :math:`\epsilon_u` in the paper :cite:`falahatgar2018limits`.
        failure_probability
            Corresponds to :math:`\delta` in the paper :cite:`falahatgar2018limits`.

        Returns
        -------
        bool
            True if arm 1 beats arm2.
        """
        epsilon_range = (
            epsilon_upper - epsilon_lower
        )  # refer to :math:`\epsilon` in paper :cite:`falahatgar2018limits`.
        epsilon_mean = (epsilon_upper + epsilon_lower) / 2
        confidence_radius = 0.5
        current_iteration_count = (
            0  # refer to variable 't' in paper :cite:`falahatgar2018limits`
        )
        calibrated_preference_estimate = 0.0

        # number of rounds is selected in such a way that compare method selects the winner with :math:`1-\delta`
        # confidence. See Algorithm 9 of paper :cite:`falahatgar2018limits`.
        rounds_for_iteration = int(
            2 / (np.power(epsilon_range, 2)) * np.log(2 / failure_probability)
        )

        def prob_scaling(num_iteration: int) -> float:
            return np.square(2 * num_iteration)

        confidence_radius_fn = HoeffdingConfidenceRadius(
            failure_probability=failure_probability,
            probability_scaling_factor=prob_scaling,
        )

        # compare two arms multiple times to get an estimate of their winnings. Refer to Algorithm 9 of paper
        # :cite:`falahatgar2018limits`.
        while (
            current_iteration_count < rounds_for_iteration
            and np.absolute(calibrated_preference_estimate - epsilon_mean)
            <= confidence_radius
        ):
            current_iteration_count += 1
            feedback_result = self.feedback_mechanism.duel(arm1, arm2)

            self.preference_estimate.enter_sample(arm1, arm2, feedback_result)

            calibrated_preference_estimate = (
                self.preference_estimate.get_mean_estimate(arm1, arm2) - 0.5
            )
            confidence_radius = confidence_radius_fn(
                self.preference_estimate.get_num_samples(arm1, arm2)
            )

        return calibrated_preference_estimate >= epsilon_mean

    def _pick_anchor(
        self,
        failure_probability: float,
        arms: list,
        epsilon_range: float,
        random_state: RandomState = RandomState(),
    ) -> int:
        r"""Return the :math:`\epsilon`-maximum arm.

        From the random subset of the input list, an :math:`\frac{\epsilon}{2}`-maximum arm is chosen by applying :class:`Sequential Elimination<duelpy.algorithms.sequential_elimination.SequentialElimination>`
        on the random subset. The anchor element returned by the above sequential elimination is again provided to sequential
        elimination and this time, an :math:`\epsilon` maximum arm is found out. Refer to Lemma 5 and its proof in Section
        4.2 of paper :cite:`falahatgar2018limits`.

        Parameters
        ----------
        failure_probability
            Corresponds to :math:`\delta` in :cite:`falahatgar2018limits`.
        arms
            List of arms that has to be pruned.
        epsilon_range
            Corresponds to :math:`\epsilon` in :cite:`falahatgar2018limits`.
        random_state
            A numpy random state. Defaults to an unseeded state when not specified.

        Raises
        ------
        AlgorithmFinishedException
            When number of duels equals time_horizon

        Returns
        -------
        int
            Returns :math:`\epsilon`-maximum anchor arm.
        """

        def prob_scaling(num_iteration: int) -> float:
            return 4 * num_iteration

        confidence_radius_fn = HoeffdingConfidenceRadius(
            failure_probability=failure_probability,
            probability_scaling_factor=prob_scaling,
            factor=2 * np.square(len(arms)),
        )

        # form a random subset (Q) size = num_arms * log(4*num_arms^2/failure_prob) as mentioned in Algorithm 2 of
        # :cite:`falahatgar2018limits`.
        pruned_arms = utility.pop_random(
            amount=int(confidence_radius_fn(len(arms))),
            input_list=arms.copy(),
            random_state=random_state,
        )

        # select a random anchor element from pruned arms and remove it from pruned arms.
        random_anchor_arm = utility.pop_random(pruned_arms, random_state=random_state)[
            0
        ]

        if self.time_horizon is not None:
            _time_horizon: Optional[int] = (
                self.time_horizon - self.feedback_mechanism.get_num_duels()
            )
        else:
            _time_horizon = None
        # sequential elimination runs on pruned arms with the selected anchor arm.
        seq_elim = SequentialElimination(
            feedback_mechanism=self.feedback_mechanism,
            failure_probability=failure_probability / 4,
            anchor_arm=random_anchor_arm,
            epsilon_upper=epsilon_range / 2,
            arms_subset=pruned_arms,
            time_horizon=_time_horizon,
        )
        while not seq_elim.exploration_finished() and not seq_elim.is_finished():
            seq_elim.explore()

        candidate_anchor_arm = seq_elim.get_copeland_winner()

        if self.time_horizon is not None:
            _time_horizon = self.time_horizon - self.feedback_mechanism.get_num_duels()
        else:
            _time_horizon = None
        # sequential elimination run on arms provided to _pick_anchor method.
        seq_elim = SequentialElimination(
            feedback_mechanism=self.feedback_mechanism,
            failure_probability=failure_probability / 2,
            anchor_arm=candidate_anchor_arm,
            epsilon_lower=epsilon_range / 2,
            epsilon_upper=epsilon_range,
            arms_subset=arms,
            time_horizon=_time_horizon,
        )
        while not seq_elim.exploration_finished() and not seq_elim.is_finished():
            seq_elim.explore()

        _copeland_winner = seq_elim.get_copeland_winner()
        if _copeland_winner is None:
            raise AlgorithmFinishedException
        assert _copeland_winner is not None
        copeland_winner: int = _copeland_winner
        return copeland_winner

    def _prune(
        self,
        arms: list,
        selected_arm: int,
        epsilon_upper: float,
        epsilon_lower: float,
        failure_probability: float,
    ) -> list:
        r"""Return a list of arms with elements that are epsilon preferable to other arms only.

        Starting with the given list and an arm provided, compare the other elements of list with the given arm and one by one
        remove the elements for which the calibrated preference probability estimate (:math:`\hat{p}_{i,j}`) of anchor
        element (i) against other element (j) is less than lower epsilon (:math:`\epsilon_l`), thus reducing the size of input
        list and ensuring only good elements for which :math:`\hat{p}_{i,j} \ge \epsilon_u`. The goal of prune is to return a
        list of size as mentioned in C.8.1 in the paper :cite:`falahatgar2018limits`.

        Parameters
        ----------
        arms
            List of arms that has to be pruned.
        selected_arm
            An arm which is competed against other arms.
        epsilon_upper
            Corresponds to :math:`\epsilon_u` in the paper :cite:`falahatgar2018limits`.
        epsilon_lower
            Corresponds to :math:`\epsilon_l` in the paper :cite:`falahatgar2018limits`.
        failure_probability
            Corresponds to :math:`\delta` in the paper :cite:`falahatgar2018limits`.

        Returns
        -------
        list
            Reduced list which contains elements for which :math:`\hat{p}_{i,j} \ge \frac{\epsilon_u + \epsilon_l}{2}`
        """
        current_round = 1
        remaining_arms = arms.copy()

        target_arms_size = (
            4
            * (np.log(2 * len(arms) / failure_probability + 1e-7))
            / failure_probability
        )
        arms_size = len(arms)

        # Refer to step 5 in Algorithm 10 in paper :cite:`falahatgar2018limits`.
        while arms_size > target_arms_size and current_round < np.square(
            np.log(arms_size + 1e-7)
        ):
            arms_eliminated = list()
            for arm in remaining_arms:
                if not (
                    self._arm1_beats_arm2(
                        arm1=arm,
                        arm2=selected_arm,
                        epsilon_lower=epsilon_lower,
                        epsilon_upper=epsilon_upper,
                        failure_probability=failure_probability
                        / np.power(2, current_round + 1),
                    )
                ):
                    arms_eliminated.append(arm)  # add bad element into list.
            remaining_arms = list(set(remaining_arms) - set(arms_eliminated))
            current_round += 1

        return remaining_arms
