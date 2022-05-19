"""An implementation of the Relative Minimum Empirical Divergence for Dueling Bandits."""
from itertools import combinations
from typing import Callable
from typing import Optional
from typing import Set

import numpy as np
from scipy.special import rel_entr

from duelnlg.duelpy.algorithms.interfaces import CondorcetProducer
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.util.utility_functions import argmin_set
from duelnlg.duelpy.util.utility_functions import kullback_leibler_divergence_plus


class RelativeMinimumEmpiricalDivergence1(CondorcetProducer):
    """Implementation of Relative Minimum Empirical Divergence 1 algorithm.

    The goal of this algorithm is to find Condorcet winner minimizing regret through the relative comparision of the
    arms.

    This algorithm assume there exist a condorcet winner. The preference matrix is estimated through series of relative
    comparision.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        How many comparisons the algorithm should do. This does not impact the
        decision of the algorithm, only after some inital time step of execution, it executes how many step the run
        executes.
    exploratory_constant
        Optional, The confidence radius grows proportional to the square root of this value. Corresponds to `alpha` in
        paper. The value of exploratory_constant must be greater than 0.5.Default value is 0.51
    random_state
        Optional, used for random choices in the algorithm.

    Attributes
    ----------
    preference_estimate
        Estimation of a preference matrix based on samples.
    feedback_mechanism
    exploratory_constant
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(41)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=random_state)
    >>> def ordinary_function(num_arms:int)-> np.float128: return 0.3 * np.power( \
            num_arms, 1.01   \
        )
    >>> test_object = RelativeMinimumEmpiricalDivergence1(feedback_mechanism, exploratory_constant=0.51, \
    ordinary_function=None,random_state=random_state, time_horizon=100)
    >>> test_object.run()
    >>> test_object.get_condorcet_winner()
    2
    >>> regret_history, cumul_regret = feedback_mechanism.calculate_average_regret(2)
    >>> np.round(cumul_regret, 2)
    17.0
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        ordinary_function: Optional[Callable[[int], np.float128]] = None,
        exploratory_constant: float = 0.51,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(
            feedback_mechanism=feedback_mechanism, time_horizon=time_horizon
        )
        if ordinary_function is None:

            def default_ordinary_function(num_arms: int) -> np.float128:
                return 0.3 * np.power(num_arms, 1.01)

            self.ordinary_function: Callable[
                [int], np.float128
            ] = default_ordinary_function
        else:
            self.ordinary_function = ordinary_function

        self.preference_estimate = PreferenceEstimate(
            num_arms=self.feedback_mechanism.get_num_arms()
        )
        self.time_step = 0
        self.exploratory_constant = exploratory_constant
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        # draw each pair of arms L times. L is denoted as self.loop
        self.loop = 1
        # arms in the current loop
        arms = list(self.feedback_mechanism.get_arms())
        # arms in the self.loop_current is arbitrarily fixed order.
        self.random_state.shuffle(arms)
        self.loop_current = set(arms)
        # arms remaining after a current loop
        self.loop_remaining = set(self.feedback_mechanism.get_arms())
        self.loop_ntilda: Set[int] = set()
        # combination of the pair of the arms
        self.unique_arm_pairs = self._initialize_unique_arms()
        # assume the best arm to be one out of list of the arms which has minimum empirical divergence
        self.best_arm = self.random_state.choice(
            list(self.feedback_mechanism.get_arms())
        )
        self.__initial_phase_comparison()
        self.best_arm = self._get_best_arm()
        self.fix_challenger = [0 for _ in self.feedback_mechanism.get_arms()]
        self._fix_challenger_for()

    def _initialize_unique_arms(self) -> list:
        """Generate an unique arm pairs and shuffle it as the arm pair should be in the arbitary fixed order."""
        arms = list(self.feedback_mechanism.get_arms())
        self.random_state.shuffle(arms)
        return list(combinations(arms, 2))

    def __initial_phase_comparison(self) -> None:
        """Run initial phase of the comparison between the arm pairs provided number of loops."""
        for arm_i, arm_j in self.unique_arm_pairs:
            comparison = 0
            while comparison != self.loop:
                self.time_step += 1
                self.preference_estimate.enter_sample(
                    arm_i, arm_j, self.feedback_mechanism.duel(arm_i, arm_j)
                )
                comparison += 1
                if self.time_step == self.time_horizon:
                    return

    def _choose_second_candidate(self, arm_i: int) -> int:
        """Select the challenger arm from the list of the arm for the reference arm.

        Parameters
        ----------
        arm_i
            The first candidate from the list of the arm.

        Returns
        -------
        int
            The second candidate(challenger arm) for the first candidate.
        """
        opponent_set = set()
        for arm_j in self.feedback_mechanism.get_arms():
            if self.preference_estimate.get_mean_estimate(arm_i, arm_j) > 0.5:
                opponent_set.add(arm_j)
        if self.best_arm in opponent_set or len(opponent_set) == 0:
            return self.best_arm

        arm_i_vs_arm_j_mean_estimates = [
            self.preference_estimate.get_mean_estimate(arm_i, arm_j)
            for arm_j in self.feedback_mechanism.get_arms()
        ]
        return self.random_state.choice(
            argmin_set(arm_i_vs_arm_j_mean_estimates, [arm_i])
        )

    def _get_best_arm(self) -> int:
        """Calculate the arm which has minimum empirical divergence.

        Returns
        -------
        int
            arm which with minimum empirical divergence.
        """
        empirical_divergence = np.zeros(self.feedback_mechanism.get_num_arms())
        for arm_i in self.feedback_mechanism.get_arms():
            empirical_divergence[arm_i] = self._empirical_divergence(arm_i)
        return self.random_state.choice(argmin_set(empirical_divergence))

    def _empirical_divergence(self, arm_i: int) -> float:
        """Calculate the empirical divergence of an arm.

        Parameters
        ----------
        arm_i
            The arm whose empirical divergence to be calculated.

        Returns
        -------
        float
           empirical divergence of an arm.
        """
        empirical_divergence_arm_i = 0
        for arm_j in self.feedback_mechanism.get_arms():
            kl_divergences = rel_entr(
                1 - self.preference_estimate.get_mean_estimate(arm_i, arm_j), 0.5,
            ) + rel_entr(self.preference_estimate.get_mean_estimate(arm_i, arm_j), 0.5,)

            empirical_divergence_arm_i += (
                self.preference_estimate.get_num_samples(arm_i, arm_j) * kl_divergences
            )
        return empirical_divergence_arm_i

    def _tilda_j(self, arm_i: int) -> bool:
        """Check whether the arm is the candidate for the Condorcet winner.

        Parameters
        ----------
        arm_i
            An arm.

        Returns
        -------
        bool
           True if the arm is candidate for the Condorcet winner.
        """
        return (
            self._empirical_divergence(arm_i)
            - self._empirical_divergence(self.best_arm)
        ) <= (
            np.log(self.time_step)
            + self.ordinary_function(self.feedback_mechanism.get_num_arms())
        )

    def _each_phase_initial_comparison(self) -> None:
        pass

    def _fix_challenger_for(self) -> None:
        pass

    def get_winner(self):
        return list(
            self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )[0]

    def get_condorcet_winner(self) -> Optional[int]:
        """Compute Condorcet winner as per the estimated preference matrix.

        Returns
        -------
        Optional[int]
            Condorcet winner from preference estimate.
        """
        if self.is_finished():
            return (
                self.preference_estimate.get_mean_estimate_matrix().get_condorcet_winner()
            )

        return None

    def is_finished(self) -> bool:
        """Check the algorithm is finished when the algorithm is executed greater than equal to the time horizon."""
        if self.time_horizon is None:
            raise NotImplementedError(
                "No time horizon set and no custom termination condition implemented."
            )
        return self.time_step >= self.time_horizon

    def step(self) -> None:
        """Each step of the algorithm after the some initial drawn pairs of arm for the competition."""
        self._each_phase_initial_comparison()
        if self.time_step == self.time_horizon:
            return
        for arm_i in self.loop_current:
            self.time_step += 1
            self.best_arm = self._get_best_arm()
            arm_j = self._choose_second_candidate(arm_i)
            self.preference_estimate.enter_sample(
                arm_i, arm_j, self.feedback_mechanism.duel(arm_i, arm_j)
            )
            self.loop_remaining = self.loop_remaining - {arm_i}
            remaining_arms = (
                set(self.feedback_mechanism.get_arms())
                - self.loop_remaining
                - self.loop_ntilda
            )
            for remaining_arm in remaining_arms:
                if self._tilda_j(remaining_arm):
                    self.loop_ntilda = {remaining_arm}.union(self.loop_ntilda)

            if self.time_step == self.time_horizon:
                return
        self.loop_current = self.loop_ntilda
        self.loop_remaining = self.loop_ntilda
        self.loop_ntilda = set()


class RelativeMinimumEmpiricalDivergence2(RelativeMinimumEmpiricalDivergence1):
    """Implementation of Relative Minimum Empirical Divergence 2 algorithm.

    The goal of this algorithm is to find Condorcet winner minimizing regret through the relative comparision of the
    arms.

    This algorithm assume there exist a condorcet winner. The preference matrix is estimated through series of relative
    comparision.


    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        How many comparisons the algorithm should do. This does not impact the
        decision of the algorithm, only after some inital time step of execution, it executes how many step the run
        executes.
    exploratory_constant
        Optional, The confidence radius grows proportional to the square root of this value. Corresponds to `alpha` in
        paper. The value of exploratory_constant must be greater than 0.5.Default value is 0.51
    random_state
        Optional, used for random choices in the algorithm.


    Attributes
    ----------
    preference_estimate
        Estimation of a preference matrix based on samples.
    feedback_mechanism
    exploratory_constant
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(41)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=random_state)
    >>> def ordinary_function(num_arms:int)-> np.float128: return 0.3 * np.power( \
            num_arms, 1.01   \
        )
    >>> test_object = RelativeMinimumEmpiricalDivergence2(feedback_mechanism, exploratory_constant=0.51, \
    ordinary_function=ordinary_function,random_state=random_state, time_horizon=100)
    >>> test_object.run()
    >>> test_object.get_condorcet_winner()
    2
    >>> regret_history, cumul_regret = feedback_mechanism.calculate_average_regret(2)
    >>> np.round(cumul_regret, 2)
    18.4
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        ordinary_function: Optional[Callable[[int], np.float128]] = None,
        random_state: Optional[np.random.RandomState] = None,
        exploratory_constant: float = 0.51,
    ):
        super().__init__(
            ordinary_function=ordinary_function,
            feedback_mechanism=feedback_mechanism,
            time_horizon=time_horizon,
            exploratory_constant=exploratory_constant,
            random_state=random_state,
        )
        self.loop = np.ceil(exploratory_constant * np.log(np.log(self.time_horizon)))

    def _each_phase_initial_comparison(self) -> None:
        """Draw the arm pairs and compare them.

        The pair of arms are drawn until the number of sample is less or equal to the product of  exploratory
        constant and loglog(self.time).
        """
        for arm_i, arm_j in self.unique_arm_pairs:
            if self.time_step == self.time_horizon:
                break
            while self.preference_estimate.get_num_samples(
                arm_i, arm_j
            ) <= self.exploratory_constant * np.log(np.log(self.time_step)):
                self.time_step += 1
                self.preference_estimate.enter_sample(
                    arm_i, arm_j, self.feedback_mechanism.duel(arm_i, arm_j)
                )
                if self.time_step == self.time_horizon:
                    break

    def _choose_second_candidate(self, arm_i: int) -> int:
        """Select the challenger arm from the list of the arm for the reference arm.

        Parameters
        ----------
        arm_i
            The first candidate from the list of the arm.

        Returns
        -------
        int
            The second candidate(challenger arm) for the first candidate.
        """
        opponent_set = set()
        for arm_j in self.feedback_mechanism.get_arms():
            if self.preference_estimate.get_mean_estimate(arm_i, arm_j) > 0.5:
                opponent_set.add(arm_j)
        challenger_arm = self.get_challenger_arm(arm_i)
        if challenger_arm in opponent_set and self.preference_estimate.get_mean_estimate(
            arm_i, self.best_arm
        ) >= (
            self.preference_estimate.get_mean_estimate(arm_i, challenger_arm)
            / np.log(np.log(self.time_step))
        ):
            return challenger_arm

        return RelativeMinimumEmpiricalDivergence1._choose_second_candidate(self, arm_i)

    def get_challenger_arm(self, arm_i: int) -> int:
        """Compute the challenger arm for reference arm.

        Parameters
        ----------
        arm_i
            The reference arm.

        Returns
        -------
        int
           The challenger arm for the reference arm.
        """
        empirical_plus_divergence = [
            float(0) for _ in self.feedback_mechanism.get_arms()
        ]
        for arm_j in self.feedback_mechanism.get_arms():
            empirical_plus_divergence[arm_j] = self.empirical_divergence_plus(
                arm_i, arm_j
            )
        return self.random_state.choice(argmin_set(empirical_plus_divergence, [arm_i]))

    def empirical_divergence_plus(self, arm_i: int, arm_j: int) -> float:
        """Calculate the empirical divergence plus of an arm with reference arm.

        Parameters
        ----------
        arm_i
            The reference arm.

        arm_j
            The arm whose empirical divergence is to be calculated.

        Returns
        -------
        float
           empirical divergence plus of an arm.
        """
        divergence_plus = kullback_leibler_divergence_plus(
            self.preference_estimate.get_mean_estimate(self.best_arm, arm_j), 0.5
        )
        if divergence_plus == float("inf"):
            return 0
        elif divergence_plus == 0:
            return 1e308
        else:
            return (
                1
                - self.preference_estimate.get_mean_estimate(self.best_arm, arm_i)
                - self.preference_estimate.get_mean_estimate(self.best_arm, arm_j)
            ) / divergence_plus

    def _fix_challenger_for(self) -> None:
        pass


class RelativeMinimumEmpiricalDivergence2FH(RelativeMinimumEmpiricalDivergence2):
    """Implementation of Relative Minimum Empirical Divergence 2FH algorithm.

    The goal of this algorithm is to find Condorcet winner minimizing regret through the relative comparision of the
    arms.

    This algorithm assume there exist a condorcet winner. The preference matrix is estimated through series of relative
    comparision.


    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        How many comparisons the algorithm should do. This does not impact the
        decision of the algorithm, only after some inital time step of execution, it executes how many step the run
        executes.
    exploratory_constant
        Optional, The confidence radius grows proportional to the square root of this value. Corresponds to `alpha` in
        paper. The value of exploratory_constant must be greater than 0.5.Default value is 0.51
    random_state
        Optional, used for random choices in the algorithm.

    Attributes
    ----------
    preference_estimate
        Estimation of a preference matrix based on samples.
    feedback_mechanism
    exploratory_constant
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelnlg.duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(41)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=random_state)
    >>> def ordinary_function(num_arms:int)-> np.float128: return 0.3 * np.power( \
            num_arms, 1.01   \
        )
    >>> test_object = RelativeMinimumEmpiricalDivergence2FH(feedback_mechanism, exploratory_constant=0.51, \
    ordinary_function=ordinary_function,random_state=random_state, time_horizon=100)
    >>> test_object.run()
    >>> test_object.get_condorcet_winner()
    2
    >>> regret_history, cumul_regret = feedback_mechanism.calculate_average_regret(2)
    >>> np.round(cumul_regret, 2)
    15.7
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        ordinary_function: Optional[Callable[[int], np.float128]] = None,
        random_state: Optional[np.random.RandomState] = None,
        exploratory_constant: float = 0.51,
    ):
        super().__init__(
            ordinary_function=ordinary_function,
            feedback_mechanism=feedback_mechanism,
            time_horizon=time_horizon,
            random_state=random_state,
            exploratory_constant=exploratory_constant,
        )
        self.loop = np.ceil(exploratory_constant * np.log(np.log(self.time_horizon)))

    def _each_phase_initial_comparison(self) -> None:
        pass

    def _fix_challenger_for(self) -> None:
        """Compute the challenger arm for reference arm."""
        for arm_i in self.feedback_mechanism.get_arms():
            self.fix_challenger[arm_i] = self.get_challenger_arm(arm_i)

    def _choose_second_candidate(self, arm_i: int) -> int:
        """Select the challenger arm from the list of the arm for the reference arm.

        Parameters
        ----------
        arm_i
            The first candidate from the list of the arm.

        Returns
        -------
        int
            The second candidate(challenger arm) for the first candidate.
        """
        opponent_set = set()
        for arm_j in self.feedback_mechanism.get_arms():
            if self.preference_estimate.get_mean_estimate(arm_i, arm_j) > 0.5:
                opponent_set.add(arm_j)
        challenger = self.fix_challenger[arm_i]
        if challenger in opponent_set and self.preference_estimate.get_mean_estimate(
            arm_i, self.best_arm
        ) >= self.preference_estimate.get_mean_estimate(arm_i, challenger) / np.log(
            np.log(self.time_horizon)
        ):
            return self.fix_challenger[arm_i]
        # pylint: disable=protected-access
        return RelativeMinimumEmpiricalDivergence1._choose_second_candidate(self, arm_i)
