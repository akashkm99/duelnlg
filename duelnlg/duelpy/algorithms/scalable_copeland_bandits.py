"""An implementation of the Scalable Copeland Bandits algorithm."""
from typing import Optional

import numpy as np

from duelnlg.duelpy.algorithms.algorithm import Algorithm
from duelnlg.duelpy.algorithms.kl_divergence_based_pac import KLDivergenceBasedPAC
from duelnlg.duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.util.exceptions import AlgorithmFinishedException
from duelnlg.duelpy.util.feedback_decorators import BudgetedFeedbackMechanism


class ScalableCopelandBandits(Algorithm):
    r"""An implementation of the Scalable Copeland Bandits (SCB) algorithm.

    The goal of the algorithm is to minimize the :term:`Copeland regret`.

    It is assumed that there are no ties between arms, i.e. the probability
    of any arm winning against another is never :math:`\frac{1}{2}`.

    The bound on the expected regret is given as
    :math:`\mathcal{O}\left(\frac{N(L_C+\ln(N)) \ln(T)}{\Delta_\min)^2}\right)`. :math:`N` is the
    number of arms, :math:`T` is the time horizon. :math:`C` is the number
    of :term:`Copeland winners<Copeland winner>` and :math:`L_C` is the number of arms against which
    a :term:`Copeland winner` will lose in expectation. For any :term:`Copeland winner` and
    any non-:term:`Copeland winner`, :math:`\Delta` is the smallest absolute distance
    to :math:`\frac{1}{2}` in the probability of these arms dueling against
    each other. :math:`\Delta_\min` is the smallest :math:`\Delta` value for
    the given set of arms such that :math:`\Delta_\min \ne 0`.  Note that the paper uses a different definition of :term:`Copeland regret`, in this library the value is half of that in the paper.

    The Scalable Copeland Bandits algorithm is based on :cite:`zoghi2015copeland`.
    SCB performs well for a large number of arms (:math:`500` or more); refer to
    :cite:`zoghi2015copeland` for more details.
    It proceeds by conducting duels which are most informative about the precedence
    of the participating arms in terms of their :term:`Copeland scores<Copeland score>`. The confirmed
    non-:term:`Copeland winners<Copeland winner>` are eliminated based on the results of the prior duels.
    After conducting sufficient rounds, the set of possible :term:`Copeland winners<Copeland winner>` will
    converge which will result in minimal increment in :term:`Copeland regret` and thus
    the goal would be achieved.

    This algorithm uses a KL-Divergence based :term:`PAC` algorithm as a subroutine to
    determine a Copeland winner. The subroutine is based on `Algorithm 2` and
    `Algorithm 4` stated in :cite:`zoghi2015copeland`. An additional termination
    condition is used in its implementation. This additional condition stops
    the exploration phase of the subroutine when there is only one Copeland
    winner candidate left. The additional condition is introduced to ensure that
    the KL-Divergence based :term:`PAC` based algorithm starts with the exploitation phase
    (i.e. dueling a :term:`PAC`-Copeland winner against itself) earlier in some cases.
    Otherwise, the algorithm might continue to duel the one remaining candidate
    against random opponents, which could incur a higher regret.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        Number of time steps to execute for.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.

    Attributes
    ----------
    rounds
        Number of rounds the algorithm has executed.
        Corresponds to :math:`r` in :cite:`zoghi2015copeland`.
    preference_estimate
        The current estimate of the preference matrix.
    time_budget
        The number of comparisons allowed in each round. This quantity
        is dependent on the current round, hence varies with each round.
        Corresponds to :math:`T` in :cite:`zoghi2015copeland`.
    copeland_winner
        The estimated :term:`Copeland winner`.

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
    ...     metrics={"copeland_regret": AverageCopelandRegret(preference_matrix)})
    >>> scb = ScalableCopelandBandits(feedback_mechanism=feedback_mechanism, time_horizon=1000,
    ...     random_state=random_state)
    >>> scb.run()
    >>> np.round(np.sum(feedback_mechanism.results["copeland_regret"]), 2)
    68.0
    >>> scb.feedback_mechanism.get_num_duels()
    1000
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.rounds: int = 0
        self.preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms()
        )
        self.time_budget: int = 0
        self.copeland_winner: Optional[int] = None

    def step(self) -> None:
        """Run one round of the algorithm."""
        self.rounds += 1
        self.time_budget = 2 ** (2 ** self.rounds)

        assert (
            self.time_horizon is not None
        )  # for mypy. Can never be none in this class, initialized in __init__.
        kl_divergence_based_pac = KLDivergenceBasedPAC(
            feedback_mechanism=BudgetedFeedbackMechanism(
                self.feedback_mechanism,
                max_duels=self.time_horizon - self.feedback_mechanism.get_num_duels(),
            ),
            time_horizon=self.time_budget,
            random_state=self.random_state,
            epsilon=0,
            failure_probability=(np.log(self.time_budget) / self.time_budget),
            preference_estimate=self.preference_estimate,
        )
        try:
            kl_divergence_based_pac.run()
        except AlgorithmFinishedException:
            pass
        self.copeland_winner = kl_divergence_based_pac.get_copeland_winner()
