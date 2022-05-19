"""An implementation of the Double Thompson Sampling algorithm for Dueling Bandits."""
from typing import Optional

import numpy as np
from scipy.special import rel_entr

from duelnlg.duelpy.algorithms.interfaces import SingleCopelandProducer
from duelnlg.duelpy.feedback import FeedbackMechanism
from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelnlg.duelpy.util.utility_functions import argmax_set
from duelnlg.duelpy.util.utility_functions import argmin_set


class DoubleThompsonSampling(SingleCopelandProducer):
    r"""Implementation of the Double Thompson Sampling algorithm.

    The goal of this algorithm is to find the :term:`Copeland winner` while incurring minimal average :term:`Copeland regret`. If there
    exist a :term:`Condorcet winner`, the :term:`Copeland winner` is also the :term:`Condorcet winner`.

    No further assumptions about the arms are needed.

    It uses the average :term:`Copeland regret`. For the general Copeland winner setting, D-TS achieves :math:`\mathcal{O}(K^2 \log T)`. Meanwhile, For
    the Condorcet setting and many practical Copeland settings, D-TS achieves :math:`\mathcal{O}(K \log T + K^2 \log \log T)`
    using a back substitution argument.

    The Double Thompson Sampling (D-TS) algorithm in paper :cite:`huasen2016dts`
    includes both  the Condorcet and the general Copeland setting. D-TS uses a double sampling structure
    where the first as well as the second candidates are selected according to independently drawn samples
    from the beta posterior distribution and then dueled. The double sampling structure of D-TS is better suited for
    dueling bandits nature. Unlike :class:`RelaviveConfidenceSampling<duelpy.algorithms.RelativeConfidenceSampling>`, launching two independent rounds of sampling provide us the opportunity to
    select the same arm in both rounds. This allows to compare the winners against themselves which significantly
    reduced regret. The confidence bounds in the algorithm are used to eliminate the
    unlikely arms which are ineligible to be winner arm and thus avoids suboptimal comparisons. While selecting the
    first candidate arm and the second candidate arm, the confidence bound is used to eliminate non-likely winners.
    So, D-TS is more robust in practice.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        This states the number of comparision to be done before the algorithm terminates.
    random_state
        Used for random choices in the algorithm.
    exploratory_constant
        Optional, the confidence radius grows proportional to the square root of this value. Corresponds to `\alpha` in
        :cite:`huasen2016dts`. The value of ``exploratory_constant`` must be greater than :math:`0.5`. Default value is ``0.51``

    Attributes
    ----------
    feedback_mechanism
    exploratory_constant
    random_state
    time_horizon
    preference_estimate
        Stores estimates of arm preferences

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
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"copeland_regret": AverageCopelandRegret(preference_matrix)}
    ... )
    >>> test_object = DoubleThompsonSampling(feedback_mechanism, random_state=random_state, time_horizon=100)
    >>> test_object.run()
    >>> test_object.get_copeland_winner()
    2
    >>> np.round(np.sum(feedback_mechanism.results["copeland_regret"]), 2)
    17.5
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.RandomState = None,
        exploratory_constant: float = 0.51,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.time_step = 0
        self.exploratory_constant = exploratory_constant
        self.preference_estimate = PreferenceEstimate(
            num_arms=self.feedback_mechanism.get_num_arms()
        )
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

    def _update_confidence_radius(self) -> None:
        r"""Update the confidence radius using latest failure probability.

        Failure probability for the upper confidence bound is :math:`1/t^(2 * \alpha)`
        where :math:`t` is the current round of the algorithm and :math:`\alpha` is the
        exploratory constant.

        Refer to :cite:`huasen2016dts` for further details.
        """
        failure_probability = 1 / (self.time_step ** (2 * self.exploratory_constant))
        confidence_radius = HoeffdingConfidenceRadius(failure_probability)
        self.preference_estimate.set_confidence_radius(confidence_radius)

    def _choose_first_candidate(self) -> int:
        r"""Choose a champion arm whose Copeland score is high in a sample.

        Select an  ``arm_c`` from the potential champion arms whose copeland score is high based on the
        preference matrix computed under beta distribution. If there exist a tie between arms, ``arm_c``  is
        selected randomly. Also, upper confidence bound is used to estimate the preference between the arms. So,
        potential champions arms are selected upon normalized copeland scores computed based upon the preference
        estimate of arms using upper confidence bound.

        Return
        ------
        int
            The champion arm with high copeland score.
        """
        # select the potential champion according to their normalized copeland scores from the preference estimate
        # upper confidence bound
        potential_champion = (
            self.preference_estimate.get_upper_estimate_matrix().get_copeland_winners()
        )

        non_potential_champion = (
            set(self.feedback_mechanism.get_arms()) - potential_champion
        )

        # sample preference matrix between the arm through beta distribution
        sample_preference_matrix = self.preference_estimate.sample_preference_matrix(
            self.random_state
        )

        # calculate copeland score for all the arms based on sample preference matrix to remove non-likely winner arms
        # Ties between the arms are broken randomly
        arm_c = self.random_state.choice(
            argmax_set(
                sample_preference_matrix.get_copeland_scores(),
                exclude_indexes=list(non_potential_champion),
            )
        )

        return arm_c

    def _choose_second_candidate(self, champion: int) -> int:
        r"""Choose challenger arm which is likely to win against the champion.

        Select an ``arm_d`` from the potential challenger arms whose preference is high compared with the
        champion arm (``arm_c``). The preference between the challenger arms with champion arm is based on. If
        there exist ties between arms, ``arm_d`` is selected randomly. Also, lower confidence bound is used to
        estimate the preference between the arms. So, arms whose lower preference estimate is less than 0.5 are
        selected as potential challengers.

        Return
        ------
        int
            The challenger arm with high preference than champion arm.
        """
        # preference estimate between the arms using lower confidence bound.
        # select the potential challenger (possible for arm_d)
        potential_challenger = argmax_set(
            self.preference_estimate.get_lower_estimate_matrix().preferences[:][
                champion
            ]
            <= 0.5
        )
        non_potential_challenger = set(self.feedback_mechanism.get_arms()) - set(
            potential_challenger
        )

        # sample preference matrix between the arm through beta distribution
        # sample the preference with champion arm from the sampled preference matrix.
        sample_preference_with_champion = self.preference_estimate.sample_preference_matrix(
            self.random_state
        ).preferences[
            :
        ][
            champion
        ]

        #  Choosing only from uncertain pairs (potential challenger). Ties are broken randomly.
        arm_d = self.random_state.choice(
            argmax_set(sample_preference_with_champion, list(non_potential_challenger))
        )
        return arm_d

    def get_copeland_winner(self) -> Optional[int]:
        """Compute single Copeland winner as per the estimated preference matrix.

        Returns
        -------
        Optional[int]
            Copeland winner from preference estimate.
        """
        # if self.is_finished():
        return list(
            self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )[0]

    def get_winner(self):
        return self.get_copeland_winner()

    def step(self) -> None:
        """Run one round of an algorithm."""
        self.time_step += 1
        self._update_confidence_radius()
        arm_c = self._choose_first_candidate()
        arm_d = self._choose_second_candidate(arm_c)
        arm1, arm2, score = self.feedback_mechanism.get_duel(arm_c, arm_d)
        self.preference_estimate.enter_sample(arm1, arm2, score)


class DoubleThompsonSamplingPlus(DoubleThompsonSampling):
    r"""Implementation of the extended version of D-TS.

    The goal of this algorithm is to find the Copeland winner while incurring minimal average Copeland regret. If there
    exist a Condorcet winner, the Copeland winner is also the Condorcet winner.

    It is assumed that the Copeland winner exists.

    It uses the average Copeland regret. For the general Copeland bandit, D-TS+ achieves :math:`\mathcal{O}(N^2 \log T)`.
    Meanwhile, for the Condorcet dueling bandit and many practical Copeland dueling bandit, D-TS+ achieves :math:`\mathcal{O}(N
    \log T + N^2 \log \log T)` using a back substitution argument. :math:`N` is the number of arms.

    As presented in :cite:`huasen2016dts`, the D-TS+ algorithm just changes the tie-breaking
    criterion while selecting the first candidate(i.e the selection of the first candidate). During the selection of
    the first candidate , all the potential champions are selected as potential Copeland winners based on an upper
    estimate of the preference probability. Now, To remove the non-likely winners, preference probability between the
    arms is sampled using beta distribution. Then, For all the potential arms, one vs all arm regret based on sampled
    preference is calculated using KL divergence. The arm with minimal one-vs-all regret is selected as a first
    candidate. The one vs all regret of a potential arm is defined by the summation of Copeland regret of a
    respective potential arm with any other arm per KL-divergence (preference of potential arm over another arm, :math:`0.5`).

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        This states the number of comparision to be done before the algorithm terminates.
    random_state
        Used for random choices in the algorithm.
    exploratory_constant
        Optional, the confidence radius grows proportional to the square root of this value. Corresponds to
        :math:`\alpha` in :cite:`huasen2016dts`. The value of ``exploratory_constant`` must be greater than :math:`0.5`.
        Default value is ``0.51``.

    Attributes
    ----------
    feedback_mechanism
    exploratory_constant
    random_state
    time_horizon
    preference_estimate
        Stores estimates of arm preferences

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
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"copeland_regret": AverageCopelandRegret(preference_matrix)}
    ... )
    >>> test_object = DoubleThompsonSamplingPlus(feedback_mechanism, exploratory_constant=0.51, random_state=random_state, time_horizon=100)
    >>> test_object.run()
    >>> test_object.get_copeland_winner()
    2
    >>> np.round(np.sum(feedback_mechanism.results["copeland_regret"]), 2)
    52.5
    """

    def _choose_first_candidate(self) -> int:
        r"""Choose a champion arm whose Copeland score is high in a sample.

        Select an ``arm_c`` from the potential champion arms whose Copeland score is high based on the preference
        matrix computed under beta distribution. If there exist a tie between arms, its broken by comparing the arms
        whose one vs all regret along with KL divergent is minimum. Also, upper confidence bound is used to estimate
        the preference between the arms. So, potential champion arms are selected upon normalized Copeland scores
        computed based upon the preference estimate of arms using upper confidence bound.

        Return
        ------
        int
            The champion arm with high Copeland score.
        """
        potential_champion_arms = (
            self.preference_estimate.get_upper_estimate_matrix().get_copeland_winners()
        )
        non_potential_champion_arms = (
            set(self.feedback_mechanism.get_arms()) - potential_champion_arms
        )

        # sample preference matrix between the arm through beta distribution
        sample_preference_matrix = self.preference_estimate.sample_preference_matrix(
            self.random_state
        )

        normalized_copeland_scores = (
            sample_preference_matrix.get_normalized_copeland_scores()
        )
        max_normalized_copeland_score = np.amax(normalized_copeland_scores)

        regret_one_vs_all = np.zeros(self.feedback_mechanism.get_num_arms())
        all_arms = np.array(self.feedback_mechanism.get_arms())
        for potential_champion in potential_champion_arms:
            # All arms whose estimated preference against the potential
            # champion is not 1/2 (most of the time this will be all arms
            # except the potential champion itself).
            challengers = all_arms[
                sample_preference_matrix.preferences[potential_champion] != 0.5
            ]
            # Average Copeland regret of all other arms compared to the
            # potential champion
            average_copeland_regret_values = max_normalized_copeland_score - 0.5 * (
                normalized_copeland_scores[potential_champion]
                + normalized_copeland_scores[challengers]
            )
            kl_divergences = rel_entr(
                1
                - sample_preference_matrix.preferences[potential_champion][challengers],
                0.5,
            ) + rel_entr(
                sample_preference_matrix.preferences[potential_champion][challengers],
                0.5,
            )

            regret_values = average_copeland_regret_values / kl_divergences
            regret_one_vs_all[potential_champion] = np.sum(regret_values)
        arm_c = argmin_set(regret_one_vs_all, list(non_potential_champion_arms),)[0]
        return arm_c
