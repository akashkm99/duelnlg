"""Utilities for estimating preference matrices based on samples."""

from typing import Tuple

import numpy as np

from duelnlg.duelpy.stats.confidence_radius import ConfidenceRadius
from duelnlg.duelpy.stats.confidence_radius import TrivialConfidenceRadius
from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix


class PreferenceEstimate:
    """An estimation of a preference matrix based on samples.

    Consider this example:

    >>> preference_estimate = PreferenceEstimate(
    ...     num_arms = 3,
    ...     confidence_radius=lambda num_samples: 1/(num_samples + 1)
    ... )

    We use a trivial confidence radius for easy illustration. Note that the
    results are not accurate, you probably want to use something like
    HoeffdingConfidenceRadius in practice.

    In the beginning, nothing is known yet.

    >>> preference_estimate.get_mean_estimate_matrix()
    array([[0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5]])
    >>> preference_estimate.get_upper_estimate_matrix()
    array([[0.5, 1. , 1. ],
           [1. , 0.5, 1. ],
           [1. , 1. , 0.5]])
    >>> preference_estimate.get_lower_estimate_matrix()
    array([[0.5, 0. , 0. ],
           [0. , 0.5, 0. ],
           [0. , 0. , 0.5]])

    If we enter a sampled win, the estimated probability of that arm increases
    and the inverse probability decreases accordingly.

    >>> preference_estimate.enter_sample(0, 1, first_won=True)
    >>> preference_estimate.get_mean_estimate_matrix()
    array([[0.5, 1. , 0.5],
           [0. , 0.5, 0.5],
           [0.5, 0.5, 0.5]])

    When entering more samples, the probability keeps adjusting. Let's make it
    one win out of four.
    >>> preference_estimate.enter_sample(0, 1, first_won=False)
    >>> preference_estimate.enter_sample(0, 1, first_won=True)
    >>> preference_estimate.enter_sample(0, 1, first_won=True)
    >>> preference_estimate.get_mean_estimate_matrix()
    array([[0.5 , 0.75, 0.5 ],
           [0.25, 0.5 , 0.5 ],
           [0.5 , 0.5 , 0.5 ]])

    Meanwhile the confidence intervals have adjusted as well:

    >>> preference_estimate.get_upper_estimate_matrix()
    array([[0.5 , 0.95, 1.  ],
           [0.45, 0.5 , 1.  ],
           [1.  , 1.  , 0.5 ]])
    >>> preference_estimate.get_lower_estimate_matrix()
    array([[0.5 , 0.55, 0.  ],
           [0.05, 0.5 , 0.  ],
           [0.  , 0.  , 0.5 ]])

    And if we tighten the confidence radius, they get changed yet again:

    >>> preference_estimate.set_confidence_radius(lambda num_samples: 1/(6 * num_samples + 1))
    >>> preference_estimate.get_upper_estimate_matrix()
    array([[0.5 , 0.79, 1.  ],
           [0.29, 0.5 , 1.  ],
           [1.  , 1.  , 0.5 ]])
    >>> preference_estimate.get_lower_estimate_matrix()
    array([[0.5 , 0.71, 0.  ],
           [0.21, 0.5 , 0.  ],
           [0.  , 0.  , 0.5 ]])

    We can now also sample a complete preference matrix from a beta
    distribution:

    >>> preference_estimate.sample_preference_matrix(
    ...     random_state=np.random.RandomState(42)
    ... )
    array([[0.5       , 0.72606244, 0.4978376 ],
           [0.27393756, 0.5       , 0.44364733],
           [0.5021624 , 0.55635267, 0.5       ]])

    Parameters
    ----------
    num_arms
        The number of arms in the estimated preference matrix.
    confidence_radius
        The confidence radius to use when computing confidence intervals.
    """

    def __init__(
        self,
        num_arms: int,
        confidence_radius: ConfidenceRadius = TrivialConfidenceRadius(),
    ) -> None:
        self.num_arms = num_arms
        self.wins = np.zeros((num_arms, num_arms))
        self.confidence_radius = confidence_radius
        self._cached_mean_estimate = np.full((num_arms, num_arms), 0.5)
        self._cached_radius = None

    def set_confidence_radius(self, confidence_radius: ConfidenceRadius) -> None:
        """Set the confidence radius to the given parameter.

        Parameters
        ----------
        confidence_radius
            The confidence radius to be set as the new `confidence_radius`.
        """
        self.confidence_radius = confidence_radius
        self._cached_radius = None

    def enter_sample(
        self, first_arm_index: int, second_arm_index: int, first_won: float
    ) -> None:
        """Enter the result of a sampled duel.

        Parameters
        ----------
        first_arm_index
            The index of the first arm of the duel.
        second_arm_index
            The index of the second arm of the duel.
        first_won
            Whether the first arm won the duel.
        """
        # It would be possible to normalize the order instead of duplicating
        # the information. That would restrict us to comparable arm
        # representations though.
        if first_won is None:
            return

        self.wins[first_arm_index][second_arm_index] += first_won
        self.wins[second_arm_index][first_arm_index] += 1 - first_won

        if first_arm_index == second_arm_index:
            # Nothing to update, the preference is known.
            return

        # based on wins array, already updated
        samples = self.get_num_samples(first_arm_index, second_arm_index)

        prev = self._cached_mean_estimate[first_arm_index][second_arm_index]
        new_mean = prev + (first_won - prev) / samples

        self._cached_mean_estimate[first_arm_index][second_arm_index] = new_mean
        self._cached_mean_estimate[second_arm_index][first_arm_index] = 1 - new_mean
        self._cached_radius = None

    def get_mean_estimate(self, first_arm_index: int, second_arm_index: int) -> float:
        """Get the estimate of the win probability of `first_arm_index` against `second_arm_index`.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        float
            The estimated probability that `first_arm_index` wins against `second_arm_index`.
        """
        return self._cached_mean_estimate[first_arm_index][second_arm_index]

    def get_confidence_interval(
        self, first_arm_index: int, second_arm_index: int
    ) -> Tuple[float, float]:
        """Get the bounds of the confidence interval on the win probability.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        Tuple[float, float]
            The lower and upper bound of the confidence estimate for the
            probability that `first_arm_index` wins against `second_arm_index`.
        """
        return (
            self.get_lower_estimate(first_arm_index, second_arm_index),
            self.get_upper_estimate(first_arm_index, second_arm_index),
        )

    def _get_confidence_radius(self, first_arm_idx: int, second_arm_idx: int) -> float:
        """Get the confidence radius and fill the cache if necessary.

        Parameters
        ----------
        first_arm_idx
            The first arm of the duel.
        second_arm_idx
            The second arm of the duel.

        Returns
        -------
        float
            The current confidence value.
        """
        # Do not use the cache, since it might be "dirty" and we do not want to
        # re-compute it entirely here. Its possible to be a little more clever
        # with partial cache invalidations, but that adds book-keeping
        # overhead. If this function is used (in place of requesting one of the
        # matrices directly), its likely that only some specific samples will
        # be requested.
        if first_arm_idx == second_arm_idx:
            return 0
        return self.confidence_radius(
            self.get_num_samples(first_arm_idx, second_arm_idx)
        )

    def get_upper_estimate(self, first_arm_index: int, second_arm_index: int) -> float:
        """Get the upper estimate of the win probability of `first_arm_index` against `second_arm_index`.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        float
            The upper bound of the confidence estimate for the probability that `first_arm_index` wins against `second_arm_index`.
        """
        return min(
            self._cached_mean_estimate[first_arm_index][second_arm_index]
            + self._get_confidence_radius(first_arm_index, second_arm_index),
            1,
        )

    def get_lower_estimate(self, first_arm_index: int, second_arm_index: int) -> float:
        """Get the lower estimate of the win probability of `first_arm` against `second_arm`.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        float
            The lower bound of the confidence estimate for the probability that `first_arm` wins against `second_arm`.
        """
        return max(
            self._cached_mean_estimate[first_arm_index][second_arm_index]
            - self._get_confidence_radius(first_arm_index, second_arm_index),
            0,
        )

    def get_num_samples(self, first_arm_index: int, second_arm_index: int) -> int:
        """Get the number of times a duel between first_arm and second_arms was sampled.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        int
            The number of times a duel between the two arms was sampled,
            regardless of the arm order.
        """
        return (
            self.wins[first_arm_index][second_arm_index]
            + self.wins[second_arm_index][first_arm_index]
        )

    def get_radius_matrix(self) -> np.array:
        """Seed the confidence radius cache and return it.

        Returns
        -------
        np.array
            A numpy matrix containing the current confidence radius values.
        """
        if self._cached_radius is None:
            num_samples = self.wins + self.wins.T
            # Use numpy to find the set of sample-sizes we're interested in and map
            # the results back to the full matrix.
            unique_num_samples, inverse_indices = np.unique(
                num_samples, return_inverse=True
            )
            unique_confidences = np.array(
                [self.confidence_radius(samples) for samples in unique_num_samples]
            )
            full_confidences = unique_confidences[inverse_indices].reshape(
                num_samples.shape
            )
            np.fill_diagonal(full_confidences, 0.0)
            self._cached_radius = full_confidences
        return self._cached_radius

    def get_mean_estimate_matrix(self) -> PreferenceMatrix:
        """Get the current mean estimates as a PreferenceMatrix.

        Returns
        -------
        PreferenceMatrix
            The current mean estimate.
        """
        return PreferenceMatrix(self._cached_mean_estimate)

    def get_upper_estimate_matrix(self) -> PreferenceMatrix:
        """Get the current upper estimates as a PreferenceMatrix.

        Returns
        -------
        PreferenceMatrix
            The current mean estimate.
        """
        return PreferenceMatrix(
            np.clip(
                self._cached_mean_estimate + self.get_radius_matrix(), a_min=0, a_max=1
            )
        )

    def get_lower_estimate_matrix(self) -> PreferenceMatrix:
        """Get the current lower estimates as a PreferenceMatrix.

        Returns
        -------
        PreferenceMatrix
            The current mean estimate.
        """
        return PreferenceMatrix(
            np.clip(
                self._cached_mean_estimate - self.get_radius_matrix(), a_min=0, a_max=1
            )
        )

    def get_pessimistic_copeland_score_estimates(self) -> np.array:
        """Get pessimistic estimates for every arm's Copeland score.

        This only counts wins that have a probability of above 50% in the
        pessimistic estimate. Those wins are "certain", assuming the confidence
        interval is correct.
        """
        wins = self.get_lower_estimate_matrix().preferences > 1 / 2
        return wins.sum(axis=1)

    def get_optimistic_copeland_score_estimates(self) -> np.array:
        """Get optimistic estimates for every arm's Copeland score.

        This counts every win that is considered possible within the confidence
        interval.
        """
        wins = self.get_upper_estimate_matrix().preferences > 1 / 2
        return wins.sum(axis=1)

    def sample_preference_matrix(
        self, random_state: np.random.RandomState
    ) -> PreferenceMatrix:
        """Sample a preference matrix based on a Beta distribution.

        The outcome is a PreferenceMatrix object which is initialized from a sampled
        preference matrix. In this preference matrix, each pairwise preference is
        drawn from a beta-distribution which is parameterized on the results of prior
        duels.

        Parameters
        ----------
        random_state
            A numpy random state.

        Returns
        -------
        PreferenceMatrix
            A PreferenceMatrix object which is initialized from a preference matrix which
            is sampled on a Beta distribution.
        """
        # Construct the parameters of a beta distribution to sample preference
        # probabilities.
        beta_a = self.wins + 1
        beta_b = beta_a.T
        # Only the upper triangle is important, the rest is adjusted afterwards.
        upper_triangle_preferences = random_state.beta(beta_a, beta_b)
        return PreferenceMatrix.from_upper_triangle(upper_triangle_preferences)

    def __str__(self) -> str:
        """Produce a string representation of the estimate."""
        result = ""
        for first_arm_index in range(self.num_arms):
            row = f"{first_arm_index} |"
            for second_arm_index in range(self.num_arms):
                mean = self.get_mean_estimate(first_arm_index, second_arm_index)
                radius = self.confidence_radius(
                    self.get_num_samples(first_arm_index, second_arm_index)
                )
                row += "  {:.2f}+-{:.2f}".format(mean, radius)
            result += row
            result += "\n"
        return result
