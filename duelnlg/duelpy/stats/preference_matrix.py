"""A preference matrix with associated utility functions."""

from typing import Any
from typing import Optional
from typing import Set

import numpy as np

from duelnlg.duelpy.util.utility_functions import argmax_set


class PreferenceMatrix:
    """Represents a preference matrix with associated utility functions.

    Parameters
    ----------
    preferences
        A quadratic matrix where p[i, j] specifies the probability that arm i
        wins against arm j. This implies p[j, i] = 1 - p[i, j] and p[i, i] =
        0.5.
    """

    def __init__(
        self, preferences: np.array,
    ):
        self.preferences = preferences

    @staticmethod
    def from_upper_triangle(matrix: np.array) -> "PreferenceMatrix":
        """Construct a coherent preference matrix from an upper triangle.

        All entries below the diagonal (including the diagonal) are ignored.
        The diagonal is filled in with 0.5, the lower triangle is filled in to
        match the upper triangle.

        >>> matrix = np.array([[-1, 0.3, 0.2],
        ...                    [42, 0.1, 0.8],
        ...                    [ 0,  -5, 0.1]])
        >>> PreferenceMatrix.from_upper_triangle(matrix)
        array([[0.5, 0.3, 0.2],
               [0.7, 0.5, 0.8],
               [0.8, 0.2, 0.5]])

        Parameters
        ----------
        matrix
            The upper-triangle matrix. All entries below the diagonal are
            ignored.

        Returns
        -------
        PreferenceMatrix
            The resulting coherent preference matrix.
        """
        upper_triangle = np.triu(matrix)
        # For some reason pylint mistakenly assumes that upper_triangle is a
        # tuple and therefore has no "T" member.
        # pylint: disable=no-member
        lower_triangle = np.tril(1 - upper_triangle.T, -1)
        result = upper_triangle + lower_triangle
        np.fill_diagonal(result, 0.5)
        return PreferenceMatrix(result)

    # Unfortunately numpy's indexing is not typed, so we can't type this either
    # if we don't want to lose any of its power.
    def __getitem__(self, key: Any) -> Any:
        """Get a preference probability."""
        return self.preferences[key]

    def get_num_arms(self) -> int:
        """Get the number of arms in the preference matrix.

        Returns
        -------
        int
            The number of arms.
        """
        return self.preferences.shape[0]

    def get_condorcet_winner(self) -> Optional[int]:
        """Get the index of the Condorcet winner if one exists.

        The Condorcet winner is the arm that is expected to beat every other
        arm in a pairwise comparison.

        Returns
        -------
        Optional[int]
            The index of the Condorcet winner if one exists.
        """
        copeland_scores = self.get_copeland_scores()
        copeland_winners = argmax_set(copeland_scores)
        # condorcet winner is an arm that beats k-1 arms where k is the total number of the arms.
        # Also,  we can say that condorcet winner is an arm whose copeland score is equal to k-1 arms.
        if (
            len(copeland_winners) == 1
            and copeland_scores[copeland_winners[0]] == self.get_num_arms() - 1
        ):
            return copeland_winners[0]
        return None

    def get_copeland_winners(self) -> Set[int]:
        """Get the set of Copeland winners.

        A Copeland winner is an arm that has the highest number of expected
        wins against all other arms. This does not need to be unique, since
        multiple arms can have the same number of expected wins.

        Returns
        -------
        Set[int]
            The indices of the Copeland winners.
        """
        return set(argmax_set(self.get_copeland_scores()))

    def get_copeland_scores(self) -> np.array:
        """Calculate Copeland scores for each arm.

        The Copeland score of an arm is the number of other arms that the arm is expected to win against.

        Returns
        -------
        np.array
            A 1-D array with the Copeland scores.
        """
        return (self.preferences > 0.5).sum(axis=1)

    def get_normalized_copeland_scores(self) -> np.array:
        """Calculate the normalized Copeland scores for each arm.

        The normalized Copeland score of an arm is the fraction of other arms it is expected to win against.

        Returns
        -------
        np.array
            A 1-D array with the normalized Copeland scores.
        """
        return self.get_copeland_scores() / (self.get_num_arms() - 1)

    def get_borda_scores(self):

        return ((self.preferences > 0.5) * self.preferences).sum(axis=1)

    def get_borda_winners(self):
        return set(argmax_set(self.get_borda_scores()))

    def calculate_average_copeland_regret_arms(self, arm_i: int, arm_j: int) -> float:
        """Calculate Copeland regret with respect to normalized Copeland score.

        Parameters
        ----------
        arm_i
            The arm with respect to which the regret is calculated
        arm_j
            The challenger arm

        Returns
        -------
        float
            A average Copeland regret.
        """
        normalized_copeland_scores = self.get_normalized_copeland_scores()
        max_normalized_copeland_score = np.amax(normalized_copeland_scores)
        return max_normalized_copeland_score - 0.5 * (
            normalized_copeland_scores[arm_i] + normalized_copeland_scores[arm_j]
        )

    def get_epsilon_condorcet_winners(self, epsilon: float) -> Set[int]:
        """Find all epsilon-Condorcet winners."""
        candidates = list(range(self.get_num_arms()))
        for arm_1 in range(self.get_num_arms()):
            for arm_2 in range(self.get_num_arms()):
                if arm_1 == arm_2 or arm_1 not in candidates:
                    continue
                if self.preferences[arm_1, arm_2] <= 1 / 2 - epsilon:
                    candidates.remove(arm_1)
                    break
        return set(candidates)

    def get_borda_scores(self) -> np.array:
        """Calculate the Borda score, also called sum of expectations.

        Returns
        -------
        np.array
            A 1-D array with the Borda scores.
        """
        return (np.sum(self.preferences, axis=1) - 0.5) / (self.get_num_arms() - 1)

    def get_borda_winners(self) -> Set[int]:
        """Get the set of borda winners.

        A Borda winner is an arm that has the highest Borda score. This does not need to be unique, since
        multiple arms can have the same number of expected wins.

        Returns
        -------
        Set[int]
            The indices of the Borda winners.
        """
        return set(argmax_set(self.get_borda_scores()))

    def __repr__(self) -> str:
        """Compute a string representation of the preference matrix."""
        return repr(self.preferences)

    def get_stats(self):

        margin_matrix = self.preferences - 0.5
        n = self.preferences.shape[0]

        sst = True  # Strong stochastic transitivity
        sti = True  # Stochastic triangle inequality
        rst = True  # Relaxed stochastic transitivity
        gamma = 1  # Gamma in RST

        for i in range(n):
            for j in range(1, n):
                for k in range(2, n):
                    if margin_matrix[i, j] > 0 and margin_matrix[j, k] > 0:

                        if margin_matrix[i, k] < max(
                            margin_matrix[i, j], margin_matrix[j, k]
                        ):
                            sst = False

                        if (
                            margin_matrix[i, k]
                            > margin_matrix[i, j] + margin_matrix[j, k]
                        ):
                            sti = False

                        if margin_matrix[i, k] > 0:
                            gamma = max(
                                gamma,
                                max(margin_matrix[i, j], margin_matrix[j, k])
                                / margin_matrix[i, k],
                            )
                        else:
                            rst = False
                            gamma = float("inf")

        copeland_scores = self.get_copeland_scores()
        copeland_scores_sorted = np.sort(copeland_scores)
        total_order = np.all(copeland_scores_sorted == np.arange(n))
        condorcet_winner = copeland_scores_sorted[-1] == n - 1
        unique_borda_winner = len(self.get_borda_winners()) == 1
        distinct_borda_scores = len(np.unique(self.get_borda_scores())) == n
        min_margin = np.min(
            np.absolute((margin_matrix + np.eye(n, dtype=np.float32) * 0.5))
        )
        no_ties = min_margin != 0

        stats = {
            "Min Margin": min_margin,
            "Total order": total_order,
            "No ties": no_ties,
            "Condorcet winner": condorcet_winner,
            "Strong stochastic transitivity": sst,
            "Relaxed stochastic transitivity": rst,
            "RST Gamma": gamma,
            "Stochastic triangle inequality": sti,
            "Unique Borda winner": unique_borda_winner,
            "Distinct Borda scores": distinct_borda_scores,
        }

        return stats
