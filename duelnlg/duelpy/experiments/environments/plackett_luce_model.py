"""An environment based on the Plackett-Luce model."""

from typing import List
from typing import Optional

import numpy as np

from duelnlg.duelpy.feedback import MatrixFeedback
from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix
from duelnlg.duelpy.util.utility_functions import argmax_set


class PlackettLuceModel(MatrixFeedback):
    r"""A feedback-mechanism based on the Plackett-Luce model.

    The probabilities are based on a non-negative 'skill' value for each arm. The probability of arm i winning against arm j is then based on their skills v: :math:`\frac{v_i}{v_i+v_j}`.

    Parameters
    ----------
    num_arms
        The size of the preference matrix to generate.
    random_state
        The numpy random state that will be used for sampling and generating skills, if they are not given.
    skill_vector
        Optional, contain scalars representing the skill of each arm. Must be of length `num_arms` and only contain non-negative values.
    """

    def __init__(
        self,
        num_arms: int,
        random_state: np.random.RandomState,
        skill_vector: Optional[List[float]] = None,
    ):

        if skill_vector is not None:
            if len(skill_vector) != num_arms:
                raise ValueError(
                    "The num_arms parameter needs to be equal to the length of the given skill_vector."
                )
            if any([x < 0 for x in skill_vector]):
                raise ValueError("The values in skill_vector must be non-negative.")
        else:
            skill_vector = random_state.uniform(size=num_arms)
        self._skill_vector = skill_vector

        preferences = np.full((num_arms, num_arms), 0.5)
        for first_arm_idx in range(num_arms):
            for second_arm_idx in range(first_arm_idx):
                relative_preference = skill_vector[first_arm_idx] / (
                    skill_vector[first_arm_idx] + skill_vector[second_arm_idx]
                )
                preferences[first_arm_idx][second_arm_idx] = relative_preference
                preferences[second_arm_idx][first_arm_idx] = 1 - relative_preference
        preference_matrix = PreferenceMatrix(preferences)

        super().__init__(preference_matrix=preference_matrix, random_state=random_state)

    def get_best_arms(self) -> List[int]:
        """Get a list of all best arms. This can (and usually is) only be the Condorcet winner. But if multiple arms have the same maximal skill value, they are returned as Copeland winners.

        Returns
        -------
        List[int]
            A list of all the best arms.
        """
        return argmax_set(self._skill_vector)

    def get_arbitrary_ranking(self) -> List[int]:
        """Get any correct ranking of the arms.

        Returns
        -------
        List[int]
            Any correct ranking of the arms, must not be the only correct one.
        """
        return sorted(
            list(range(len(self._skill_vector))), key=lambda i: self._skill_vector[i]
        )

    def test_ranking(self, ranking: List[int]) -> bool:
        r"""Check whether a ranking is admissible.

        Generating all correct rankings might take too long (possibly :math:`\mathit{num\_arms}!` many), so we simply allow checking of given rankings.

        Parameters
        ----------
        ranking
            The ranking that should be checked

        Returns
        -------
        bool
            Whether the ranking is correct.
        """
        current_skill = self._skill_vector[ranking[0]]
        for arm in ranking:
            if self._skill_vector[arm] > current_skill:
                return False
            current_skill = self._skill_vector[arm]
        return True
