from duelnlg.duelpy.stats import PreferenceEstimate
from duelnlg.duelpy.algorithms.algorithm import Algorithm
import numpy as np


class UniformExploration(Algorithm):
    def __init__(self, feedback_mechanism, random_state=np.random.RandomState()):

        self.feedback_mechanism = feedback_mechanism
        self.num_arms = feedback_mechanism.get_num_arms()
        self.preference_estimate = PreferenceEstimate(feedback_mechanism.get_num_arms())
        self.random_state = random_state

    def step(self):
        random_arms = self.random_state.randint(self.num_arms, size=[2])
        self.preference_estimate.enter_sample(
            random_arms[0],
            random_arms[1],
            self.feedback_mechanism.duel(random_arms[0], random_arms[1]),
        )

    def get_winner(self):
        return list(
            self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )[0]
