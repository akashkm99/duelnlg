import numpy as np
from duelnlg.duelpy.models.default import Default
from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix
import logging


class UCBElimination(Default):
    def __init__(
        self,
        metric_name,
        metric_name_ensemble,
        link_function,
        num_ensemble,
        copeland_threshold=0.5,
        confidence_scale=2.0,
    ):

        self.metric_name = metric_name
        self.metric_name_ensemble = self.metric_name_ensemble
        self.link_function = link_function
        self.num_ensemble = num_ensemble
        self.model_names = [
            "{}_{}_{}_prob".format(model_name, i, link_function)
            for i in range(num_ensemble)
        ]
        self.model_names_pred = "{}_{}_{}_predictions".format(
            metric_name, link_function
        )

        self.copeland_threshold = copeland_threshold
        self.confidence_scale = confidence_scale

    def start(self, env):

        arms = np.array(env.arms)
        self.num_arms = env.num_arms
        self.ucb_matrix = np.zeros([self.num_arms, self.num_arms], dtype=np.float32)

        for system1, system2 in env.samples:
            if system1 == system2:
                self.ucb_matrix[system1, system2] = 0.5
            else:
                curr_samples = env.samples[(system1, system2)]
                preferences = []
                ucb_preferences = []

                for i in range(len(curr_samples)):
                    probs = np.array(
                        [sample[model_name] for model_name in self.model_names]
                    )
                    ucb_preference = min(
                        1,
                        sample[self.model_names_pred]
                        + self.confidence_scale * np.std(probs),
                    )
                    ucb_preferences.append(ucb_preference)

                ucb_preferences = np.mean(ucb_preferences)
                self.ucb_matrix[system1, system2] = ucb_preferences

        optimistic_copeland_scores = np.sum(self.ucb_matrix > 0.5, axis=1) / (
            self.num_arms - 1
        )
        arms_to_keep = arms[
            optimistic_copeland_scores > self.copeland_threshold
        ].tolist()
        env.arms = list(arms_to_keep)
