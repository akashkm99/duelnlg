import numpy as np
import sys
from duelnlg.duelpy.models.default import Default
import logging
from collections import Counter

eps = 1e-9


class Uncertainity(Default):
    def __init__(
        self,
        metric_name,
        metric_name_ensemble,
        link_function,
        num_ensemble,
        uncertainity_threshold=0.2,
        uncertainity_strategy="BALD",
    ):

        self.metric_name = metric_name
        self.metric_name_ensemble = metric_name_ensemble
        self.link_function = link_function
        self.num_ensemble = num_ensemble

        self.model_names = [
            "{}_{}_{}_prob".format(metric_name_ensemble, i, link_function)
            for i in range(num_ensemble)
        ]
        self.model_names_pred = "{}_{}_{}_predictions".format(
            metric_name, link_function
        )

        self.uncertainity_threshold = uncertainity_threshold
        self.uncertainity_strategy = uncertainity_strategy

    def cross_entropy(self, y_true, y_pred, reduction="mean"):
        loss = -1 * (
            y_true * (np.log(y_pred + eps))
            + (1 - y_true) * (np.log((1 - y_pred) + eps))
        )
        return loss

    def bald(self, y_pred):
        y_mean = np.mean(y_pred)
        entropy_mean = self.cross_entropy(y_mean, y_mean, reduction="none")
        entropy = self.cross_entropy(y_pred, y_pred, reduction="none")
        mean_entropy = np.mean(entropy)
        return entropy_mean - mean_entropy

    def std(self, y_pred):
        return np.std(y_pred)

    def get_duel(self, sample):

        if self.model_name not in sample:
            raise NotImplementedError(
                "Model {} is not implemented, please add the model's predictions to the sample".format(
                    self.model_name
                )
            )

        probs = np.array([sample[model_name] for model_name in self.model_names])
        if self.uncertainity_strategy == "BALD":
            bald = self.bald(probs)
            return bald < self.uncertainity_threshold

        elif self.uncertainity_strategy == "std":
            std_value = self.std(probs)
            return std_value < self.uncertainity_threshold
        else:
            raise NotImplementedError(
                "Only BALD/STD uncertainity estimation is implemented"
            )

    def duel(self, sample):
        return sample[self.model_names_pred]
