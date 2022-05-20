import numpy as np
from duelnlg.duelpy.models.default import Default


class Mixing(Default):
    def __init__(
        self,
        metric_name,
        link_function,
        mixing_probability=0.2,
        mixing_strategy="hard",
        random_state=None,
    ):

        self.metric_name = metric_name
        self.link_function = link_function

        self.model_name = "{}_{}".format(metric_name, link_function)
        self.mixing_probability = mixing_probability
        self.mixing_strategy = mixing_strategy

        if self.mixing_strategy == "hard":
            self.model_name += "_predictions"  # use predicted values (0, 1 or 0.5)
        elif self.mixing_strategy == "soft":
            self.model_name += "_prob"  # use the probablities (b/w 0 and 1)
        else:
            raise NotImplementedError(
                "Only soft and hard mixing strategies are implemented!"
            )

        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

    def get_duel(self, sample):
        return self.random_state.binomial(n=1, p=self.mixing_probability)

    def duel(self, sample):

        if self.model_name not in sample:
            raise NotImplementedError(
                "Metric {} with link function {} is not implemented, \
            please add the model's predictions to the sample".format(
                    self.metric_name, self.link_function
                )
            )

        outcome = sample[self.model_name]
        return outcome
