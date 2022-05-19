import numpy as np
from duelnlg.duelpy.feedback.nlg_feedback import NLGFeedback


class NLGFeedbackDelay(NLGFeedback):
    def __init__(self, model, filename, random_seed=None, delay=0):
        super(NLGFeedbackDelay, self).__init__(model, filename, random_seed)
        self.delay = delay
        self.feedback_buffer = []

    def get_duel(self, arm_i_index: int, arm_j_index: int) -> float:
        """Perform a duel between two arms based on a given probability matrix.

        Parameters
        ----------
        arm_i_index
            The challenger arm.
        arm_j_index
            The arm to compare against.

        Returns
        -------
        bool
            True if arm_i_index wins.
        """
        self.current_arms = (arm_i_index, arm_j_index)
        self.history.append(self.current_arms)
        num_samples = len(self.samples[arm_i_index, arm_j_index])

        if num_samples == 0:
            raise AssertionError(
                "Number of samples between arms {} and {} is 0!".format(
                    arm_i_index, arm_j_index
                )
            )

        if arm_i_index == arm_j_index:
            return (arm_i_index, arm_j_index, 0.5)

        random_idx = np.random.randint(0, num_samples, size=[1])[0]
        sample = self.samples[arm_i_index, arm_j_index][random_idx]

        if self.model.get_duel(sample):
            return (arm_i_index, arm_j_index, self.model.duel(sample))
        else:
            self.feedback_buffer.append((arm_i_index, arm_j_index, sample["label"]))
            if len(self.feedback_buffer) <= self.delay:
                return (0, 0, None)
            else:
                self.sample_complexity += 1
                return self.feedback_buffer.pop(0)
