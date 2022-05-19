import numpy as np
import pandas as pd
import sys
import os
import pickle
from glob import glob
import json
from duelnlg.duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix
from duelnlg.duelpy.feedback.nlg_feedback import NLGFeedback


class NLGFeedbackHyperParam(NLGFeedback):
    def __init__(self, model, filename, random_seed=None, num_arms=3):
        super(NLGFeedbackHyperParam, self).__init__(model, filename, random_seed)
        self.num_arms = num_arms
        copeland_scores = self.preference_matrix.get_copeland_scores()
        arms_to_keep = np.argsort(copeland_scores)[::-1][: self.num_arms]
        self.arms = np.arange(self.num_arms)
        arm_mapping = {}
        for i, arm in enumerate(arms_to_keep):
            arm_mapping[arm] = i

        new_samples = {}
        for arm1, arm2 in self.samples:
            if arm1 not in arm_mapping or arm2 not in arm_mapping:
                continue
            new_arm1 = arm_mapping[arm1]
            new_arm2 = arm_mapping[arm2]
            new_samples[(new_arm1, new_arm2)] = self.samples[(arm1, arm2)]

        self.samples = new_samples
        self.preference_matrix_data = self.preference_matrix_data[arms_to_keep][
            :, arms_to_keep
        ]
        self.preference_matrix = PreferenceMatrix(self.preference_matrix_data)
