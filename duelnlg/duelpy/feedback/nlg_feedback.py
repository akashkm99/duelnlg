"""Gather feedback from a ground-truth preference matrix."""

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import sys
import os
import pickle
from glob import glob
import json

# import random
# from random import shuffle

from duelnlg.duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix
from duelnlg.duelpy.models.default import Default


class NLGFeedback(FeedbackMechanism):
    """Compare two arms based on the nlg human evaluation data.

    Parameters
    ----------
    filename
        The path to the file containing the nlg relative ranking 
    preference_matrix
        A quadratic matrix where p[i, j] specifies the probability that arm i
        wins against arm j. This implies p[j, i] = 1 - p[i, j] and p[i, i] =
        0.5.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    """

    def __init__(self, model, filename: str, random_seed: int = None):

        np.random.seed(random_seed)
        # random.seed(random_seed)
        # pd.set_option("display.precision", 2)

        self.filename = filename

        filetype = filename.split(".")[-1]
        if filetype != "pkl":
            raise ValueError(
                "Given file: {} is not a processed pkl file".format(filename)
            )

        with open(filename, "rb") as fp:
            processed_data = pickle.load(fp)

        self.preference_matrix_data = processed_data["preference_matrix"]
        self.preference_matrix = PreferenceMatrix(self.preference_matrix_data)
        self.samples = processed_data["samples"]
        self.idx_to_arm = processed_data["idx_to_arm"]
        self.count_matrix = processed_data["count_matrix"]
        self.total_samples = processed_data["total_samples"]
        self.arms = processed_data["arms"]
        self.history: List[Tuple[int, int]] = []
        self.num_arms = len(self.arms)
        self.num_pairs = (self.num_arms) * (self.num_arms - 1) / 2
        self.sample_complexity = 0

        self.model = model
        model.start(self)
        # self.shuffle_samples()
        # self.num_pulls = np.zeros([self.num_arms, self.num_arms], dtype=np.int32)

        # print ("-"*20 + "Enviroinmet Details" + "-"*20)
        # print ("Enviroinment Name: nlg {}".format(filename))
        # print ("Total Number of Arms: {}".format(self.num_arms))
        # print ("Total number of samples: {}".format(self.total_samples))
        # print ("Average number of samples per pair: {:.0f}".format(self.total_samples/(self.num_pairs)))
        # print ("Preference Count Matrix: \n", pd.DataFrame(self.count_matrix))
        # print ("Preference Matrix: \n", pd.DataFrame(preference_matrix_data))
        # print ("-"*30)

    def shuffle_samples(self):
        for i in range(self.num_arms):
            for j in range(self.num_arms):
                np.random.shuffle(self.samples[(i, j)])

    def count_equal_samples(self):
        counts = np.zeros([self.num_arms, self.num_arms], dtype=np.float32)
        for i in range(self.num_arms):
            for j in range(self.num_arms):
                counts[i, j] = (
                    np.sum(np.array(self.samples[(i, j)]) == 0.5)
                    * 100.0
                    / len(self.samples[(i, j)])
                )
        print("Equal Samples: ", counts.astype(np.int32))

    def get_stats(self):
        stats = self.preference_matrix.get_stats()
        stats["Feedback"] = " ".join(self.filename.split("/")[-1].split(".")[:-1])
        stats["Num of systems"] = self.num_arms
        stats["Num of system pairs"] = self.num_pairs
        stats["Total num of samples"] = self.total_samples
        stats["Average num of samples per pair"] = self.total_samples / self.num_pairs
        return stats

    def get_duel(self, arm_i_index: int, arm_j_index: int):
        return arm_i_index, arm_j_index, self.duel(arm_i_index, arm_j_index)

    def duel(self, arm_i_index: int, arm_j_index: int) -> float:
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
            return 0.5

        random_idx = np.random.randint(0, num_samples, size=[1])[0]
        sample = self.samples[arm_i_index, arm_j_index][random_idx]

        if self.model.get_duel(sample):
            return self.model.duel(sample)
        else:
            self.sample_complexity += 1
            return sample["label"]

    def get_num_duels(self) -> int:
        """Get the number of duels that were already performed.

        Returns
        -------
        int
            The number of duels.
        """
        return len(self.history)

    def reset_history(self) -> None:
        """Delete the regret history."""
        self.history.clear()

    def _calculate_regret(
        self,
        best_arm: int,
        aggregation_function: Callable[[float, float], float],
        history: List[Tuple[int, int]],
    ) -> Tuple[list, float]:
        """Calculate the regret.

        The regret is calculated with respect to the given arm. The regret type is specified by the aggregation
        function.

        Parameters
        ----------
        best_arm
            The arm on which the regret is based on.
        aggregation_function
            This function is used to calculate the regret. E.g. minimum for weak regret.

        Returns
        -------
        regret_history
            A list containing the regret per round.
        cumulative_regret
            The cumulative regret.
        """
        regret_history = []
        cumulative_regret = 0.0
        for arm_i, arm_j in history:
            regret = (
                aggregation_function(
                    self.preference_matrix[best_arm, arm_i],
                    self.preference_matrix[best_arm, arm_j],
                )
                - 0.5
            )
            regret_history.append(regret)
            cumulative_regret += regret
        return regret_history, cumulative_regret

    def calculate_weak_regret(
        self, best_arm: int, history: List[Tuple[int, int]]
    ) -> Tuple[List[float], float]:
        """Calculate the weak regret with respect to an arm.

        The weak regret is defined as the distance from the best chosen arm to the best arm overall.

        Parameters
        ----------
        best_arm
            The arm with respect to which the regret is calculated

        Returns
        -------
        regret_history
            A list containing the weak regret per round.
        cumulative_regret
            The cumulative weak regret.
        """
        return self._calculate_regret(best_arm, min, history)

    def calculate_strong_regret(
        self, best_arm: int, history: List[Tuple[int, int]]
    ) -> Tuple[List[float], float]:
        """Calculate the strong regret with respect to an arm.

        The strong regret is defined as the distance from the worst chosen arm to the best arm overall.

        Parameters
        ----------
        best_arm
            The arm with respect to which the regret is calculated

        Returns
        -------
        regret_history
            A list containing the strong regret per round.
        cumulative_regret
            The cumulative strong regret.
        """
        return self._calculate_regret(best_arm, max, history)

    def calculate_average_regret(
        self, best_arm: int, history: List[Tuple[int, int]]
    ) -> Tuple[List[float], float]:
        """Calculate the average regret with respect to an arm.

        The average regret is defined as the average of the distances from the chosen arms to the best arm overall.

        Parameters
        ----------
        best_arm
            The arm with respect to which the regret is calculated.

        Returns
        -------
        regret_history
            A list containing the average regret per round.
        cumulative_regret
            The cumulative average regret.
        """

        def average(
            preference_probability_against_arm1: float,
            preference_probability_against_arm2: float,
        ) -> float:
            return (
                preference_probability_against_arm1
                + preference_probability_against_arm2
            ) / 2

        return self._calculate_regret(best_arm, average, history)

    def calculate_average_copeland_regret(
        self, history: List[Tuple[int, int]]
    ) -> Tuple[List[float], float]:
        """Calculate copeland regret with respect to normalized copeland score.

        The average Copeland regret of a single comparison is the difference between the average normalized Copeland score of
        the pulled arms and the maximum normalized Copeland score. It can only be 0 if a Copeland winner is compared against
        another Copeland winner. Copeland score is normalized by the number of Arms(i.e number_of_arms-1). Finally, This function
        calculates the normalized cumulative Copeland regret accumulated over all time steps.

        Returns
        -------
        regret_history
            A list containing the Copeland regret per round.
        cumulative_regret
            The cumulative average regret.
        """
        regret_history = []
        cumulative_regret = 0.0
        normalized_copeland_scores = (
            self.preference_matrix.get_normalized_copeland_scores()
        )
        max_normalized_copeland_score = np.amax(normalized_copeland_scores)
        for arm_i, arm_j in history:
            regret = max_normalized_copeland_score - 0.5 * (
                normalized_copeland_scores[arm_i] + normalized_copeland_scores[arm_j]
            )
            regret_history.append(regret)
            cumulative_regret += regret
        return regret_history, cumulative_regret


if __name__ == "__main__":

    # feedback_config = json.load(open("configs/nlg/master_nlg_feedback.json", "r"))
    feedback_config = json.load(open("configs/openai_tldr/feedback.json", "r"))

    df = pd.DataFrame([])

    for idx, (name, feed_dict) in enumerate(feedback_config.items()):
        filename = feed_dict["filename"]
        print(filename)
        model = Default()
        nlg = NLGFeedback(model, filename)
        preference_stats = nlg.preference_matrix.get_stats()
        # print ("preference matrix: ", nlg.preference_matrix)

        print("Total Num. of Arms: ", nlg.num_arms)
        copland_scores = nlg.preference_matrix.get_copeland_scores().tolist()
        for i in range(len(copland_scores)):
            print("Arm: ", nlg.idx_to_arm[i], "Copland Score: ", copland_scores[i])
        print("--" * 30)

        stats = {}
        stats["Dataset"] = os.path.split(filename)[1].replace("_processed.pkl", "")
        stats["Num systems"] = nlg.num_arms
        stats["Num system pairs"] = nlg.num_pairs
        stats["Total samples"] = nlg.total_samples
        stats.update(preference_stats)
        stats = pd.DataFrame(stats, index=[idx])
        df = pd.concat([df, stats], axis=0)

    df.to_csv("assumptions_tldr.csv")
