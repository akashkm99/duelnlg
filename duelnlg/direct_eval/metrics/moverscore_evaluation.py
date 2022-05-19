import numpy as np
from .moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict
import torch


class MoverScorer:
    def __init__(self, batch_size=64, device="cuda:0"):
        self.batch_size = batch_size
        self.device = device
        if not torch.cuda.is_available():
            self.device = "cpu"

    def score(self, hypothesis, references):

        if isinstance(references[0], list):
            references = list(zip(*references))
        else:
            references = [references]
        score_list = []

        num_references = len(references)
        new_hypothesis = hypothesis * num_references
        new_references = [sent for ref in references for sent in ref]

        assert len(new_hypothesis) == len(new_references)

        idf_dict_hyp = get_idf_dict(
            new_hypothesis
        )  # idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = get_idf_dict(
            new_references
        )  # idf_dict_ref = defaultdict(lambda: 1.)

        scores = word_mover_score(
            new_references,
            new_hypothesis,
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=True,
            batch_size=self.batch_size,
            device=self.device,
        )

        score_list = np.array_split(scores, num_references)
        scores = np.mean(np.stack(score_list, axis=1), axis=1).reshape(-1, 1)
        return scores, ["MoverScore"]
