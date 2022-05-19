from nlgeval import NLGEval
from multiprocessing import Pool
import numpy as np

nlgeval = NLGEval(metrics_to_omit=["SkipThoughtCS", "METEOR"])
nlgeval_metrics = [
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "ROUGE_L",
    "CIDEr",
    "EmbeddingAverageCosineSimilarity",
    "VectorExtremaCosineSimilarity",
    "GreedyMatchingScore",
]


def func(data):
    hypothesis, reference = data
    if not isinstance(reference, list):
        reference = [reference]

    result = nlgeval.compute_individual_metrics(reference, hypothesis)
    return [result[metric] for metric in nlgeval_metrics]


class NlgevalScorer:
    def __init__(self, nthreads=1):
        self.nthreads = nthreads

    def score(self, hypothesis, references):

        p = Pool(self.nthreads)
        score = p.map(func, list(zip(hypothesis, references)))
        score = np.array(score)
        return score, nlgeval_metrics
