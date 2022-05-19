import numpy as np

from vizseq.scorers.bert_score import BERTScoreScorer
from vizseq.scorers.chrf import ChrFScorer
from vizseq.scorers.wer import WERScorer
from vizseq.scorers.laser import LaserScorer


class VizseqScorer:
    def __init__(self, nthreads=1):
        self.nthreads = nthreads
        self.vizseq_merics = [
            ("bert_score", BERTScoreScorer),
            ("chrf", ChrFScorer),
            ("wer", WERScorer),
            ("laser", LaserScorer),
        ]
        self.metrics_dict = [
            (
                metric_name,
                metric(
                    corpus_level=False,
                    sent_level=True,
                    n_workers=self.nthreads,
                    verbose=False,
                    extra_args=None,
                ),
            )
            for metric_name, metric in self.vizseq_merics
        ]

    def score(self, hypothesis, references):

        if isinstance(references[0], list):
            references = list(zip(*references))
        else:
            references = [references]

        scores = []
        metric_names = []
        for metric_name, metric in self.metrics_dict:
            score = metric.score(hypothesis, references).sent_scores
            scores.append(score)
            metric_names.append(metric_name)
        scores = np.transpose(np.array(scores))
        return scores, metric_names
