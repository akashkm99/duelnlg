from bleurt.score_tpu import BleurtScorerTPU
import numpy as np
import os


class BleurtScorer:
    def __init__(
        self,
        checkpoint,
        output_dir,
        batch_size,
        dropout_during_inference=False,
        metirc_name="bleurt_base",
        use_tpu=True,
        tpu_name=None,
        tpu_zone=None,
        num_tpu_cores=None,
        num_ensemble=1,
    ):
        self.scorer = BleurtScorerTPU(
            checkpoint=checkpoint,
            dropout_during_inference=dropout_during_inference,
            output_dir=output_dir,
            batch_size=batch_size,
            use_tpu=use_tpu,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            num_tpu_cores=num_tpu_cores,
        )
        self.metirc_name = metirc_name
        self.output_dir = output_dir
        self.num_ensemble = num_ensemble
        if (not use_tpu) and (not os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)

    def score(self, hypothesis, references, tfrecord_name=None):

        tfrecord_file = None
        if tfrecord_name is not None:
            tfrecord_file = os.path.join(self.output_dir, tfrecord_name)

        if isinstance(references[0], list):
            references = list(zip(*references))
        else:
            references = [references]
        score_list = []

        num_references = len(references)
        new_hypothesis = hypothesis * num_references
        new_references = [sent for ref in references for sent in ref]

        assert len(new_hypothesis) == len(new_references)

        scores = self.scorer.score(
            new_references, new_hypothesis, tfrecord_file=tfrecord_file
        )
        score_list = np.array_split(scores, num_references)
        scores = np.mean(np.stack(score_list, axis=1), axis=1).reshape(-1, 1)
        return scores, [self.metirc_name]

    def score_ensemble(self, hypothesis, references, tfrecord_name=None):

        tfrecord_file = None
        if tfrecord_name is not None:
            tfrecord_file = os.path.join(self.output_dir, tfrecord_name)

        if isinstance(references[0], list):
            references = list(zip(*references))
        else:
            references = [references]
        score_list = []

        num_references = len(references)
        new_hypothesis = hypothesis * num_references
        new_references = [sent for ref in references for sent in ref]

        assert len(new_hypothesis) == len(new_references)

        ensemble_scores = []
        metric_names = []
        for i in range(self.num_ensemble):
            scores = self.scorer.score(
                new_references, new_hypothesis, tfrecord_file=tfrecord_file
            )
            score_list = np.array_split(scores, num_references)
            scores = np.mean(np.stack(score_list, axis=1), axis=1)
            ensemble_scores.append(scores)
            metric_names.append(self.metirc_name + "_{}".format(i))
        ensemble_scores = np.stack(ensemble_scores, axis=1)

        return ensemble_scores, metric_names
