# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""BLEURT scoring library."""
import itertools
import os
import logging

from bleurt import checkpoint as checkpoint_lib
from bleurt import encoding
from bleurt.lib import modeling
from bleurt.lib import tokenization
from bleurt import model
import tensorflow as tf

tf.get_logger().setLevel("INFO")


def _get_default_checkpoint():
    pkg = os.path.abspath(__file__)
    pkg, _ = os.path.split(pkg)
    ckpt = os.path.join(pkg, "test_checkpoint")
    assert tf.io.gfile.exists(
        ckpt
    ), "Default checkpoint not found! Are you sure the install is complete?"
    return ckpt


# Python API for BLEURT.
class BleurtScorerTPU(object):
    """Class for scoring the BLEURT-similarity between two sentences."""

    def __init__(
        self,
        checkpoint=None,
        dropout_during_inference=False,
        output_dir=None,
        batch_size=64,
        use_tpu=True,
        tpu_name=None,
        tpu_zone=None,
        gcp_project=None,
        num_tpu_cores=8,
    ):
        """Initializes the BLEURT model.

    Args:
      checkpoint: BLEURT checkpoint. Will default to BLEURT-tiny if None.
      predict_fn: (optional) prediction function, overrides chkpt_dir. Mostly
        used for testing.

    Returns:
      A BLEURT scorer export.
    """
        if not checkpoint:
            logging.info("No checkpoint specified, defaulting to BLEURT-tiny.")
            checkpoint = _get_default_checkpoint()

        logging.info("Reading checkpoint {}.".format(checkpoint))
        config = checkpoint_lib.read_bleurt_config(checkpoint)
        max_seq_length = config["max_seq_length"]
        vocab_file = config["vocab_file"]
        do_lower_case = config["do_lower_case"]
        bert_config_file = config["bert_config_file"]

        logging.info("Creating BLEURT scorer.")

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case
        )
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.use_tpu = use_tpu

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        tpu_cluster_resolver = None
        if use_tpu and tpu_name:
            tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu_name, zone=tpu_zone, project=gcp_project
            )

        is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig
        run_config = tf.compat.v1.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=output_dir,
            tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
                num_shards=num_tpu_cores, per_host_input_for_training=is_per_host
            ),
        )

        logging.info("Creating model.")
        model_fn = model.model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=os.path.join(checkpoint, "variables/variables"),
            learning_rate=0,
            num_train_steps=0,
            num_warmup_steps=0,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu,
            n_hidden_layers=0,
            hidden_layers_width=128,
            dropout_rate=0,
            dropout_during_inference=dropout_during_inference,
        )

        self.estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
            use_tpu=use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            predict_batch_size=batch_size,
        )

        logging.info("BLEURT initialized.")

    def score(self, references, candidates, tfrecord_file=None):
        """Scores a collection of references and candidates.

    Args:
      references: a list of strings.
      candidates: a list of strings.
    Returns:
      A list of scores.
    """

        candidates, references = list(candidates), list(references)
        assert len(candidates) == len(references), (
            "The number of candidate sentences must match the number of "
            "reference sentences."
        )

        if not candidates:
            return []

        num_actual_predict_examples = len(references)

        if self.use_tpu:
            while len(references) % self.batch_size != 0:
                references.append("Dummy Input")
                candidates.append("Dummy Input")

        if not tfrecord_file:
            tfrecord_file = os.path.join(self.output_dir, "tmp_predict.tf_record")
            if tf.io.gfile.exists(tfrecord_file):
                logging.info("Deleting existing file: {}".format(tfrecord_file))
                tf.io.gfile.remove(tfrecord_file)
                print("Done.")

        if not tf.io.gfile.exists(tfrecord_file):
            encoding.encode_and_serialize_from_list(
                references,
                candidates,
                tfrecord_file,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
            )

        predict_input_fn = model.input_fn_builder(
            tfrecord_file,
            seq_length=self.max_seq_length,
            is_training=False,
            batch_size=self.batch_size,
            drop_remainder=self.use_tpu,
        )

        result = self.estimator.predict(input_fn=predict_input_fn)
        all_results = []
        num_written_lines = 0
        for (i, prediction) in enumerate(result):
            if i >= num_actual_predict_examples:
                break
            all_results.append(float(prediction["predictions"]))
            num_written_lines += 1

        assert (
            len(all_results) == num_written_lines
        ), "Number of predictions does not match sentences: {} vs. {}".format(
            len(all_results), len(candidates)
        )
        return all_results


if __name__ == "__main__":
    ckpt_path = "gs://akash_bucket/bleurt_checkpoints/bleurt_base"
    scorer = BleurtScorerTPU(
        checkpoint=ckpt_path,
        output_dir="gs://akash_bucket/bleurt_experiments/tmp",
        dropout_during_inference=True,
        batch_size=80,
        use_tpu=True,
        tpu_name="bleurt1",
        tpu_zone="europe-west4-a",
        num_tpu_cores=8,
    )

    import pandas as pd

    df = pd.read_csv("../../../../../data/wmt15-16-data/test.csv").iloc[:80]

    references = df["referenceText"].tolist()
    candidates = df["system1Text"].tolist()
    bleurt_scores = scorer.score(references, candidates)
    print("bleurt_scores1: ", bleurt_scores)
    bleurt_scores = scorer.score(references, candidates)
    print("bleurt_scores2: ", bleurt_scores)
