import numpy as np
import pandas as pd
import json
import os
import argparse
from joblib import delayed
from joblib import Parallel
import time
import pickle
import logging
import inspect
import os

from metrics import algorithms
from pairwise_models import direct_to_pairwise
from utils import update_processed


def read_dataset(filename, multiref=False):

    logging.info("Evaluating {} ...".format(filename))
    df = pd.read_csv(filename)
    hypothesis1 = df["system1Text"].tolist()
    hypothesis2 = df["system2Text"].tolist()
    if multiref:
        reference = df["referenceText"].apply(eval).tolist()
    else:
        reference = df["referenceText"].tolist()
    label = df["label"].tolist()

    data = {
        "hypothesis1": hypothesis1,
        "hypothesis2": hypothesis2,
        "reference": reference,
        "label": label,
    }
    return data


def run_experiments(data, metrics_parameters, filename, ensemble=False):

    metrics_class = algorithms[metrics_parameters["name"]]
    metrics_parameters_to_pass = dict()
    for parameter in metrics_parameters.keys():
        if parameter in inspect.getfullargspec(metrics_class.__init__)[0]:
            metrics_parameters_to_pass[parameter] = metrics_parameters[parameter]

    metrics = metrics_class(**metrics_parameters_to_pass)

    scoring_fn = metrics.score
    scoring_fn_cls = metrics_class.score
    if ensemble:
        scoring_fn = metrics.score_ensemble
        scoring_fn_cls = metrics_class.score_ensemble

    score_parameters_to_pass = {
        "hypothesis": data["hypothesis1"],
        "references": data["reference"],
    }
    if "tfrecord_name" in inspect.getfullargspec(scoring_fn_cls)[0]:
        score_parameters_to_pass["tfrecord_name"] = os.path.split(filename)[1].replace(
            ".csv", "1.tf_record"
        )
    score1, metric_names = scoring_fn(**score_parameters_to_pass)

    score_parameters_to_pass["hypothesis"] = data["hypothesis2"]
    if "tfrecord_name" in inspect.getfullargspec(scoring_fn_cls)[0]:
        score_parameters_to_pass["tfrecord_name"] = os.path.split(filename)[1].replace(
            ".csv", "2.tf_record"
        )
    score2, _ = scoring_fn(**score_parameters_to_pass)
    return {
        "score1": np.array(score1.tolist()),
        "score2": np.array(score2.tolist()),
        "metric names": metric_names,
        "label": np.array(data["label"]),
    }


def main(args):
    metrics_param_dict = json.load(open(args.metrics_config, "r"))
    start_time = time.time()
    filenames = [args.val_path, args.test_path]
    splits = ["val", "test"]

    print("Running Automatic Evaluation Metrics ...")
    data = {}
    for idx, split in enumerate(splits):
        score1, score2, metric_names = [], [], []
        label = None
        data[split] = {}

        for metric_name, metrics_param in metrics_param_dict.items():

            filename = filenames[idx]
            input_data = read_dataset(filename, multiref=args.multiref)
            d = run_experiments(
                input_data, metrics_param, filename, ensemble=args.ensemble
            )
            score1.append(d["score1"])
            score2.append(d["score2"])
            metric_names.extend(d["metric names"])
            if label is None:
                label = d["label"]
            else:
                assert np.all(label == d["label"])
            print("Completed running {} for file: {}".format(metric_name, filename))

        data[split]["score1"] = np.concatenate(score1, axis=1)
        data[split]["score2"] = np.concatenate(score2, axis=1)
        data[split]["metric_names"] = metric_names
        data[split]["label"] = label
        print("Completed running all automatic metrics for file: {}".format(filename))

    print("Converting Direct Assesment Scores to Pairwise Probabilities ..")

    results, predictions, metric_names = direct_to_pairwise(data, args.test_path)

    if not os.path.exists(os.path.split(args.output_results)[0]):
        os.makedirs(os.path.split(args.output_results)[0])
    results.to_csv(args.output_results)
    print("Written results to {}".format(args.output_results))

    update_processed(args.processed_dir, predictions, metric_names)

    runtime = time.time() - start_time
    print(f"Script took {round(runtime)}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Direct Assessment Metrics and Convert to Pairwise Probabilities"
    )
    parser.add_argument("--metrics-config", type=str, required=True)
    parser.add_argument(
        "--val-path", type=str, required=True, help="path to the validation dataset"
    )
    parser.add_argument(
        "--test-path", type=str, required=True, help="path to the test dataset"
    )
    parser.add_argument(
        "--processed-dir", type=str, required=True, help="path to processed .pkl files"
    )
    parser.add_argument(
        "--output-results", type=str, required=True, help="path to output the results"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="whether to use ensembling by applying dropout during inference (default: false)",
    )
    parser.add_argument(
        "--multiref",
        action="store_true",
        help="whether the test and val datasets contain mulitple references (default: false)",
    )
    args = parser.parse_args()
    main(args)
