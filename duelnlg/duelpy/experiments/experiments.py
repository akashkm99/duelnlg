""" Experiments of various Online Pairwise NLG Evaluation. 

Implementation of WMT experiments in a pure exploration setup. 
"""

import argparse
import inspect
import time
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Type

from joblib import delayed
from joblib import Parallel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import datetime
import pickle
import json
from glob import glob
import logging
from duelnlg.duelpy.algorithms import Algorithm
from duelnlg.duelpy.algorithms import algorithm_list
from duelnlg.duelpy.models import model_list
from duelnlg.duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelnlg.duelpy.experiments.environments import HardCondorcetMatrix
from duelnlg.duelpy.feedback.nlg_feedback_hyperparam import NLGFeedbackHyperParam
from duelnlg.duelpy.feedback.nlg_feedback import NLGFeedback
from duelnlg.duelpy.feedback.nlg_feedback_delay import NLGFeedbackDelay

from duelnlg.duelpy.models import Default
from pandas.api.types import is_numeric_dtype

feedback_dict = {
    "nlg": NLGFeedback,
    "nlg_hyperparam_expt": NLGFeedbackHyperParam,
    "nlg_delay": NLGFeedbackDelay,
}
algorithms_dict = {algorithm.__name__: algorithm for algorithm in algorithm_list}
model_dict = {model.__name__: model for model in model_list}


def mean_str(col):
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique() if col.nunique() == 1 else np.NaN


def run_single_algorithm(
    task_random_seed: int,
    feedback_name: Type[FeedbackMechanism],
    feedback_parameters: Dict,
    algorithm_name: Type[Algorithm],
    algorithm_parameters: Dict,
    model_name,
    model_parameters,
    run_id: int,
) -> pd.DataFrame:
    """Execute one algorithm for one problem setting and return the results."""

    print(
        "Executing Algorithm: {}, Model: {}, Feedback: {}, Run Id: {}".format(
            algorithm_name, model_name, feedback_name, run_id
        )
    )

    feedback = feedback_parameters["feedback"]
    feedback_class = feedback_dict[feedback]

    num_steps = feedback_parameters["num_steps"]

    algorithm = algorithm_parameters["algorithm"]
    algorithm_class = algorithms_dict[algorithm]
    pure_exploration = algorithm_parameters["pure_exploration"]
    model = model_parameters["model"]
    model_class = model_dict[model]

    model_parameters["random_state"] = np.random.RandomState(task_random_seed)
    model_parameters_to_pass = dict()
    for parameter in model_parameters.keys():
        if parameter in inspect.getfullargspec(model_class.__init__)[0]:
            model_parameters_to_pass[parameter] = model_parameters[parameter]
    feedback_parameters_to_pass = dict()
    for parameter in feedback_parameters.keys():
        if parameter in inspect.getfullargspec(feedback_class.__init__)[0]:
            feedback_parameters_to_pass[parameter] = feedback_parameters[parameter]
    feedback_parameters_to_pass["random_seed"] = task_random_seed

    algorithm_parameters["random_state"] = np.random.RandomState(task_random_seed)

    if "time_horizon" not in algorithm_parameters:
        algorithm_parameters["time_horizon"] = num_steps
    algorithm_parameters_to_pass = dict()
    for parameter in algorithm_parameters.keys():
        if parameter in inspect.getfullargspec(algorithm_class.__init__)[0]:
            algorithm_parameters_to_pass[parameter] = algorithm_parameters[parameter]

    model = model_class(**model_parameters_to_pass)
    feedback_mechanism = feedback_class(model, **feedback_parameters_to_pass)
    algorithm = algorithm_class(feedback_mechanism, **algorithm_parameters_to_pass)

    if model_parameters["model"] != "Default":
        algorithm_name = algorithm_name + "_" + model_name

    condorcet_winner = feedback_mechanism.preference_matrix.get_condorcet_winner()
    # print ("condorcet_winner: ", condorcet_winner)
    normalized_copeland_scores = (
        feedback_mechanism.preference_matrix.get_normalized_copeland_scores()
    )
    max_normalized_copeland_score = np.amax(normalized_copeland_scores)
    copland_winners = np.where(
        normalized_copeland_scores == max_normalized_copeland_score
    )[0].tolist()
    # print ("max_normalized_copeland_score: ", max_normalized_copeland_score)

    data: Dict[str, Any] = {}
    time_elapsed = []
    perstep_regret_history = []
    copland_perstep_regret_history = []
    prediction_accuracy = []
    sample_complexity = []

    start_time = time.time()
    steps = 0
    while ((not pure_exploration) and steps < num_steps) or (
        pure_exploration and not algorithm.is_finished()
    ):
        algorithm.step()
        predicted_winner = algorithm.get_winner()
        perstep_regret = (
            feedback_mechanism.preference_matrix[condorcet_winner, predicted_winner]
            - 0.5
            if condorcet_winner is not None
            else None
        )
        copland_perstep_regret = (
            max_normalized_copeland_score - normalized_copeland_scores[predicted_winner]
        )
        accuracy = int(predicted_winner in copland_winners)

        if not pure_exploration:
            time_elapsed.append(time.time() - start_time)
            perstep_regret_history.append(perstep_regret)
            copland_perstep_regret_history.append(copland_perstep_regret)
            prediction_accuracy.append(accuracy)
            sample_complexity.append(feedback_mechanism.sample_complexity)
            steps += 1

    if pure_exploration:
        time_elapsed.append(time.time() - start_time)
        perstep_regret_history.append(perstep_regret)
        copland_perstep_regret_history.append(copland_perstep_regret)
        prediction_accuracy.append(accuracy)
        sample_complexity.append(feedback_mechanism.sample_complexity)
        steps += 1

    num_steps = steps
    data["run_id"] = [run_id] * (num_steps)
    data["feedback_name"] = [feedback_name] * (num_steps)
    data["algorithm_name"] = [algorithm_name] * (num_steps)
    data["wall_clock"] = time_elapsed
    data["time_step"] = list(range(1, num_steps + 1))
    data["perstep_regret"] = perstep_regret_history
    data["copland_perstep_regret"] = copland_perstep_regret_history
    data["prediction_accuracy"] = prediction_accuracy
    data["sample_complexity"] = sample_complexity

    print(
        "Done executing Algorithm: {}, Model: {}, Feedback: {}, Run Id: {}".format(
            algorithm_name, model_name, feedback_name, run_id
        )
    )

    return pd.DataFrame(data)


def run_experiment(
    feedback_param_dict,
    algorithm_param_dict,
    model_param_dict,
    runs,
    base_random_seed,
    output_dir,
):

    """Run the experiment."""

    start_time = time.time()

    def job_producer() -> Generator:

        for feedback_name, feedback_parameters in feedback_param_dict.items():
            for algorithm_name, algorithm_parameters in algorithm_param_dict.items():
                for model_name, model_parameters in model_param_dict.items():
                    for run_id in range(runs):
                        random_seed = base_random_seed + run_id
                        yield delayed(run_single_algorithm)(
                            task_random_seed=random_seed,
                            feedback_name=feedback_name,
                            feedback_parameters=feedback_parameters,
                            algorithm_name=algorithm_name,
                            algorithm_parameters=algorithm_parameters,
                            model_name=model_name,
                            model_parameters=model_parameters,
                            run_id=run_id,
                        )

    jobs = list(job_producer())
    results = Parallel(n_jobs=-1, verbose=10)(jobs)
    runtime = time.time() - start_time
    print(f"Experiments took {round(runtime)}s.")

    final_results = {}
    for result in results:
        algorithm_name = result["algorithm_name"][0]
        feedback_name = result["feedback_name"][0]
        run_id = result["run_id"][0]
        base_dir = os.path.join(
            output_dir, "{}/{}".format(feedback_name, algorithm_name)
        )

        if base_dir not in final_results:
            final_results[base_dir] = {}
        final_results[base_dir][run_id] = result

    for base_dir, save_dict in final_results.items():

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        results_filename = os.path.join(base_dir, "results.pkl")
        with open(results_filename, "wb") as fp:
            pickle.dump(save_dict, fp)


def aggregate(data, group="sample_complexity"):
    data_mean = data.groupby(group).mean().reset_index()
    data_mean["algorithm_name"] = data["algorithm_name"].iloc[0]
    data_mean["feedback_name"] = data["feedback_name"].iloc[0]
    data_mean["run_id"] = data["run_id"].iloc[-1]
    return data_mean


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def min_sample_complexity(condition, sample_complexity=None):

    condition = np.cumprod(condition[::-1])[::-1]
    if len(np.where(condition)[0]) > 0:
        time_step = np.where(condition)[0][0].item()
    else:
        return None

    if sample_complexity is not None:
        return sample_complexity[time_step]
    else:
        return time_step


def summary(data, pure_exploration=False, eps=0.001, eps_copland=0.005, delta_acc=0.05):

    summary = {}
    by = "time_step" if pure_exploration else "sample_complexity"
    data_mean = data.groupby(by).mean().reset_index()
    copland_perstep_regret = data_mean["copland_perstep_regret"].values
    accuracy = data_mean["prediction_accuracy"].values
    perstep_regret = data_mean["perstep_regret"].values

    sample_complexity = None
    if "sample_complexity" in data_mean:
        sample_complexity = data_mean["sample_complexity"].values

    summary["algorithm_name"] = data["algorithm_name"].iloc[0]
    summary["feedback_name"] = data["feedback_name"].iloc[0]
    summary["num_runs"] = data["run_id"].iloc[-1] + 1
    summary["total_steps"] = data["time_step"].iloc[-1]
    # summary["query_complexity_regret"] = min_sample_complexity((perstep_regret < eps), sample_complexity=sample_complexity)
    # summary["annotation_complexity_copland_regret"] = min_sample_complexity((copland_perstep_regret < eps_copland), sample_complexity=sample_complexity)
    summary["annotation_complexity"] = min_sample_complexity(
        (accuracy > 1 - delta_acc), sample_complexity=sample_complexity
    )
    summary["total_time"] = data_mean["wall_clock"].iloc[-1]
    return summary


def plot_results(feedback_names, algorithm_param_dict, model_param_dict, output_dir):

    print("Calculating stats and plotting results")
    print("=" * 30)

    for feedback_name in feedback_names:
        data = pd.DataFrame({})
        for algorithm_name, algorithm_parameters in algorithm_param_dict.items():
            for model_name, model_parameters in model_param_dict.items():
                if model_parameters["model"] != "Default":
                    algorithm_name_file = algorithm_name + "_" + model_name
                else:
                    algorithm_name_file = algorithm_name

                results_filename = os.path.join(
                    output_dir,
                    "{}/{}/results.pkl".format(feedback_name, algorithm_name_file),
                )
                summary_filename = os.path.join(
                    output_dir,
                    "{}/{}/summary.json".format(feedback_name, algorithm_name_file),
                )

                if os.path.exists(results_filename):
                    print("Loading saved data from {}".format(results_filename))
                    with open(results_filename, "rb") as fp:
                        save_dict = pickle.load(fp)
                        data_saved = pd.concat(
                            [value for key, value in save_dict.items()]
                        )
                        data_summary = summary(
                            data_saved,
                            pure_exploration=algorithm_parameters["pure_exploration"],
                        )
                        data = pd.concat([data, data_saved])

                    with open(summary_filename, "w") as g:
                        print(data_summary)
                        json.dump(data_summary, g, cls=NpEncoder)
                        print("Summary saved to {}".format(summary_filename))
                        print("-" * 20)
                else:
                    print("Could not find saved data at {}".format(results_filename))
                    print("Did you run the algorithm?")
                    print("Not found: {}".format(results_filename))

        plot_dir = os.path.join(output_dir, "{}/plots".format(feedback_name))
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        plot_filename = os.path.join(
            plot_dir, "plot_{}.png".format(datetime.datetime.now())
        )
        plot_fn(data, feedback_name, plot_filename)


def plot_fn(data: pd.DataFrame, feedback_name: str, plot_filename: str) -> None:

    """Plot experiment results using seaborn.

    Parameters
    ----------
    data
        A pandas dataframe with columns time_step, cum_average_regret and
        wall_clock.
    """

    data = data.reset_index()
    x_column = "sample_complexity" if "sample_complexity" in data else "time_step"
    sns.set()
    _fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    sns.lineplot(
        data=data,
        x=x_column,
        y="prediction_accuracy",
        hue="algorithm_name",
        estimator="mean",
        ci=None,
        linewidth=2,
        ax=axs[0, 0],
    )
    axs[0, 0].set_xlabel("Number of Human Annotations")
    axs[0, 0].set_ylabel("Top-rank Prediction Accuracy")

    sns.lineplot(
        data=data,
        x=x_column,
        y="perstep_regret",
        hue="algorithm_name",
        estimator="mean",
        ci=None,
        linewidth=2,
        ax=axs[0, 1],
    )

    axs[0, 1].set_xlabel("Number of Human Annotations")
    axs[0, 1].set_ylabel("Perstep Gap Regret")

    sns.lineplot(
        data=data,
        x=x_column,
        y="copland_perstep_regret",
        hue="algorithm_name",
        estimator="mean",
        ci=None,
        linewidth=2,
        ax=axs[1, 0],
    )

    axs[1, 0].set_xlabel("Number of Human Annotations")
    axs[1, 0].set_ylabel("Perstep Copland Regret")

    sns.lineplot(
        data=data,
        x=x_column,
        y="wall_clock",
        hue="algorithm_name",
        estimator="mean",
        ci=None,
        linewidth=2,
        ax=axs[1, 1],
    )

    axs[1, 1].set_xlabel("Number of Human Annotations")
    axs[1, 1].set_ylabel("Wall clock time (in s)")

    plt.title("Dueling Bandits Algorithms in {} enviroinment".format(feedback_name))
    plt.savefig(plot_filename)
    print("Plot saved to {}".format(plot_filename))


def main():
    parser = argparse.ArgumentParser(
        description="Run PB-MAB experiments and plot results."
    )

    parser.add_argument(
        "--feedback-config",
        dest="feedback_config",
        type=str,
        help="Path to the feedback config dictionary (json file)",
    )

    parser.add_argument(
        "--algorithm-config",
        dest="algorithm_config",
        type=str,
        help="Path to the algorithm config dictionary (json file)",
    )

    parser.add_argument(
        "--model-config",
        dest="model_config",
        type=str,
        help="Path to the model config dictionary (json file)",
    )

    parser.add_argument(
        "--num-runs",
        dest="num_runs",
        default=100,  # savage paper uses 1000
        type=int,
        help="How often to run each algorithm. Results will be averaged. (default: 100)",
    )

    parser.add_argument(
        "--random-seed",
        dest="base_random_seed",
        default=42,
        type=int,
        help="Base random seed for reproducible results.",
    )

    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="./results",
        type=str,
        help="Directory to dump the results and plots",
    )

    args = parser.parse_args()
    feedback_param_dict = json.load(open(args.feedback_config, "r"))
    algorithm_param_dict = json.load(open(args.algorithm_config, "r"))

    if args.model_config is not None:
        model_param_dict = json.load(open(args.model_config, "r"))
    else:
        model_param_dict = {"default": {"model": "Default"}}

    run_experiment(
        feedback_param_dict=feedback_param_dict,
        algorithm_param_dict=algorithm_param_dict,
        model_param_dict=model_param_dict,
        runs=args.num_runs,
        base_random_seed=args.base_random_seed,
        output_dir=args.output_dir,
    )

    plot_results(
        feedback_param_dict.keys(),
        algorithm_param_dict,
        model_param_dict,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
