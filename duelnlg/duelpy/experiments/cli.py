"""Run ad-hoc experiments on the command line.

Run ``python3 -m duelpy.experiments.cli --help`` from the root of the
repository to get started.
"""

import argparse
import inspect
import time
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

from duelnlg.duelpy.algorithms import Algorithm
from duelnlg.duelpy.algorithms import regret_minimizing_algorithms
from duelnlg.duelpy.experiments.environments import environment_list
from duelnlg.duelpy.feedback import MatrixFeedback
from duelnlg.duelpy.stats.metrics import AverageRegret
from duelnlg.duelpy.stats.metrics import BestArmRate
from duelnlg.duelpy.stats.metrics import Cumulative
from duelnlg.duelpy.stats.metrics import ExponentialMovingAverage
from duelnlg.duelpy.stats.metrics import Metric
from duelnlg.duelpy.stats.metrics import TotalWallClock
from duelnlg.duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism


def run_single_algorithm(
    task_random_state: np.random.RandomState,
    num_arms: int,
    algorithm_class: Type[Algorithm],
    environment_class: Type[MatrixFeedback],
    parameters: Dict,
    sample_interval: int,
    run_id: int,
) -> pd.DataFrame:
    """Execute one algorithm for one problem setting and return the results."""
    environment_parameters = {
        "num_arms": num_arms,
        "random_state": task_random_state,
    }
    # Remove parameters that the environment does not expect. For example a
    # deterministic environment might not take a random state.
    environment_parameters = {
        k: v
        for (k, v) in environment_parameters.items()
        if k in inspect.getfullargspec(environment_class.__init__)[0]
    }
    # This is a bit of a hack, since "MatrixFeedback"'s interface doesn't cover
    # the constructor. We cannot be sure what parameters the environment class
    # actually expects. We have to take care that all our environments adhere
    # to this constructor convention.
    feedback_mechanism = environment_class(**environment_parameters)
    metrics: Dict[str, Metric] = {
        "wall_clock": TotalWallClock(),
        "cum_average_regret": Cumulative(
            AverageRegret(feedback_mechanism.preference_matrix)
        ),
        "best_arm_rate (EMA)": ExponentialMovingAverage(
            BestArmRate(feedback_mechanism.preference_matrix.get_condorcet_winner()),
            alpha=0.01,
        ),
    }
    wrapped_feedback = MetricKeepingFeedbackMechanism(
        feedback_mechanism, metrics=metrics, sample_interval=sample_interval
    )  # type: ignore
    # Filter accepted parameters.
    parameters["random_state"] = task_random_state
    parameters_to_pass = dict()
    for parameter in parameters.keys():
        if parameter in inspect.getfullargspec(algorithm_class.__init__)[0]:
            parameters_to_pass[parameter] = parameters[parameter]
    algorithm_class(wrapped_feedback, **parameters_to_pass).run()
    data_frame = pd.DataFrame(wrapped_feedback.results)
    data_frame["algorithm"] = algorithm_class.__name__
    data_frame["run_id"] = run_id
    return data_frame


# This function will be replaced in order to make the experiments more
# flexible, so its not worth refactoring this right now.
# pylint: disable=too-many-arguments
def run_experiment(
    algorithms: List[Type[Algorithm]],
    environment_class: Type[MatrixFeedback],
    time_horizon: int,
    sample_interval: int,
    num_arms: int,
    runs: int,
    n_jobs: int,
    base_random_seed: int,
) -> pd.DataFrame:
    """Run the experiment.

    Parameters
    ----------
    algorithms
        A list of algorithms to run.
    time_horizon
        For how long each algorithm should be run.
    num_arms
        How many arms to include in the generated problems.
    runs
        How often each algorithm should be run.
    n_jobs
        How many jobs to execute in parallel.
    base_random_seed
        Used to derive random states for each experiment.

    Returns
    -------
    DataFrame
        The experiment results.
    """
    start_time = time.time()
    failure_probability = (
        1 / time_horizon
    )  # for PAC algorithms, to give them some time for exploitation as well

    parameters = {
        "time_horizon": time_horizon,
        "failure_probability": failure_probability,
    }

    def job_producer() -> Generator:
        for algorithm_class in algorithms:
            algorithm_name = algorithm_class.__name__
            for run_id in range(runs):
                random_state = np.random.RandomState(
                    (base_random_seed + hash(algorithm_name) + run_id) % 2 ** 32
                )
                yield delayed(run_single_algorithm)(
                    random_state,
                    num_arms,
                    algorithm_class,
                    environment_class,
                    parameters,
                    sample_interval,
                    run_id,
                )

    jobs = list(job_producer())
    result = Parallel(n_jobs=n_jobs, verbose=10)(jobs)
    runtime = time.time() - start_time
    print(f"Experiments took {round(runtime)}s.")
    return pd.concat(result)


def plot_results(data: pd.DataFrame) -> None:
    """Plot experiment results using seaborn.

    Parameters
    ----------
    data
        A pandas dataframe with columns time_step, cum_average_regret and
        wall_clock.
    """
    sns.set()
    _fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    sns.lineplot(
        data=data,
        x="time_step",
        y="cum_average_regret",
        hue="algorithm",
        style="algorithm",
        ci=None,
        linewidth=2,
        ax=ax1,
    )
    sns.lineplot(
        data=data,
        x="time_step",
        y="best_arm_rate (EMA)",
        hue="algorithm",
        style="algorithm",
        ci=None,
        linewidth=2,
        ax=ax2,
    )
    sns.lineplot(
        data=data,
        x="time_step",
        y="wall_clock",
        hue="algorithm",
        style="algorithm",
        ci=None,
        linewidth=2,
        ax=ax3,
    )
    plt.show()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PB-MAB experiments and plot results."
    )
    algorithm_names_to_algorithms = {
        algorithm.__name__: algorithm for algorithm in regret_minimizing_algorithms
    }
    environment_names_to_environments = {
        environment.__name__: environment for environment in environment_list
    }
    algorithm_choices_string = " ".join(algorithm_names_to_algorithms.keys())
    parser.add_argument(
        "-a, --algorithms",
        nargs="+",
        metavar="ClassName",
        dest="algorithms",
        default=algorithm_names_to_algorithms.keys(),
        help=f"Algorithms to compare. (default: {algorithm_choices_string})",
        choices=algorithm_names_to_algorithms.keys(),
    )
    parser.add_argument(
        "--environment",
        dest="environment",
        default="HardCondorcetMatrix",
        help="Algorithms to compare. (default: HardCondorcetMatrix)",
        choices=environment_names_to_environments.keys(),
    )
    parser.add_argument(
        "--runs",
        dest="runs",
        default=10,  # savage paper uses 1000
        type=int,
        help="How often to run each algorithm. Results will be averaged. (default: 10)",
    )
    parser.add_argument(
        "--arms",
        dest="num_arms",
        default=20,  # savage paper uses 100, but our current savage implementation does not scale well
        type=int,
        help="How many arms the generated preference matrices should contain. (default: 20)",
    )
    parser.add_argument(
        "--time-horizon",
        dest="time_horizon",
        default=int(1e4),  # savage paper uses 1e6
        type=int,
        help="For how many time steps to run each algorithm. (default: 1e4)",
    )
    parser.add_argument(
        "--sample-interval",
        dest="sample_interval",
        default=1,
        type=int,
        help="The number of time steps per sample.",
    )
    parser.add_argument(
        "--random-seed",
        dest="base_random_seed",
        default=42,
        type=int,
        help="Base random seed for reproducible results.",
    )
    parser.add_argument(
        "--jobs",
        dest="n_jobs",
        default=-1,
        type=int,
        help="How many experiments to run in parallel. The special value -1 stands for the number of processor cores.",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=False,
        help="Enable profiling mode. This disables parallelism and plotting. Intended for use with cProfile.",
    )

    args = parser.parse_args()
    algorithms = [
        algorithm_names_to_algorithms[algorithm] for algorithm in args.algorithms
    ]
    environment = environment_names_to_environments[args.environment]

    results = run_experiment(
        algorithms=algorithms,
        environment_class=environment,
        n_jobs=1 if args.profile else args.n_jobs,
        time_horizon=args.time_horizon,
        sample_interval=args.sample_interval,
        num_arms=args.num_arms,
        runs=args.runs,
        base_random_seed=args.base_random_seed,
    )
    if args.profile:
        last_sample = (args.time_horizon // args.sample_interval) * args.sample_interval
        final_results = results[results["time_step"] == last_sample]
        averaged_times = (
            final_results[["algorithm", "wall_clock"]].groupby(["algorithm"]).mean()
        )
        print(averaged_times)
    else:
        plot_results(results)


if __name__ == "__main__":
    _main()
