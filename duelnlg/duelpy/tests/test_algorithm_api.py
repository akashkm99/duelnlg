"""Some general checks about expected algorithm behavior."""
import inspect
from typing import Type

import numpy as np
import pytest

from duelnlg.duelpy.algorithms import Algorithm
from duelnlg.duelpy.algorithms import regret_minimizing_algorithms
from duelnlg.duelpy.experiments.environments import HardCondorcetMatrix


@pytest.mark.parametrize("algorithm_class", list(regret_minimizing_algorithms))
@pytest.mark.parametrize("time_horizon", [1, 2, 5, 10, 50])
def test_time_horizon_kept(algorithm_class: Type[Algorithm], time_horizon: int) -> None:
    """Test that all algorithm keep to the time horizon if given.

    Executes the algorithms with a given time horizon (a small selection is
    tested, to make sure that no algorithm just happens to keep to one value)
    and then checks that the number of duels exactly matches the time horizon.
    """
    random_state = np.random.RandomState(42)
    feedback_mechanism = HardCondorcetMatrix(num_arms=5, random_state=random_state)
    parameters = {
        "time_horizon": time_horizon,
        "feedback_mechanism": feedback_mechanism,
    }
    if "random_state" in inspect.getfullargspec(algorithm_class.__init__)[0]:
        parameters["random_state"] = random_state
    algorithm = algorithm_class(**parameters)  # type: ignore
    algorithm.run()
    assert feedback_mechanism.get_num_duels() == time_horizon
