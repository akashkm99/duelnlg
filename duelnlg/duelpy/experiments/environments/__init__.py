"""Various dueling bandit environments for use in experiments."""

from duelnlg.duelpy.experiments.environments.hard_condorcet_matrix import (
    HardCondorcetMatrix,
)
from duelnlg.duelpy.experiments.environments.mallows_model import MallowsModel
from duelnlg.duelpy.experiments.environments.plackett_luce_model import (
    PlackettLuceModel,
)

# Pylint insists that environment_list is a constant and should be named in
# UPPER_CASE. See duelpy/algorithms/__init__.py for an explanation of why we
# disable the check here.
# pylint: disable=invalid-name

# Make the environment classes available for easy enumeration in experiments
# and tests.
environment_list = [HardCondorcetMatrix, PlackettLuceModel, MallowsModel]

__all__ = [environment.__name__ for environment in environment_list]
