"""Various algorithms to solve Preference-Based Multi-Armed Bandit Problems."""
from duelnlg.duelpy.algorithms.algorithm import Algorithm
from duelnlg.duelpy.algorithms.approximate_probability import ApproximateProbability
from duelnlg.duelpy.algorithms.beat_the_mean import BeatTheMeanBandit
from duelnlg.duelpy.algorithms.beat_the_mean import BeatTheMeanBanditPAC
from duelnlg.duelpy.algorithms.copeland_confidence_bound import CopelandConfidenceBound
from duelnlg.duelpy.algorithms.double_thompson_sampling import DoubleThompsonSampling
from duelnlg.duelpy.algorithms.double_thompson_sampling import (
    DoubleThompsonSamplingPlus,
)
from duelnlg.duelpy.algorithms.interleaved_filtering import InterleavedFiltering
from duelnlg.duelpy.algorithms.kl_divergence_based_pac import KLDivergenceBasedPAC
from duelnlg.duelpy.algorithms.knockout_tournament import KnockoutTournament
from duelnlg.duelpy.algorithms.mallows import MallowsMPI
from duelnlg.duelpy.algorithms.mallows import MallowsMPR
from duelnlg.duelpy.algorithms.merge_rucb import MergeRUCB
from duelnlg.duelpy.algorithms.multisort import Multisort
from duelnlg.duelpy.algorithms.optmax import OptMax
from duelnlg.duelpy.algorithms.plackett_luce import PlackettLuceAMPR
from duelnlg.duelpy.algorithms.plackett_luce import PlackettLucePACItem
from duelnlg.duelpy.algorithms.relative_confidence_sampling import (
    RelativeConfidenceSampling,
)
from duelnlg.duelpy.algorithms.relative_ucb import RelativeUCB
from duelnlg.duelpy.algorithms.rmed_new import RMED1
from duelnlg.duelpy.algorithms.rmed import RelativeMinimumEmpiricalDivergence1
from duelnlg.duelpy.algorithms.rmed import RelativeMinimumEmpiricalDivergence2
from duelnlg.duelpy.algorithms.rmed import RelativeMinimumEmpiricalDivergence2FH
from duelnlg.duelpy.algorithms.savage import Savage
from duelnlg.duelpy.algorithms.scalable_copeland_bandits import ScalableCopelandBandits
from duelnlg.duelpy.algorithms.sequential_elimination import SequentialElimination
from duelnlg.duelpy.algorithms.single_elimination_tournament import (
    SingleEliminationTop1Select,
)
from duelnlg.duelpy.algorithms.single_elimination_tournament import (
    SingleEliminationTopKSorting,
)
from duelnlg.duelpy.algorithms.successive_elimination import SuccessiveElimination
from duelnlg.duelpy.algorithms.winner_stays import WinnerStaysStrongRegret
from duelnlg.duelpy.algorithms.winner_stays import WinnerStaysWeakRegret
from duelnlg.duelpy.algorithms.uniform_exploration import UniformExploration

# Pylint insists that regret_minimizing_algorithms and interfaces are constants and should be
# named in UPPER_CASE. Technically that is correct, but it doesn't feel quite
# right for this use case. Its not a typical constant. A similar use-case would
# be numpy's np.core.numerictypes.allTypes, which is also not names in
# UPPER_CASE.
# pylint: disable=invalid-name

# Make the actual algorithm classes available for easy enumeration in
# experiments and tests.
algorithm_list = [
    UniformExploration,
    Savage,
    WinnerStaysWeakRegret,
    WinnerStaysStrongRegret,
    BeatTheMeanBandit,
    BeatTheMeanBanditPAC,
    RelativeConfidenceSampling,
    RelativeUCB,
    InterleavedFiltering,
    KnockoutTournament,
    CopelandConfidenceBound,
    MallowsMPI,
    MallowsMPR,
    Multisort,
    MergeRUCB,
    SequentialElimination,
    SingleEliminationTopKSorting,
    SingleEliminationTop1Select,
    SuccessiveElimination,
    DoubleThompsonSampling,
    DoubleThompsonSamplingPlus,
    PlackettLucePACItem,
    PlackettLuceAMPR,
    OptMax,
    RelativeMinimumEmpiricalDivergence1,
    RelativeMinimumEmpiricalDivergence2,
    RelativeMinimumEmpiricalDivergence2FH,
    RMED1,
    ScalableCopelandBandits,
    KLDivergenceBasedPAC,
    ApproximateProbability,
]

# This is not really needed, but otherwise zimports doesn't understand the
# __all__ construct and complains that the Algorithm import is unnecessary.
interfaces = [Algorithm]

# Generate __all__ for tab-completion etc.
__all__ = ["Algorithm"] + [algorithm.__name__ for algorithm in algorithm_list]
