from duelnlg.duelpy.models.default import Default
from duelnlg.duelpy.models.mixing import Mixing
from duelnlg.duelpy.models.uncertainity import Uncertainity
from duelnlg.duelpy.models.ucb_elimination import UCBElimination
from duelnlg.duelpy.models.uncertainity_ucb_elimination import (
    Uncertainity_UCBElimination,
)

model_list = [
    Default,
    Mixing,
    Uncertainity,
    UCBElimination,
    Uncertainity_UCBElimination,
]
