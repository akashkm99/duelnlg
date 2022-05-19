from .nlgeval_evaluation import NlgevalScorer
from .vizseq_evaluation import VizseqScorer
from .bleurt_evaluation import BleurtScorer
from .moverscore_evaluation import MoverScorer

algorithm_names = [NlgevalScorer, VizseqScorer, BleurtScorer, MoverScorer]
algorithms = {
    "nlgeval": NlgevalScorer,
    "vizseq": VizseqScorer,
    "bleurt": BleurtScorer,
    "moverscore": MoverScorer,
}
