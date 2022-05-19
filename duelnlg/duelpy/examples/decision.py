"""A simple example of how a decision-tool using duelpy could work."""

from duelnlg.duelpy.algorithms.savage import Savage
from duelnlg.duelpy.feedback import CommandlineFeedback


def _run_decision_experiment() -> None:
    arms = [
        "The best arm.",  # 0
        "The second best arm.",  # 1
        "The other second best arm, flip a coin.",  # 2
        "Third best arm.",  # 3
        "Least favorite arm.",  # 4
    ]
    feedback_mechanism = CommandlineFeedback(arms)
    algorithm = Savage(feedback_mechanism=feedback_mechanism, failure_probability=0.5)
    while not algorithm.is_finished():
        algorithm.step()
        print("Preference estimate is now")
        print(algorithm.preference_estimate)
    print("Estimated Copeland winner:")
    print(algorithm.get_copeland_winner())


if __name__ == "__main__":
    _run_decision_experiment()
