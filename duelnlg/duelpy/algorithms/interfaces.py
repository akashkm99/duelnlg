"""Standardized interfaces for different kinds of Algorithms."""

from typing import Collection
from typing import List
from typing import Optional

from duelnlg.duelpy.algorithms.algorithm import Algorithm
from duelnlg.duelpy.stats.preference_matrix import PreferenceMatrix


class PacAlgorithm(Algorithm):
    """A PAC algorithm with optional exploitation."""

    def explore(self) -> None:
        """Do one step of exploration."""
        raise NotImplementedError()

    def exploit(self) -> None:
        """Do one step of exploitation."""
        raise NotImplementedError()

    def step(self) -> None:
        """Run one step of the algorithm.

        This corresponds to a logical step of the algorithm and may perform
        multiple comparisons. What exactly a "logical step" is depends on the
        algorithm.

        This will delegate to either the ``explore`` or ``exploit`` function,
        depending on whether or not the exploration is finished.
        """
        if not self.exploration_finished():
            self.explore()
        else:
            self.exploit()

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        Returns
        -------
        bool
            True if the exploration phase is finished.
        """
        raise NotImplementedError()

    def is_finished(self) -> bool:
        """Determine if the algorithm is finished.

        If a time horizon is given ("regret minimizing mode"), this function
        returns true if and only if the time horizon has been reached. The time
        horizon serves as both an upper and a lower bound, the
        ``exploration_finished`` condition is ignored.

        If no time horizon is given ("PAC mode") this function delegates to
        ``exploration_finished``.

        Returns
        -------
        bool
            True if the algorithm is finished and should be stopped.
        """
        # if self.time_horizon is not None:
        #     # "Regret-minimizing mode"
        #     return self.feedback_mechanism.get_num_duels() >= self.time_horizon
        # else:
        # "PAC mode"
        return self.exploration_finished()


class CondorcetProducer(Algorithm):
    """An Algorithm that computes or estimates a Condorcet winner."""

    def get_condorcet_winner(self) -> Optional[int]:
        """Return the computed Condorcet winner if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError

    def exploit(self) -> None:
        """Run one step of exploitation."""
        winner = self.get_condorcet_winner()
        assert winner is not None
        self.feedback_mechanism.duel(winner, winner)


class SingleCopelandProducer(Algorithm):
    """An Algorithm that computes or estimates one of the Copeland winners."""

    def get_copeland_winner(self) -> Optional[int]:
        """Return the computed Copeland winner if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError

    def exploit(self) -> None:
        """Run one step of exploitation."""
        winner = self.get_copeland_winner()
        assert winner is not None
        self.feedback_mechanism.duel(winner, winner)


class AllCopelandProducer(Algorithm):
    """An Algorithm that computes or estimates all Copeland winners."""

    def get_copeland_winners(self) -> Optional[Collection[int]]:
        """Return the computed Copeland winner if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError

    def exploit(self) -> None:
        """Run one step of exploitation."""
        winners = self.get_copeland_winners()
        assert winners is not None and len(winners) > 0
        # Pick any winner. The variance in the cumulative regret would be
        # smaller if we picked one at random, but we do not have access to a
        # random state here.
        winner = list(winners)[0]
        self.feedback_mechanism.duel(winner, winner)


class CopelandRankingProducer(Algorithm):
    """An Algorithm that computes or estimates the Copeland ranking over the arms."""

    def get_ranking(self) -> Optional[List[int]]:
        """Return the computed Copeland ranking if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError

    def exploit(self) -> None:
        """Run one step of exploitation."""
        ranking = self.get_ranking()
        assert ranking is not None
        self.feedback_mechanism.duel(ranking[0], ranking[0])


class PartialRankingProducer(Algorithm):
    """An Algorithm that computes or estimates the partial ranking over the arms."""

    def get_partial_ranking(self) -> Optional[List[int]]:
        """Return the computed partial ranking if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError

    def exploit(self) -> None:
        """Run one step of exploitation."""
        ranking = self.get_partial_ranking()
        assert ranking is not None
        self.feedback_mechanism.duel(ranking[0], ranking[0])


class AllApproximateCondorcetProducer(Algorithm):
    """An Algorithm that approximates the Condorcet winner."""

    def get_approximate_condorcet_winners(self) -> Optional[Collection[int]]:
        """Return all approximate Condorcet winners if they are determined.

        This will only return a result when ``step`` has been called a
        sufficient amount of times.
        """
        raise NotImplementedError

    def exploit(self) -> None:
        """Run one step of exploitation."""
        winners = self.get_approximate_condorcet_winners()
        assert winners is not None and len(winners) > 0
        # Pick any winner. The variance in the cumulative regret would be
        # smaller if we picked one at random, but we do not have access to a
        # random state here.
        winner = list(winners)[0]
        self.feedback_mechanism.duel(winner, winner)


class BordaProducer(Algorithm):
    """An Algorithm that computes or estimates a Borda winner."""

    def get_borda_winner(self) -> Optional[int]:
        """Return the computed Borda winner if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError

    def exploit(self) -> None:
        """Run one step of exploitation."""
        borda_winner = self.get_borda_winner()
        assert borda_winner is not None
        self.feedback_mechanism.duel(borda_winner, borda_winner)


class PreferenceMatrixProducer(Algorithm):
    """An Algorithm that computes or estimates approximate pairwise probability over the arms."""

    def get_preference_matrix(self) -> Optional[PreferenceMatrix]:
        """Return the computed preference matrix if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError


class TopKArmsProducer(Algorithm):
    """An algorithm that computes the best k arms. The definition of best is up for the specific algorithm."""

    def get_top_k(self) -> Optional[Collection[int]]:
        """Return the top k arms.

        This will only return a result when ``step`` has been called a
        sufficient amount of times.
        """
        raise NotImplementedError

    def exploit(self) -> None:
        """Run one step of exploitation."""
        winners = self.get_top_k()
        assert winners is not None and len(winners) > 0
        # Pick any winner. The variance in the cumulative regret would be
        # smaller if we picked one at random, but we do not have access to a
        # random state here.
        winner = list(winners)[0]
        self.feedback_mechanism.duel(winner, winner)
