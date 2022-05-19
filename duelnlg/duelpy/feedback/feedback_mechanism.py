"""A generic way to compare two arms against each other."""


class FeedbackMechanism:
    """Some means of comparing two arms."""

    def __init__(self, arms: list) -> None:
        self.arms = arms

    # In our final design we will probably want a better arm representation to
    # avoid restricting it to int.
    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        raise NotImplementedError

    def get_num_duels(self) -> int:
        """Get the number of duels that were already performed.

        Returns
        -------
        int
            The number of duels.
        """
        raise NotImplementedError

    def get_arms(self) -> list:
        """Get the pool of arms available."""
        return self.arms.copy()

    def get_num_arms(self) -> int:
        """Get the number of arms."""
        return len(self.arms)
