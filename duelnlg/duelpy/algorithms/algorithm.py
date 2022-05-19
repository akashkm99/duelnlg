"""A generic superclass which contains the common functions required in the implemented PB-MAB algorithms."""

from typing import Optional

from duelnlg.duelpy.feedback import FeedbackMechanism


class Algorithm:
    """Parent class of all the implemented PB-MAB algorithms."""

    def __init__(
        self, feedback_mechanism: FeedbackMechanism, time_horizon: Optional[int]
    ):
        self.feedback_mechanism = feedback_mechanism
        self.time_horizon = time_horizon

    def step(self) -> None:
        """Run one step of the algorithm.

        This corresponds to a logical step of the algorithm and may perform
        multiple comparisons. What exactly a "logical step" is depends on the
        algorithm.
        """
        raise NotImplementedError

    def is_finished(self) -> bool:
        """Determine if the algorithm is finished.

        This may be based on a time horizon or a different algorithm-specific
        termination criterion if time_horizon is ``None``.
        """
        if self.time_horizon is None:
            raise NotImplementedError(
                "No time horizon set and no custom termination condition implemented."
            )
        return self.feedback_mechanism.get_num_duels() >= self.time_horizon

    def run(self) -> None:
        """Run the algorithm until completion.

        Completion is determined through the ``is_finished`` function. Refer to
        its documentation for more details. You can equivalently call ``step``
        manually until ``is_finished`` returns true.
        """
        while not self.is_finished():
            self.step()
