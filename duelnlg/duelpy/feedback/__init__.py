"""Various mechanisms for comparing arms."""

from duelnlg.duelpy.feedback.commandline_feedback import CommandlineFeedback
from duelnlg.duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelnlg.duelpy.feedback.matrix_feedback import MatrixFeedback
from duelnlg.duelpy.feedback.nlg_feedback import NLGFeedback

__all__ = ["FeedbackMechanism", "MatrixFeedback", "CommandlineFeedback"]
