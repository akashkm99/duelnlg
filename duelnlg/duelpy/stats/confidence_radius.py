"""Implementation of various confidence intervals with different assumptions."""

from typing import Callable

import numpy as np


class ConfidenceRadius:
    """An abstract superclass for confidence-radius definitions."""

    def __call__(self, num_samples: int) -> float:
        """Compute the confidence radius.

        Parameters
        ----------
        num_samples
            The number of samples that were already taken.

        Returns
        -------
        float
            The confidence radius around the empirical mean.
        """
        raise NotImplementedError


class TrivialConfidenceRadius(ConfidenceRadius):
    """A trivial confidence radius that contains no information.

    Only useful as a place-holder. Always returns a pre-determined confidence-radius.

    Parameters
    ----------
    radius
        The constant confidence radius to return.
    """

    def __init__(self, radius: float = 1.0):
        self._radius = radius

    def __call__(self, num_samples: int) -> float:
        """Compute the confidence radius.

        Parameters
        ----------
        num_samples
            The number of samples that were already taken.

        Returns
        -------
        float
            The confidence radius around the empirical mean.
        """
        return self._radius


class HoeffdingConfidenceRadius(ConfidenceRadius):
    r"""A confidence radius based on Hoeffding's inequality and the Union bound.

    :math:`\sqrt{\frac{\textit{factor}}{2 \cdot \textit{num_samples}} \log\left(\frac{\textit{prob_scaling(num_samples)}}{\textit{failure_probability}}\right)}`

    Parameters
    ----------
    failure_probability
        The probability that the actual value does not lie within the computed
        confidence interval.
    probability_scaling_factor
        A factor by which to scale the failure_probability, dependent on the
        number of samples that were already taken. This is often useful when
        multiple random variables are estimated using this confidence interval
        and we want to bound the union of their failures. In that case, it can
        be necessary to scale the failure probability of any individual
        estimate.
    factor
        This factor is applied inside the square root of the radius calculation.
        It allows to scale the influence of the number of samples taken.

    Attributes
    ----------
    failure_probability
    probability_scaling_factor
    factor
    """

    def __init__(
        self,
        failure_probability: float,
        probability_scaling_factor: Callable[[int], float] = lambda num_samples: 1,
        factor: float = 1,
    ):
        self.failure_probability = failure_probability
        self.probability_scaling_factor = probability_scaling_factor
        self.factor = factor

    def __call__(self, num_samples: int) -> float:
        """Compute the confidence radius.

        The random variable will deviate by at most this radius from the
        empirical mean estimate with high probability.

        For more details about the Hoeffding inequality and the derivation of
        this confidence radius, refer to `wikipedia`_ and re-solve for ``t``.

        .. _wikipedia: https://en.wikipedia.org/wiki/Hoeffding%27s_inequality#Confidence_intervals

        Parameters
        ----------
        num_samples
            The number of samples that were already taken.

        Returns
        -------
        float
            The confidence radius around the empirical mean.
        """
        if num_samples == 0:
            return 1
        adjusted_probability = (
            self.probability_scaling_factor(num_samples) / self.failure_probability
        )

        # If we are still too uncertain (otherwise there would be a negative
        # value in the square root).

        # print ("adjusted_probability: ", adjusted_probability)
        if adjusted_probability < 1:
            return 0
        in_sqrt = self.factor / (2 * num_samples) * np.log(adjusted_probability)

        # print ("Condifence bound for {} samples is: {}".format(num_samples, np.sqrt(in_sqrt)))

        return np.sqrt(in_sqrt)
