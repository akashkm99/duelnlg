"""Implementation of various helper functions."""
import math
from typing import Callable
from typing import List
from typing import Optional

import numpy as np
from scipy.special import rel_entr


def argmin_set(
    array: np.array, exclude_indexes: Optional[List[int]] = None
) -> List[int]:
    """Calculate the complete argmin set, returning an array with all indices.

    It removes the ``exclude_indexes`` from the  array and calculates the indices set with minimum value from remaining
    indexes of array.

    Parameters
    ----------
    array
        The 1-D array for which the argmin should be calculated.
    exclude_indexes
        Indices to exclude in the argmin operation.

    Returns
    -------
    indices
        A 1-D array containing all indices which point to the minimum value.
    """
    # np.argmin only returns the first index, to get the whole set,
    # we first find the minimum and then search for all indices which point
    # to a value equal to this minimum
    if exclude_indexes is None or len(exclude_indexes) == 0:
        # For this case the simpler implementation is more efficient, although
        # the other one with a trivial mask would also work.
        return np.argwhere(array == np.amin(array)).flatten()
    mask = np.zeros(len(array), dtype=bool)
    mask[exclude_indexes] = True
    min_value = np.min(np.ma.array(array, mask=mask))
    indices = set(np.ndarray.flatten(np.argwhere(array == min_value)))
    indices = indices - set(exclude_indexes) if exclude_indexes is not None else indices
    return list(indices)


def argmax_set(
    array: np.array, exclude_indexes: Optional[List[int]] = None
) -> List[int]:
    """Calculate the complete argmax set, returning an array with all indices..

    It removes the ``exclude_indexes`` from the  array and calculates the indices set with maximum value from the remaining
    indexes of array.

    Parameters
    ----------
    array
        The 1-D array for which the argmax should be calculated
    exclude_indexes
        Indices to exclude in the argmax operation.

    Returns
    -------
    indices
        A 1-D array containing all indices which point to the maximum value.
    """
    # np.argmax only returns the first index, to get the whole set,
    # we first find the maximum and then search for all indices which point
    # to a value equal to this maximum
    if exclude_indexes is None or len(exclude_indexes) == 0:
        # For this case the simpler implementation is more efficient, although
        # the other one with a trivial mask would also work.
        return np.argwhere(array == np.amax(array)).flatten()
    mask = np.zeros(len(array), dtype=bool)
    mask[exclude_indexes] = True
    max_value = np.max(np.ma.array(array, mask=mask))
    indices = set(np.ndarray.flatten(np.argwhere(array == max_value)))
    indices = indices - set(exclude_indexes) if exclude_indexes is not None else indices
    return list(indices)


def pop_random(
    input_list: List[int], random_state: np.random.RandomState, amount: int = 1
) -> List[int]:
    """Remove randomly chosen elements from a given list and return them.

    If the list contains less than or exactly `amount` elements, all elements are chosen.

    Parameters
    ----------
    input_list
        The list from which an arm should be removed.
    random_state
        The random state to use.
    amount
        The amount of elements to pick, defaults to 1.

    Returns
    -------
    List[int]
        The list containing the removed elements.
    """
    if len(input_list) <= amount:
        list_copy = input_list.copy()
        input_list.clear()
        return list_copy
    else:
        picked = []
        left_over = input_list
        for _ in range(amount):
            random_index = random_state.randint(0, len(input_list))
            removed_element = left_over.pop(random_index)
            picked.append(removed_element)
        return picked


def kullback_leibler_divergence_plus(
    probability_p: float, probability_q: float
) -> float:
    """Implement KL divergence plus for Bernoulli random variable.	
    The KL-divergence plus (i.e for two Bernoulli random variable with parameters as probability p, q) from q to	
    p is formulated as :math:`d(p,q)+ = d(p,q)` if :math:`p<q` else 0.	
    Parameters	
    ----------	
    probability_p	
        The preference probability of one arm over another.	
    probability_q	
        The preference probability of one arm over another.	
    Returns	
    -------	
    float	
        divergence plus measures between two probability.	
    """
    if probability_p < probability_q:
        return rel_entr(1 - probability_p, probability_q,) + rel_entr(
            probability_p, probability_q,
        )
    return 0


def newton_raphson(
    reference_point: float,
    function: Callable[[float], float],
    derivative_of_function: Callable[[float], float],
    allowed_number_of_iterations: int = 50,
    function_value_epsilon: float = 1.48e-8,
) -> float:
    """Use Newton-Raphson method of finding an approximate root.

    Find an approximate root of the provided `function` within a given
    number of iterations. The Newton-Raphson method starts with a given
    `reference_point` around which a root can be expected to exist.
    Refer https://en.wikipedia.org/wiki/Newton%27s_method for details.

    Parameters
    ----------
    reference_point
        The start point for finding the root.
    function
        The function whose root needs to be found.
    derivative_of_function
        A non-zero derivative of the above function.
    allowed_number_of_iterations
        Number of iterations allowed to find the root.
        Default value is 50.
    function_value_epsilon
        The small difference or :math:`epsilon` over the value of the given
        function for which a root could be assumed to have been found and
        returned. Default value is ``1.48e-8``. The value ``1.48e-8`` is
        taken from the parameter ``tol`` of the ``newton`` function of
        ``scipy``. Refer the following link for details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html

    Returns
    -------
    float
        An approximate root or `math.inf` for corner cases.
    """
    iteration = 0
    approx_root = reference_point

    while iteration < allowed_number_of_iterations:
        function_value = function(approx_root)
        if function_value <= function_value_epsilon:
            break

        if math.isnan(function_value) or math.isinf(function_value):
            return math.inf

        function_derivative_value = derivative_of_function(approx_root)
        if (
            math.isnan(function_derivative_value)
            or math.isinf(function_derivative_value)
            or function_derivative_value == 0.0
        ):
            return math.inf

        approx_root -= function_value / function_derivative_value
        iteration += 1

    return approx_root
