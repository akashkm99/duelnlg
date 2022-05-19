"""Collection of various sorting algorithms which allow a step-by-step execution."""
from typing import Callable
from typing import List
from typing import Optional

import numpy as np

from duelnlg.duelpy.util.utility_functions import pop_random


class SortingAlgorithm:
    """Superclass for sorting algorithms."""

    def __init__(
        self,
        items: List[int],
        compare_fn: Callable[[int, int], int],
        random_state: np.random.RandomState,
    ):
        self.items = items
        self.compare_fn = compare_fn
        self.random_state = random_state

    def step(self) -> None:
        """Execute one step of sorting."""
        raise NotImplementedError

    def is_finished(self) -> bool:
        """Determine whether the sorting is complete."""
        raise NotImplementedError

    def get_result(self) -> Optional[List[int]]:
        """Get sorted list."""
        raise NotImplementedError

    @staticmethod
    def get_comparison_bound(num_arms: int) -> float:
        """Get a bound on the amount of comparisons made."""
        raise NotImplementedError


class MergeSort(SortingAlgorithm):
    """Implement the Mergesort algorithm.

    Parameters
    ----------
    items
        The list of indices to compare.
    compare_fn
        A function comparing two indices, the returned value should be ``1`` if the first one should precede the second one and ``-1`` otherwise. Optionally, if no order can be determined (yet), ``0`` can be returned to defer the decision.
    random_state
        Used for randomization of the pivot selection.

    Attributes
    ----------
    current_node
        The node currently undergoing the partitioning step. The algorithm creates a node for each recursive division of the elements, forming a rooted tree.

    Examples
    --------
    >>> rs = np.random.RandomState(2)
    >>> items = rs.uniform(size=10)
    >>> ms = MergeSort(list(items), lambda x,y: 1 if x<y else -1, rs)
    >>> while not ms.is_finished() :
    ...     ms.step()
    >>> ground_truth = np.sort(items)
    >>> all(ground_truth == ms.get_result())
    True
    """

    class Node:
        """Helper class.

        This class is only to be used internally, representing nodes in the constructed tree.
        """

        def __init__(self, result: List[int]):
            # notice the implicit link here, result is not copied!
            # editing node.result will therefore change node.parent.left or right, since the list instance is the same
            # the slicing will create new lists though
            self.result = result
            split_index = int(np.ceil(len(self.result) / 2))
            self.left = self.result[:split_index]
            self.right = self.result[split_index:]

    def __init__(
        self,
        items: List[int],
        compare_fn: Callable[[int, int], int],
        random_state: np.random.RandomState,
    ):
        super().__init__(items, compare_fn, random_state)
        # prepare all recursion steps
        # the list todo contains all nodes of the tree that is implicitly created when running mergesort
        self.todo = [MergeSort.Node(self.items)]

        for node in self.todo:
            if len(node.result) > 1:
                self.todo.append(MergeSort.Node(node.right))
                self.todo.append(MergeSort.Node(node.left))
        self.current_node: MergeSort.Node = self.todo[-1]
        self.index_left = 0
        self.index_right = 0

        self._is_finished = False
        self.result: Optional[List[int]] = None

    def step(self) -> None:
        """Execute one sort step."""
        if self._is_finished:
            return
        if (
            len(self.current_node.left) == self.index_left
            and len(self.current_node.right) == self.index_right
        ):
            # merge step completed, next node is selected from todo
            if len(self.todo) > 0:
                self.current_node = self.todo.pop()
                self.current_node.result.clear()
                self.index_left = 0
                self.index_right = 0
            else:
                self._is_finished = True
                self.result = self.current_node.result
        else:
            # merge two lists, by adding one item

            # compare the current element of both child lists
            if self.index_left < len(self.current_node.left) and self.index_right < len(
                self.current_node.right
            ):
                arm_left = self.current_node.left[self.index_left]
                arm_right = self.current_node.right[self.index_right]

                comparison_result = 0
                while comparison_result == 0:
                    comparison_result = self.compare_fn(arm_left, arm_right)
                if comparison_result == 1:  # arm_left -> arm_right
                    self.current_node.result.append(arm_left)
                    self.index_left += 1
                else:  # arm_right -> arm_left
                    self.current_node.result.append(arm_right)
                    self.index_right += 1
            else:
                # if there are nodes left in one list, one of them is added

                if self.index_left < len(self.current_node.left):
                    arm_left = self.current_node.left[self.index_left]
                    self.current_node.result.append(arm_left)
                    self.index_left += 1

                if self.index_right < len(self.current_node.right):
                    arm_right = self.current_node.right[self.index_right]
                    self.current_node.result.append(arm_right)
                    self.index_right += 1

    def is_finished(self) -> bool:
        """Determine whether the sorting is complete."""
        return self._is_finished

    def get_result(self) -> Optional[List[int]]:
        """Get sorted list."""
        return self.result

    @staticmethod
    def get_comparison_bound(num_arms: int) -> float:
        """Get the upper bound for the amount of comparisons made by the two-way top-down merge sort algorithm."""
        # For details see Theorem 1 of Flajolet, P. and Golin, M. J. Mellin transforms and asymptotics: The mergesort recurrence.Acta Inf., 31(7):673–696, 1994
        return int(np.ceil(num_arms * np.log2(num_arms) - 0.91392 * num_arms + 1))


class Quicksort(SortingAlgorithm):
    """Implement the Quicksort algorithm.

    Some algorithms depend on Quicksort to rank arms, in order to support a step function, Quicksort is implemented with the ability to advance single sorting steps.

    Parameters
    ----------
    items
        The list of indices to compare.
    compare_fn
        A function comparing two indices, the returned value should be ``1`` if the first one should precede the second one and ``-1``. Optionally, if no order can be determined (yet), ``0`` can be returned to defer the decision.
    random_state
        Used for randomization of the pivot selection.

    Attributes
    ----------
    current_node
        The node currently undergoing the partitioning step. The algorithm creates a node for each recursive division of the elements, forming a rooted tree.

    Examples
    --------
    >>> rs = np.random.RandomState(2)
    >>> items = rs.uniform(size=10)
    >>> qs =Quicksort(list(items), lambda x,y: 1 if x<y else -1, rs)
    >>> while not qs.is_finished() :
    ...     qs.step()
    >>> ground_truth = np.sort(items)
    >>> all(ground_truth == qs.get_result())
    True
    """

    class Node:
        """Helper class.

        This class is only to be used internally, representing nodes in the constructed tree.
        """

        def __init__(
            self,
            items: List[int],
            random_state: np.random.RandomState,
            parent: Optional["Quicksort.Node"] = None,
        ):
            self.items = items
            self.random_state = random_state
            self.parent = parent
            self.left: Optional["Quicksort.Node"] = None
            self.right: Optional["Quicksort.Node"] = None
            self.pivot = pop_random(self.items, self.random_state)[0]
            self.marked = False

        def mark(self) -> None:
            """Mark the node as done."""
            self.marked = True

        def __repr__(self) -> str:
            """Transform the object to a human-readable string."""
            return f"Quicksort.Node({self.items},{self.pivot},marked:{self.marked},parent:{self.parent is not None},left:{self.left is not None},right:{self.right is not None})"

    def __init__(
        self,
        items: List[int],
        compare_fn: Callable[[int, int], int],
        random_state: np.random.RandomState,
    ):
        super().__init__(items, compare_fn, random_state)
        self.current_node = Quicksort.Node(items.copy(), self.random_state)
        self._result: Optional[List[int]] = None
        self._is_finished = False

    @staticmethod
    def _merge_children(node: "Quicksort.Node") -> List[int]:
        left_list: List[int] = []
        if node.left is not None:
            left_list = node.left.items
        right_list: List[int] = []
        if node.right is not None:
            right_list = node.right.items
        return left_list + [node.pivot] + right_list

    # pylint: disable=too-many-branches
    def step(self) -> None:
        """Execute one sort step."""
        if self.is_finished():
            return
        if (
            self.current_node.left is None
            and self.current_node.right is None
            and len(self.current_node.items) > 0
        ):
            # create new child nodes with smaller problems
            current_list = self.current_node.items
            list_left = []
            list_right = []
            while len(current_list) > 0:
                for item in current_list:
                    comparison_result = self.compare_fn(self.current_node.pivot, item)
                    if comparison_result == 1:  # pivot -> item
                        list_right.append(item)
                        current_list.remove(item)
                    elif comparison_result == -1:  # item -> pivot
                        list_left.append(item)
                        current_list.remove(item)
            if len(list_left) > 0:
                self.current_node.left = Quicksort.Node(
                    list_left, self.random_state, self.current_node
                )
            if len(list_right) > 0:
                self.current_node.right = Quicksort.Node(
                    list_right, self.random_state, self.current_node
                )
        elif self.current_node.left is not None and not self.current_node.left.marked:
            # solve left child
            self.current_node = self.current_node.left
            self.step()  # this branch does not count as a step
        elif self.current_node.right is not None and not self.current_node.right.marked:
            # solve right child
            self.current_node = self.current_node.right
            self.step()  # this branch does not count as a step
        elif self.current_node.parent is not None:
            # move to parent and merge results of children
            self.current_node.items = self._merge_children(self.current_node)
            self.current_node.mark()
            self.current_node = self.current_node.parent
            self.step()  # this branch does not count as a step
        else:
            # Quicksort terminates, root reached
            self._result = self._merge_children(self.current_node)
            self._is_finished = True

    def is_finished(self) -> bool:
        """Determine whether the sorting is complete."""
        return self._is_finished

    def get_result(self) -> Optional[List[int]]:
        """Get sorted list."""
        return self._result

    def _gather_intermediate_result(self, node: "Quicksort.Node") -> List[List[int]]:
        """Get the partial intermediate result associated with a node."""
        result = []
        if node.left is not None or node.right is not None:
            if node.left is not None:
                result += self._gather_intermediate_result(node.left)
            result.append([node.pivot])
            if node.right is not None:
                result += self._gather_intermediate_result(node.right)
        else:
            items = node.items.copy()
            if not node.marked:
                items.append(
                    node.pivot
                )  # this is not stable, we are unable to insert the pivot at the old position!
            result.append(items)
        return result

    def get_intermediate_result(self) -> List[List[int]]:
        """Get best estimate of the result given the previously executed steps."""
        root = self.current_node
        while root.parent is not None:
            root = root.parent
        return self._gather_intermediate_result(root)

    @staticmethod
    def get_comparison_bound(num_arms: int) -> float:
        """Get the upper bound for the amount of comparisons made by the Quicksort algorithm."""
        return num_arms ** 2 / 2
