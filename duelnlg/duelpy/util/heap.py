"""Implementation of a heap which allows a step-by-step execution."""
from enum import Enum
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple


class Heap:
    """Superclass for sorting algorithms.

    This implements only insert, get_min and update_min_key, as the comparisons may be probabilistic.

    Parameters
    ----------
    compare_fn
        A function comparing two given keys, returns 1 if the first key is smaller than the second, -1 otherwise and 0 if it cannot be decided yet.

    Attributes
    ----------
    compare_fn

    Examples
    --------
    A build heap example:
    >>> heap = Heap(lambda x,y: 1 if x<y else -1)
    >>> heap.build([3,1,5,2,4], [3,1,5,2,4])
    >>> comparisons = 0
    >>> while not heap.is_finished():
    ...     heap.step()
    ...     comparisons += 1
    >>> heap.get_min()
    (1, 1)
    >>> comparisons
    4

    An insert example:

    >>> heap = Heap(lambda x,y: 1 if x<y else -1)
    >>> heap.insert(2,2)
    >>> comparisons = 0
    >>> while not heap.is_finished():
    ...     heap.step()
    ...     comparisons += 1
    >>> heap.insert(1,1)
    >>> while not heap.is_finished():
    ...     heap.step()
    ...     comparisons += 1
    >>> heap.insert(3,3)
    >>> while not heap.is_finished():
    ...     heap.step()
    ...     comparisons += 1
    >>> heap.get_min()
    (1, 1)
    >>> comparisons
    3
    >>> heap.delete(1)
    >>> while not heap.is_finished():
    ...     heap.step()
    >>> heap.get_min()
    (2, 2)
    >>> heap.delete(1)
    >>> while not heap.is_finished():
    ...     heap.step()
    >>> heap.get_min()
    (2, 2)
    """

    class State(Enum):
        """Enumeration for all possible states of the heap."""

        NOOP = 0
        DOWN = 1
        UP = 2
        BUILD = 3

    class Node:
        """Represents a node in the heap."""

        def __init__(self, key: int, value: Any):
            self.key = key
            self.value = value

        def __repr__(self) -> str:
            """Transform the object to a human-readable string."""
            return str(self.value)

    def __init__(self, compare_fn: Callable[[int, int], int]):
        self.compare_fn = compare_fn
        self._data: List["Heap.Node"] = []
        self._current_operation = Heap.State.NOOP
        self._current_node: Optional[int] = None
        self._build_node: Optional[int] = None
        self._heapify_down_left_checked = (
            False  # this stores whether the left child has been compared
        )

    @staticmethod
    def _get_parent_index(index: int) -> Optional[int]:
        """Calculate the index of a node parent.

        Parameters
        ----------
        index
            The index to calculate the parent for.

        Returns
        -------
        Optional[int]
            The index of the parent or None if no parent exists.
        """
        if index == 0:
            return None
        else:
            return (index - 1) // 2

    @staticmethod
    def _get_child_indices(index: int) -> Tuple[int, int]:
        """Calculate the indices of the children of a node.

        Parameters
        ----------
        index
            The index to calculate the children for.

        Returns
        -------
        Tuple[Optional[int], Optional[int]]
            The indices of the left and right child.
        """
        return 2 * (index + 1) - 1, 2 * (index + 1)

    def _heapify_up(self) -> None:
        """Perform one step of heapify up."""
        if self._current_node is None:
            raise RuntimeError("Current node is None.")
        parent = Heap._get_parent_index(self._current_node)
        # abort if the root is reached or the heap condition is met
        if parent is None:
            self._current_node = None
            self._current_operation = Heap.State.NOOP
        else:
            comparison_result = self.compare_fn(
                self._data[parent].key, self._data[self._current_node].key
            )

            if comparison_result == 1:
                self._current_node = None
                self._current_operation = Heap.State.NOOP
            elif comparison_result == -1:
                self._data[parent], self._data[self._current_node] = (
                    self._data[self._current_node],
                    self._data[parent],
                )
                self._current_node = parent

    def _heapify_down(self) -> None:
        """Perform one step of heapify down."""
        if self._current_node is None:
            raise RuntimeError("Current node is None.")
        left, right = Heap._get_child_indices(self._current_node)
        if left < len(self._data) and not self._heapify_down_left_checked:
            comparison_result = self.compare_fn(
                self._data[left].key, self._data[self._current_node].key
            )
            if comparison_result == 1:
                self._data[left], self._data[self._current_node] = (
                    self._data[self._current_node],
                    self._data[left],
                )
                if Heap._get_child_indices(left)[0] < len(self._data):
                    self._current_node = left
                    self._heapify_down_left_checked = False
                else:
                    self._current_node = None
                    self._heapify_down_left_checked = False
                    self._current_operation = Heap.State.NOOP
            elif comparison_result == -1:
                self._heapify_down_left_checked = True
        elif right < len(self._data):
            comparison_result = self.compare_fn(
                self._data[right].key, self._data[self._current_node].key
            )
            if comparison_result == 1:
                self._data[right], self._data[self._current_node] = (
                    self._data[self._current_node],
                    self._data[right],
                )
                if Heap._get_child_indices(left)[0] < len(self._data):
                    self._current_node = right
                    self._heapify_down_left_checked = False
                else:
                    self._current_node = None
                    self._heapify_down_left_checked = False
                    self._current_operation = Heap.State.NOOP
            elif comparison_result == -1:
                self._current_node = None
                self._heapify_down_left_checked = False
                self._current_operation = Heap.State.NOOP
        else:
            self._current_node = None
            self._heapify_down_left_checked = False
            self._current_operation = Heap.State.NOOP

    def _build_step(self) -> None:
        """Perform one step of build heap."""
        self._heapify_down()
        if self._current_node is None:
            if self._build_node == 0:
                self._current_operation = Heap.State.NOOP
                self._build_node = None
            else:
                assert self._build_node is not None
                self._build_node -= 1
                self._current_operation = Heap.State.BUILD
                self._current_node = self._build_node

    def step(self) -> None:
        """Execute one step of sorting."""
        if self._current_operation == Heap.State.NOOP:
            raise RuntimeError("step is called but no operation is in progress.")
        if self._current_operation == Heap.State.UP:
            self._heapify_up()
        elif self._current_operation == Heap.State.DOWN:
            self._heapify_down()
        elif self._current_operation == Heap.State.BUILD:
            self._build_step()

    def is_finished(self) -> bool:
        """Determine whether the current operation is complete."""
        return self._current_operation == Heap.State.NOOP

    def insert(self, key: int, value: Any) -> None:
        """Insert an element in the heap.

        Parameters
        ----------
        key
            The key, an integer.
        value
            The associated object.
        """
        new_node = Heap.Node(key, value)
        self._data.append(new_node)
        if len(self._data) > 1:
            self._current_node = len(self._data) - 1
            self._current_operation = Heap.State.UP

    def get_min(self) -> Optional[Tuple[int, Any]]:
        """Get minimum key and value.

        Returns
        -------
        Optional[Tuple[int, Any]]
            The key and value of the minimum, or None if the heap is empty.
        """
        if len(self._data) == 0:
            return None
        else:
            return self._data[0].key, self._data[0].value

    def update_min_key(self, new_key: int) -> None:
        """Change the minimum key and repair the heap.

        Parameters
        ----------
        new_key
            The key replacing the minimum key.
        """
        self._data[0].key = new_key
        if len(self._data) > 1:
            self._current_operation = Heap.State.DOWN
            self._current_node = 0

    def build(self, keys: List[int], values: List[Any]) -> None:
        """Build a heap from given keys and values, has better comparison complexity than inserting them individually.

        Parameters
        ----------
        keys
            A list of integer keys.
        values
            A list of associated objects.

        Raises
        ------
        ValueError
            Raised if the number of keys and values do not match.
        """
        self._data = []
        if len(keys) != len(values):
            raise ValueError("An equal amount of keys and values is required.")
        for key, value in zip(keys, values):
            self._data.append(Heap.Node(key, value))
        if len(self._data) > 1:
            self._build_node = len(self._data) // 2 - 1
            self._current_node = self._build_node
            self._current_operation = Heap.State.BUILD

    def delete(self, key: int) -> None:
        """Delete the first element with the given key.

        Parameters
        ----------
        key
            The key of the node to delete.
        """
        node_index: int = 0
        for index, element in enumerate(self._data):
            if element.key == key:
                node_index = index
                break
        else:
            return
        self._data[node_index] = self._data[-1]
        self._data.pop()
        self._current_node = node_index
        self._current_operation = Heap.State.DOWN
