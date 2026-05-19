from dataclasses import dataclass
from typing import (
    Optional,
    Any,
    Callable,
    TypeAlias,
)
from enum import Enum

NodePredicate: TypeAlias = Callable[[Any], bool]


class ConditionMode(str, Enum):
    """
    Supported matching modes for value conditions.

    ## Modes:
    - **EXACT**: Match when the actual value is exactly equal to the expected value.
    - **NEGATION**: Match when the actual value is not equal to the expected value.
    - **WILDCARD**: Match any value (expected value is ignored, must be None).
    - **MEMBERSHIP**: Match when the actual (scalar) value is in the expected iterable
      of condition values.  The *condition* holds a collection; the *attribute* is scalar.
    - **NOT_MEMBERSHIP**: Match when the actual (scalar) value is not in the expected
      iterable of condition values.  This is the logical inverse of MEMBERSHIP.
    - **CONTAINS**: Match when the expected (scalar) condition value is found among the
      elements or keys of a collection-valued attribute.  This is the logical reverse of
      MEMBERSHIP: the *attribute* holds a collection; the *condition* is scalar.

      For dict-valued attributes, CONTAINS iterates over the **keys** of the dict
      (not the values) and checks whether any key equals the condition value.  This
      mirrors Python's ``value in dict`` semantics.  For list, tuple, set, and frozenset
      attributes, it iterates over the elements.  Strings are treated as scalar values
      and will never match in CONTAINS mode (use EXACT for string equality).
    """

    EXACT = "exact"  # Match when actual value is exactly equal to expected value
    NEGATION = "negation"  # Match when actual value is not equal to expected value
    WILDCARD = "wildcard"  # Match any value (expected value is ignored, must be None)
    MEMBERSHIP = "membership"  # Match when actual value is in the expected iterable (list, tuple, set, etc.)
    NOT_MEMBERSHIP = "not_membership"  # Match when actual value is not in the expected iterable (list, tuple, set, etc.)
    CONTAINS = "contains"  # Match when the expected value is found among the elements/keys of a collection-valued attribute


class DirectionMode(str, Enum):
    """
    Supported edge direction modes for iterating edges in the syntax graph.
    """

    UP = "up"  # Move from id to head (up the tree)
    DOWN = "down"  # Move from head to id (down the tree)
    BOTH = "both"  # Include both up and down edges (default)


@dataclass(frozen=True, slots=True)
class EdgeContext:
    """
    Context for an edge in the dependency graph, used for matching edges during feature extraction.
    """

    direction: DirectionMode
    deprel: Optional[str] = None
    hops: int = 1
    crosses_sentence: bool = False
