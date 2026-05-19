import estnltk
from typing import (
    Dict,
    Optional,
    Tuple,
    Self,
    Any,
    Callable,
)
from dataclasses import dataclass

from .types import ConditionMode, DirectionMode, NodePredicate, EdgeContext

from .config import (
    DEFAULT_MISSING_MARKERS,
    SELECTIVITY_WEIGHT_EXACT,
    SELECTIVITY_WEIGHT_NEGATION,
    SELECTIVITY_WEIGHT_MEMBERSHIP,
    SELECTIVITY_WEIGHT_CONTAINS,
    SELECTIVITY_WEIGHT_EXTRA_PREDICATE,
    RESERVED_NODE_ATTRIBUTE_NAMES,
    RESERVED_EDGE_ATTRIBUTE_NAMES,
)


@dataclass(frozen=True, slots=True)
class ValueCondition:
    """
    Match one value using exact, negation, wildcard, membership, or contains logic.

    ## Attributes:
    - **mode** (`ConditionMode`): The matching mode to use (EXACT, NEGATION, WILDCARD, MEMBERSHIP, or CONTAINS).
    - **value** (`Any`, optional): The value to match against. Required for EXACT, NEGATION, MEMBERSHIP, and CONTAINS modes. Must be None for WILDCARD mode. Defaults to None.
    - **allow_missing** (`bool`, optional): Whether to allow missing values (e.g., None, empty string, or other specified missing markers) as a match. Defaults to False.
    - **normalizer** (`Optional[Callable[[Any], Any]]`, optional): An optional function to normalize both the expected value and the actual value before comparison. This can be used to implement case-insensitive matching, for example. When using CONTAINS mode, the normalizer is applied to each element of the collection individually. Defaults to None (no normalization).
    - **missing_markers** (`Tuple[Any, ...]`, optional): A tuple of values that should be treated as missing when `allow_missing` is True. Defaults to (None, "", "_").

    ## Mode semantics:
    - **EXACT**: The actual attribute value must be exactly equal to ``value``.
    - **NEGATION**: The actual attribute value must not be equal to ``value``.
    - **WILDCARD**: Any attribute value matches (``value`` must be None).
    - **MEMBERSHIP**: The actual (scalar) attribute value must be in the iterable ``value``.
      The *condition* holds a collection; the *attribute* is scalar.
    - **CONTAINS**: The (scalar) ``value`` must be found among the elements or keys of
      a collection-valued attribute.  For dict-valued attributes, CONTAINS iterates over
      the **keys** (not the values); for list, tuple, set, and frozenset attributes it
      iterates over the elements.  Strings are treated as scalar values and will never
      match in CONTAINS mode (use EXACT for string equality).  The *attribute* holds a
      collection; the *condition* is scalar.  This is the logical reverse of MEMBERSHIP.

    ## Methods:
    - :func:`~ValueCondition.matches`: Checks whether a given actual value satisfies this condition.
    - :func:`~ValueCondition.describe`: Returns a human-readable explanation of the condition.
    """

    mode: ConditionMode
    value: Any = None
    allow_missing: bool = False
    normalizer: Optional[Callable[[Any], Any]] = None
    missing_markers: Tuple[Any, ...] = DEFAULT_MISSING_MARKERS

    def __post_init__(self: Self) -> None:
        """
        Validate config and pre-normalise expected value once.
        """
        self._validate_or_raise()

        if (
            self.normalizer is not None
            and self.value is not None
            and self.mode
            not in (
                ConditionMode.WILDCARD,
                ConditionMode.MEMBERSHIP,
                ConditionMode.NOT_MEMBERSHIP,
            )
        ):
            # dataclass is frozen, so we use object.__setattr__
            object.__setattr__(self, "value", self.normalizer(self.value))

    def matches(self: Self, actual_value: Any) -> bool:
        """
        Check whether `actual_value` satisfies this condition.

        Args:
            actual_value (Any): The value to check against this condition.

        Returns:
            bool: True if the actual value satisfies the condition, False otherwise.
        """
        if self.mode is ConditionMode.WILDCARD:
            return True

        if self._is_missing(actual_value):
            return self.allow_missing

        # CONTAINS handles normalization per-element internally, so we branch
        # before the whole-value normalizer is applied.
        if self.mode is ConditionMode.CONTAINS:
            return self._matches_contains(actual_value)

        if self.normalizer is not None:
            actual_value = self.normalizer(actual_value)

        if self.mode is ConditionMode.EXACT:
            return actual_value == self.value
        if self.mode is ConditionMode.NEGATION:
            return actual_value != self.value
        if self.mode is ConditionMode.MEMBERSHIP:
            return actual_value in self.value
        if self.mode is ConditionMode.NOT_MEMBERSHIP:
            return actual_value not in self.value

        # Defensive fallback; should be unreachable due to validation.
        raise ValueError(f"Unsupported mode: {self.mode}")

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of the condition.
        """
        if self.mode is ConditionMode.EXACT:
            return f"Value must be exactly {self.value!r}"
        if self.mode is ConditionMode.NEGATION:
            return f"Value must not be {self.value!r}"
        if self.mode is ConditionMode.WILDCARD:
            return "Value can be any value"
        if self.mode is ConditionMode.MEMBERSHIP:
            return f"Value must be in {self.value!r}"
        if self.mode is ConditionMode.NOT_MEMBERSHIP:
            return f"Value must not be in {self.value!r}"
        if self.mode is ConditionMode.CONTAINS:
            return f"Collection must contain {self.value!r}"
        raise ValueError(f"Unsupported mode: {self.mode}")

    def _matches_contains(self: Self, actual_value: Any) -> bool:
        """
        Check whether the condition value is found among the elements or keys
        of a collection-valued attribute.

        For dict-valued attributes, this iterates over the **keys** (not the
        values) and checks whether any key equals the condition value.  For
        list, tuple, set, and frozenset attributes, it iterates over the
        elements.  Strings are treated as scalar values and will never match
        in CONTAINS mode (use EXACT for string equality).

        If a normalizer is provided, it is applied to each element individually
        before comparison with the (already pre-normalized) condition value.

        Args:
            actual_value (Any): The collection-valued attribute value to search.

        Returns:
            bool: True if any element/key matches the condition value, False otherwise.
        """
        # Determine what to iterate over based on the type of actual_value
        if isinstance(actual_value, dict):
            # For dicts, iterate over keys (mirrors Python's ``value in dict``)
            elements = actual_value.keys()
        elif isinstance(actual_value, (list, tuple, set, frozenset)):
            # For list/tuple/set/frozenset, iterate over elements
            elements = actual_value
        else:
            # Scalars (including str) cannot "contain" anything meaningful
            # in the collection sense.  Return False rather than raising.
            return False

        for element in elements:
            norm_element = (
                self.normalizer(element) if self.normalizer is not None else element
            )
            if norm_element == self.value:
                return True

        return False

    def _is_missing(self: Self, value: Any) -> bool:
        """
        Return True when value should be treated as missing.

        Args:
            value (Any): The value to check for missingness.
        Returns:
            bool: True if the value should be treated as missing, False otherwise.
        """
        return value in self.missing_markers

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments and raise explicit errors.
        """
        if not isinstance(self.mode, ConditionMode):
            raise TypeError(
                "mode must be ConditionMode (EXACT, NEGATION, WILDCARD, MEMBERSHIP, or CONTAINS)."
            )

        if self.mode in (
            ConditionMode.EXACT,
            ConditionMode.NEGATION,
            ConditionMode.CONTAINS,
        ):
            if self.value is None:
                raise ValueError(
                    "value is required for EXACT, NEGATION, and CONTAINS modes."
                )

        if self.mode is ConditionMode.WILDCARD and self.value is not None:
            raise ValueError("value must be None when mode is WILDCARD.")

        if self.mode in (ConditionMode.MEMBERSHIP, ConditionMode.NOT_MEMBERSHIP):
            if self.value is None:
                raise ValueError(
                    "value is required for MEMBERSHIP / NOT_MEMBERSHIP mode."
                )
            # Check if value is iterable (but not string)
            if isinstance(self.value, str):
                raise TypeError(
                    "value for MEMBERSHIP / NOT_MEMBERSHIP mode must be an iterable (list, tuple, set) but not a string."
                )
            try:
                iter(self.value)
            except TypeError:
                raise TypeError(
                    f"value for MEMBERSHIP / NOT_MEMBERSHIP mode must be iterable, got {type(self.value).__name__}."
                )

        if self.normalizer is not None and not callable(self.normalizer):
            raise TypeError("normalizer must be callable or None.")


@dataclass(frozen=True, slots=True)
class FeatureCondition:
    """
    Match a dictionary of features using exact, negation, or wildcard logic.
    ## Attributes:
    - **mode** (`ConditionMode`): The matching mode to use (EXACT, NEGATION, or WILDCARD).
    - **required** (`Optional[Dict[str, Any]]`): A dictionary of feature keys and their expected values that must be present for the condition to match. When `mode` is EXACT, all `required` pairs must be present and equal. When `mode` is NEGATION, reject if all `required` pairs match simultaneously.
    Defaults to None.
    - **forbidden** (`Optional[Dict[str, Any]]`): A dictionary of feature keys and their values that must not be present for the condition to match. When `mode` is EXACT, all `forbidden` pairs must not be present with equal value. When `mode` is NEGATION, reject if any `forbidden` pair appears with equal value.
    Defaults to None.
    - **allow_extra_keys** (`bool`, optional): Whether to allow extra keys in the actual features that are not specified in either `required` or `forbidden`. When `mode` is EXACT and `allow_extra_keys` is False, no keys outside union of `required` and `forbidden` are allowed. When `mode` is NEGATION, `allow_extra_keys` has no effect since we only check the specified keys.
    Defaults to False.
    - **allow_missing** (`bool`, optional): Whether to allow missing keys (i.e., keys that are specified in `required` but not present in the actual features) as a match.
    Defaults to False.
    - **normalizer** (`Optional[Callable[[Any], Any]]`, optional): An optional function to normalize both the expected values and the actual values before comparison. This can be used to implement case-insensitive matching, for example.
    Defaults to None (no normalization).
    ## Methods:
    - :func:`~FeatureCondition.matches`: Checks whether a given actual features dictionary satisfies this condition.
    - :func:`~FeatureCondition.describe`: Returns a human-readable explanation of the condition.
    """

    mode: ConditionMode
    required: Optional[Dict[str, Any]] = None
    forbidden: Optional[Dict[str, Any]] = None
    allow_extra_keys: Optional[bool] = False
    allow_missing: Optional[bool] = False
    normalizer: Optional[Callable[[Any], Any]] = None

    def __post_init__(self: Self) -> None:
        """
        Validate config and pre-normalise expected values once.
        """
        self._validate_or_raise()

        if self.normalizer is not None:
            if self.required is not None:
                # dataclass is frozen, so we use object.__setattr__
                object.__setattr__(
                    self,
                    "required",
                    {k: self.normalizer(v) for k, v in self.required.items()},
                )
            if self.forbidden is not None:
                # dataclass is frozen, so we use object.__setattr__
                object.__setattr__(
                    self,
                    "forbidden",
                    {k: self.normalizer(v) for k, v in self.forbidden.items()},
                )

    def matches(self: Self, actual_value: Dict[str, Any] | None) -> bool:
        """
        Check whether `actual_value` satisfies this condition.

        Args:
            actual_value (Dict[str, Any] | None): The value to check against this condition.

        Returns:
            bool: True if the actual value satisfies the condition, False otherwise.
        """
        if self.mode is ConditionMode.WILDCARD:
            return True

        if not isinstance(actual_value, dict):
            return False

        def norm(v: Any) -> Any:
            """
            Apply normalizer if defined, otherwise return value as is.

            Args:
                v (Any): The value to normalize.

            Returns:
                Any: The normalized value if normalizer is defined, otherwise the original value.
            """
            return self.normalizer(v) if self.normalizer is not None else v

        required = self.required or {}
        forbidden = self.forbidden or {}

        if self.mode is ConditionMode.EXACT:
            # Required checks
            for key, expected in required.items():
                if key not in actual_value:
                    if not self.allow_missing:
                        return False
                    continue
                if norm(actual_value[key]) != expected:
                    return False

            # Forbidden checks
            for key, forbidden_value in forbidden.items():
                if key in actual_value and norm(actual_value[key]) == forbidden_value:
                    return False

            # Extra-key policy check
            if not self.allow_extra_keys:
                allowed_keys = set(required.keys()) | set(forbidden.keys())
                if any(key not in allowed_keys for key in actual_value.keys()):
                    return False

            return True

        if self.mode is ConditionMode.NEGATION:
            # Negate required pattern: if full required pattern matches, reject
            if required:
                full_required_match = True
                for key, expected in required.items():
                    if key not in actual_value or norm(actual_value[key]) != expected:
                        full_required_match = False
                        break
                if full_required_match:
                    return False

            # Forbidden still rejects if any forbidden pair matches
            for key, forbidden_value in forbidden.items():
                if key in actual_value and norm(actual_value[key]) == forbidden_value:
                    return False

            return True

        if self.mode is ConditionMode.MEMBERSHIP:
            # At least one required pair must match (any-match semantics)
            if required:
                any_match = False
                for key, expected in required.items():
                    if key not in actual_value:
                        continue
                    if norm(actual_value[key]) == expected:
                        any_match = True
                        break
                if not any_match:
                    return False

            # Forbidden still rejects if any forbidden pair matches
            for key, forbidden_value in forbidden.items():
                if key in actual_value and norm(actual_value[key]) == forbidden_value:
                    return False

            return True

        raise ValueError(f"Unsupported mode: {self.mode}")

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of the condition.
        """
        if self.mode is ConditionMode.EXACT:
            return f"Features must include {self.required!r} and exclude {self.forbidden!r}"
        if self.mode is ConditionMode.NEGATION:
            return f"Features must not simultaneously match all of {self.required!r}; and must not include any of {self.forbidden!r}"
        if self.mode is ConditionMode.MEMBERSHIP:
            parts = []
            if self.required:
                parts.append(f"at least one of {self.required!r} must be present")
            if self.forbidden:
                parts.append(f"none of {self.forbidden!r} may be present")
            return (
                "Features: " + "; and ".join(parts)
                if parts
                else "Features: membership with no constraints"
            )
        if self.mode is ConditionMode.WILDCARD:
            return "Features can be any value"
        raise ValueError(f"Unsupported mode: {self.mode}")

    def _validate_or_raise(self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.mode, ConditionMode):
            raise TypeError("mode must be ConditionMode.")

        if self.required is not None and not isinstance(self.required, dict):
            raise TypeError("required must be dict or None.")

        if self.forbidden is not None and not isinstance(self.forbidden, dict):
            raise TypeError("forbidden must be dict or None.")

        if self.normalizer is not None and not callable(self.normalizer):
            raise TypeError("normalizer must be callable or None.")

        if self.mode in (
            ConditionMode.EXACT,
            ConditionMode.NEGATION,
            ConditionMode.MEMBERSHIP,
        ):
            if self.required is None and self.forbidden is None:
                raise ValueError(
                    "Provide required and/or forbidden for EXACT/NEGATION/MEMBERSHIP."
                )

        if self.mode is ConditionMode.WILDCARD:
            if self.required is not None or self.forbidden is not None:
                raise ValueError("required/forbidden must be None for WILDCARD mode.")

        if self.mode is ConditionMode.CONTAINS:
            raise ValueError(
                "CONTAINS mode is not supported by FeatureCondition. "
                "CONTAINS is designed for scalar values checked against "
                "collection-valued attributes (ValueCondition). "
                "For FeatureCondition, use MEMBERSHIP mode for any-match "
                "semantics (at least one required pair must match) or "
                "EXACT mode for all-match semantics (all required pairs must match)."
            )


@dataclass(frozen=True, slots=True)
class NodeConstraint:
    """
    Constraint for a single node in the dependency graph, used for matching nodes during feature extraction.

    All scalar attribute conditions are specified via ``attribute_conditions``: a dictionary
    that maps attribute names to ``ValueCondition`` objects. At match time each key is used
    as a ``getattr()`` lookup on the node annotation span, and the retrieved value is tested
    against the corresponding condition. This makes the constraint extensible to any attribute
    on the annotation layer without requiring new dataclass fields.

    ## Attributes:
    - **role** (`str`): The role of the node in the dependency chain (e.g., "self", "parent", "child", "sibling", etc.).
    - **attribute_conditions** (`Optional[Dict[str, ValueCondition]]`): An optional dictionary mapping attribute names to `ValueCondition` objects. These are intended only for scalar attributes (e.g. `upostag`, `lemma`, `deprel`). Do not use `attribute_conditions` for dict-valued attributes such as `feats` — use `feats_condition` instead. Each key is an attribute name that will be looked up on the node annotation via ``getattr(node_annotation, key, None)``, and the retrieved value is matched against the corresponding condition.
    - **feats_condition** (`Optional[FeatureCondition]`): An optional FeatureCondition to match the morphological features (feats) of the node. This remains a dedicated field because ``feats`` is a dictionary that requires ``FeatureCondition`` (with required/forbidden semantics), not ``ValueCondition``.
    - **extra_predicates** (`Optional[Tuple[NodePredicate, ...]]`): An optional tuple of additional callables that take the node annotation as input and return a boolean indicating whether the node satisfies some custom condition. These can be used for more complex checks that are not easily expressed with the other conditions.

    ## Methods:
    - :func:`~NodeConstraint.matches`: Checks whether a given node annotation satisfies all the specified conditions in this constraint.
    - :func:`~NodeConstraint.score_selectivity`: Calculates a heuristic selectivity score for this constraint, which can be used to prioritize more selective constraints during matching.
    - :func:`~NodeConstraint.describe`: Returns a human-readable explanation of this node constraint, including the role and the specified conditions.
    """

    role: str
    attribute_conditions: Optional[Dict[str, ValueCondition]] = None
    feats_condition: Optional[FeatureCondition] = None
    extra_predicates: Optional[Tuple[NodePredicate, ...]] = None

    def __post_init__(self: Self) -> None:
        """
        Validate config and pre-normalise expected values once.
        """
        self._validate_or_raise()

    def matches(self: Self, node_annotation: estnltk.Span) -> bool:
        """
        Check whether the given node annotation satisfies this constraint.

        Each key in ``attribute_conditions`` is resolved via
        ``getattr(node_annotation, key, None)`` and the resulting value is
        tested against the corresponding ``ValueCondition``.

        Args:
            node_annotation (estnltk.Span): The estnltk Span annotation of the node to check against this constraint.

        Returns:
            bool: True if the node annotation satisfies all specified conditions,
            otherwise False. Conditions that are None are ignored.
        """
        if self.attribute_conditions:
            for attr_name, condition in self.attribute_conditions.items():
                actual_value = getattr(node_annotation, attr_name, None)
                if not condition.matches(actual_value):
                    return False
        if self.feats_condition:
            feats = getattr(node_annotation, "feats", None)
            if not self.feats_condition.matches(feats):
                return False
        if self.extra_predicates:
            for pred in self.extra_predicates:
                if not pred(node_annotation):
                    return False
        return True

    def score_selectivity(self: Self) -> float:
        """
        Calculate a heuristic selectivity score for this constraint, which can be used to prioritize more selective constraints during matching.

        Returns:
            float: A selectivity score where higher values indicate more selective constraints. The score is calculated based on the number and restrictiveness of the specified conditions. For example, an EXACT ValueCondition is more selective than a NEGATION, and both are more selective than a WILDCARD. Similarly, having multiple conditions (e.g., UPOS, lemma, feats) increases selectivity compared to having only one or none.
        """
        score = 0.0
        # Exact > Membership ≈ Contains > Negation > Wildcard(0.0) in terms of selectivity
        if self.attribute_conditions:
            for cond in self.attribute_conditions.values():
                if cond.mode == ConditionMode.EXACT:
                    score += SELECTIVITY_WEIGHT_EXACT
                elif cond.mode == ConditionMode.MEMBERSHIP:
                    score += SELECTIVITY_WEIGHT_MEMBERSHIP
                elif cond.mode == ConditionMode.NOT_MEMBERSHIP:
                    score += SELECTIVITY_WEIGHT_MEMBERSHIP
                elif cond.mode == ConditionMode.CONTAINS:
                    score += SELECTIVITY_WEIGHT_CONTAINS
                elif cond.mode == ConditionMode.NEGATION:
                    score += SELECTIVITY_WEIGHT_NEGATION
        if self.feats_condition is not None:
            if self.feats_condition.mode == ConditionMode.EXACT:
                score += SELECTIVITY_WEIGHT_EXACT
            elif self.feats_condition.mode == ConditionMode.MEMBERSHIP:
                score += SELECTIVITY_WEIGHT_MEMBERSHIP
            elif self.feats_condition.mode == ConditionMode.NOT_MEMBERSHIP:
                score += SELECTIVITY_WEIGHT_MEMBERSHIP
            elif self.feats_condition.mode == ConditionMode.NEGATION:
                score += SELECTIVITY_WEIGHT_NEGATION
        if self.extra_predicates:
            score += SELECTIVITY_WEIGHT_EXTRA_PREDICATE * len(self.extra_predicates)

        return score

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of this node constraint, including the role and the specified conditions.

        Returns:
            str: A human-readable string describing this node constraint, including the role and the details of each specified condition. This can be used for debugging or for explaining why a particular node did or did not match this constraint.
        """
        parts = [f"Role: {self.role}"]
        if self.attribute_conditions:
            for attr_name, condition in self.attribute_conditions.items():
                parts.append(f"Attribute '{attr_name}': {condition.describe()}")
        if self.feats_condition:
            parts.append(f"Feats: {self.feats_condition.describe()}")
        if self.extra_predicates:
            parts.append(
                f"Extra predicates: {len(self.extra_predicates)} predicates defined"
            )
        return "; ".join(parts)

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.role, str) or self.role.strip() == "":
            raise TypeError("role must be a non-empty string.")

        if self.attribute_conditions is not None:
            if not isinstance(self.attribute_conditions, dict):
                raise TypeError(
                    "attribute_conditions must be a Dict[str, ValueCondition] or None."
                )
            for key, cond in self.attribute_conditions.items():
                if not isinstance(key, str) or key.strip() == "":
                    raise TypeError(
                        "Each key in attribute_conditions must be a non-empty string."
                    )
                if not isinstance(cond, ValueCondition):
                    raise TypeError(
                        f"Each value in attribute_conditions must be a ValueCondition, "
                        f"got {type(cond).__name__} for key '{key}'."
                    )
                # Disallow dict-valued expected values in attribute_conditions to
                # avoid confusing use for dict-like attributes (e.g. `feats`).
                # Users should place multi-key feature constraints in `feats_condition`.
                if getattr(cond, "value", None) is not None and isinstance(
                    cond.value, dict
                ):
                    raise ValueError(
                        f"attribute_conditions entry '{key}' has a dict-valued expected "
                        "value; use feats_condition (FeatureCondition) for dict-valued attributes."
                    )
            # Reject attribute names that must use a different condition type
            # or are handled by dedicated non-condition fields.
            overlapping = set(self.attribute_conditions.keys()) & set(
                RESERVED_NODE_ATTRIBUTE_NAMES.keys()
            )
            if overlapping:
                details = {
                    attr: RESERVED_NODE_ATTRIBUTE_NAMES[attr] for attr in overlapping
                }
                raise ValueError(
                    f"attribute_conditions keys {overlapping} are reserved. "
                    f"These attributes require a different condition type or are "
                    f"handled by dedicated fields: {details}. "
                    f"Use the dedicated fields instead."
                )

        if self.feats_condition is not None and not isinstance(
            self.feats_condition, FeatureCondition
        ):
            raise TypeError("feats_condition must be FeatureCondition or None.")

        if self.extra_predicates is not None:
            if not isinstance(self.extra_predicates, tuple):
                raise TypeError(
                    "extra_predicates must be a tuple of callables or None."
                )
            for pred in self.extra_predicates:
                if not callable(pred):
                    raise TypeError("Each item in extra_predicates must be callable.")


@dataclass(frozen=True, slots=True)
class EdgeConstraint:
    """
    A constraint for filtering edges in the syntax graph based on their properties.

    Scalar attribute conditions on the ``EdgeContext`` are specified via
    ``attribute_conditions``: a dictionary that maps attribute names to
    ``ValueCondition`` objects. At match time each key is used as a
    ``getattr()`` lookup on the edge context, and the retrieved value is tested
    against the corresponding condition. This replaces the former dedicated
    ``deprel_condition`` field; use ``attribute_conditions={"deprel": ...}``
    instead. Structural fields (``direction``, ``hops``, ``crosses_sentence``)
    remain as dedicated dataclass fields because they use range/enum logic
    rather than simple value matching.

    ## Attributes:
    - **direction** (`DirectionMode`): The direction of the edge to consider (up, down, or both).
    - **attribute_conditions** (`Optional[Dict[str, ValueCondition]]`): An optional dictionary mapping attribute names on the EdgeContext to `ValueCondition` objects. These are intended only for scalar edge attributes (e.g. `deprel`). Do not use `attribute_conditions` for dict-valued edge attributes; prefer a dedicated structure if your edge model includes dict-valued fields. Each key is an attribute name that will be looked up on the edge context via ``getattr(edge_context, key, None)``, and the retrieved value is matched against the corresponding condition.
    - **min_hops** (`Optional[int]`): The minimum number of hops (edges) to traverse in the specified direction for this constraint to apply. Defaults to 1.
    - **max_hops** (`Optional[int]`): The maximum number of hops (edges) to traverse in the specified direction for this constraint to apply. Defaults to 1.
    ## Methods:
    - :func:`~EdgeConstraint.matches`: Checks whether a given edge context satisfies this constraint.
    - :func:`~EdgeConstraint.describe`: Returns a human-readable explanation of this edge constraint, including the direction, attribute conditions, hop range, and other settings.
    """

    direction: DirectionMode
    attribute_conditions: Optional[Dict[str, ValueCondition]] = None
    min_hops: Optional[int] = 1
    max_hops: Optional[int] = 1

    def __post_init__(self: Self) -> None:
        """
        Validate config once.
        """
        self._validate_or_raise()

    def matches(self: Self, edge_context: EdgeContext) -> bool:
        """
        Check whether the given edge context satisfies this constraint.

        Each key in ``attribute_conditions`` is resolved via
        ``getattr(edge_context, key, None)`` and the resulting value is
        tested against the corresponding ``ValueCondition``.

        Args:
            edge_context (EdgeContext): The context of the edge to check against this constraint, including its direction, deprel, hop count, and whether it crosses sentence boundaries.

        Returns:
            bool: True if the edge context satisfies this constraint, False otherwise.
        """
        # Check attribute conditions (e.g. deprel, or any future edge attribute)
        if self.attribute_conditions:
            for attr_name, condition in self.attribute_conditions.items():
                actual_value = getattr(edge_context, attr_name, None)
                if not condition.matches(actual_value):
                    return False
        # Check direction
        # If BOTH, we allow any direction, so no check needed. Otherwise, the edge's direction must match the specified direction.
        if (
            self.direction != DirectionMode.BOTH
            and edge_context.direction != self.direction
        ):
            return False
        # Check hop bounds
        if self.min_hops is not None and edge_context.hops < self.min_hops:
            return False
        if self.max_hops is not None and edge_context.hops > self.max_hops:
            return False
        return True

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of this edge constraint, including the direction, attribute conditions, hop range, and other settings.

        Returns:
            str: A human-readable string describing this edge constraint, including the direction, attribute conditions, hop range, and other settings. This can be used for debugging or for explaining why a particular edge did or did not match this constraint.
        """
        parts = [f"Direction: {self.direction.value}"]
        if self.attribute_conditions:
            for attr_name, condition in self.attribute_conditions.items():
                parts.append(f"Attribute '{attr_name}': {condition.describe()}")
        if self.min_hops is not None or self.max_hops is not None:
            parts.append(f"Hops: {self.min_hops or 0} to {self.max_hops or '∞'}")
        return "; ".join(parts)

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.direction, DirectionMode):
            raise TypeError("direction must be an instance of DirectionMode.")

        if self.attribute_conditions is not None:
            if not isinstance(self.attribute_conditions, dict):
                raise TypeError(
                    "attribute_conditions must be a Dict[str, ValueCondition] or None."
                )
            for key, cond in self.attribute_conditions.items():
                if not isinstance(key, str) or key.strip() == "":
                    raise TypeError(
                        "Each key in attribute_conditions must be a non-empty string."
                    )
                if not isinstance(cond, ValueCondition):
                    raise TypeError(
                        f"Each value in attribute_conditions must be a ValueCondition, "
                        f"got {type(cond).__name__} for key '{key}'."
                    )
                # Disallow dict-valued expected values in attribute_conditions to
                # avoid confusing use for dict-like attributes.
                if getattr(cond, "value", None) is not None and isinstance(
                    cond.value, dict
                ):
                    raise ValueError(
                        f"attribute_conditions entry '{key}' has a dict-valued expected "
                        "value; attribute_conditions are for scalar edge attributes."
                    )
            # Reject attribute names that are handled by dedicated structural
            # fields (direction, hops) which use range/enum logic rather than
            # simple value matching.
            overlapping = set(self.attribute_conditions.keys()) & set(
                RESERVED_EDGE_ATTRIBUTE_NAMES.keys()
            )
            if overlapping:
                details = {
                    attr: RESERVED_EDGE_ATTRIBUTE_NAMES[attr] for attr in overlapping
                }
                raise ValueError(
                    f"attribute_conditions keys {overlapping} are reserved. "
                    f"These attributes are handled by dedicated structural fields: "
                    f"{details}. Use the dedicated fields instead."
                )

        if self.min_hops is not None:
            if not isinstance(self.min_hops, int) or self.min_hops < 0:
                raise ValueError("min_hops must be a non-negative integer or None.")

        if self.max_hops is not None:
            if not isinstance(self.max_hops, int) or self.max_hops < 0:
                raise ValueError("max_hops must be a non-negative integer or None.")

        if (
            self.min_hops is not None
            and self.max_hops is not None
            and self.min_hops > self.max_hops
        ):
            raise ValueError("min_hops cannot be greater than max_hops.")
