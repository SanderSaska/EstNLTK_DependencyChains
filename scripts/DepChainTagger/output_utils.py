from typing import Any, Dict, Iterable, List, Tuple

from .conditions import (
    FeatureCondition,
    NodeConstraint,
    ValueCondition,
)
from .patterns import ChainMatch, PathPattern
from .types import ConditionMode


def _append_unique(values: List[str], seen: set[str], item: str) -> None:
    """Append a string once while preserving first-seen order."""
    if item in seen:
        return
    seen.add(item)
    values.append(item)


def _serialize_value_condition(condition: ValueCondition) -> Any:
    """Serialize a scalar value condition into a tabular-friendly value."""
    if condition.mode is ConditionMode.WILDCARD:
        return "WILDCARD"
    return {
        "mode": condition.mode.value,
        "value": condition.value,
        "allow_missing": condition.allow_missing,
        "missing_markers": condition.missing_markers,
        "normalizer": getattr(condition.normalizer, "__name__", None),
    }


def _serialize_feature_condition(condition: FeatureCondition) -> Dict[str, Any]:
    """Serialize a feature condition into a JSON-friendly dictionary."""
    return {
        "mode": condition.mode.value,
        "required": condition.required,
        "forbidden": condition.forbidden,
        "allow_extra_keys": condition.allow_extra_keys,
        "allow_missing": condition.allow_missing,
        "missing_markers": condition.missing_markers,
        "normalizer": getattr(condition.normalizer, "__name__", None),
    }


def _serialize_node_constraint(node_constraint: NodeConstraint) -> Dict[str, Any]:
    """Flatten node-constraint metadata with a role prefix."""
    payload: Dict[str, Any] = {}
    prefix = f"{node_constraint.role}_"

    if node_constraint.attribute_conditions:
        for attr_name, condition in node_constraint.attribute_conditions.items():
            if isinstance(condition, FeatureCondition):
                value = _serialize_feature_condition(condition)
            else:
                value = _serialize_value_condition(condition)
            if value is not None:
                payload[f"{prefix}{attr_name}"] = value

    # If a dedicated feats_condition exists but wasn't normalised into
    # attribute_conditions, include it as flattened fields as well.
    if node_constraint.feats_condition is not None and not (
        node_constraint.attribute_conditions
        and "feats" in node_constraint.attribute_conditions
    ):
        value = _serialize_feature_condition(node_constraint.feats_condition)
        if value is not None:
            payload[f"{prefix}feats"] = value

    if node_constraint.extra_predicates:
        payload[f"{prefix}extra_predicates"] = tuple(
            getattr(predicate, "__name__", repr(predicate))
            for predicate in node_constraint.extra_predicates
        )

    return payload


# Edge constraint metadata is intentionally not serialized into the
# relation output. Edge constraints are structural and placing their
# metadata on a parent or child role is ambiguous, so we omit them to
# avoid confusion.


def collect_role_span_names(patterns: Iterable[PathPattern]) -> Tuple[str, ...]:
    """Collect unique role names across all patterns in first-seen order."""
    span_names: List[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for node_constraint in pattern.node_steps:
            _append_unique(span_names, seen, node_constraint.role)
    return tuple(span_names)


def collect_output_attribute_names(
    patterns: Iterable[PathPattern], include_pattern_constraints: bool = False
) -> Tuple[str, ...]:
    """Collect the flattened metadata fields needed to render all patterns."""
    attribute_names: List[str] = ["pattern_name", "matched_text"]
    seen: set[str] = set(attribute_names)

    if not include_pattern_constraints:
        return tuple(attribute_names)

    # for pattern in patterns:
    #     for node_constraint in pattern.node_steps:
    #         prefix = f"{node_constraint.role}_"
    #         if (
    #             node_constraint.attribute_conditions
    #             or node_constraint.feats_condition is not None
    #             or node_constraint.extra_predicates
    #         ):
    #             _append_unique(attribute_names, seen, f"{prefix}text")

    for pattern in patterns:
        for node_constraint in pattern.node_steps:
            prefix = f"{node_constraint.role}_"
            if node_constraint.attribute_conditions:
                for attr_name in node_constraint.attribute_conditions:
                    _append_unique(attribute_names, seen, f"{prefix}{attr_name}")
            if node_constraint.feats_condition is not None:
                _append_unique(attribute_names, seen, f"{prefix}feats")
            if node_constraint.extra_predicates:
                _append_unique(attribute_names, seen, f"{prefix}extra_predicates")
        # Note: edge constraint fields (direction/hops/attrs) are not
        # included in the flattened attribute names because edge metadata
        # is not serialized into the relation output.

    return tuple(attribute_names)


def build_match_annotation_payload(
    match: ChainMatch,
    patterns_by_name: Dict[str, PathPattern],
    span_names: Tuple[str, ...],
    include_pattern_constraints: bool = False,
) -> Dict[str, Any]:
    """Build one relation-layer annotation per match."""
    pattern = patterns_by_name.get(match.pattern_name)
    if pattern is None:
        raise KeyError(f"Unknown pattern name: {match.pattern_name!r}")

    payload: Dict[str, Any] = {role: None for role in span_names}
    payload["pattern_name"] = match.pattern_name
    payload["matched_text"] = getattr(match, "matched_text", None)

    for role in span_names:
        node = match.role_to_node.get(role)
        if node is None:
            continue
        node_constraint = pattern.get_node_constraint(role)
        start = getattr(node, "start", None)
        end = getattr(node, "end", None)
        if start is None or end is None:
            raise ValueError(
                f"Cannot add relation span for role {role!r}: node is missing start/end offsets."
            )
        payload[role] = (int(start), int(end))

    if include_pattern_constraints:
        for node_constraint in pattern.node_steps:
            payload.update(_serialize_node_constraint(node_constraint))
        # Edge constraint metadata intentionally not added to payload.

    return payload
