from typing import Any, Dict, Iterable, List, Tuple
from collections.abc import Mapping
import json

from .conditions import (
    EdgeConstraint,
    FeatureCondition,
    NodeConstraint,
    ValueCondition,
    _is_text_scalar,
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
        return None
    # if (
    #     condition.mode is ConditionMode.EXACT
    #     and condition.allow_missing is False
    #     and condition.normalizer is None
    #     and condition.missing_markers == (None, "", "_")
    # ):
    #     return condition.value
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

    def _flatten_value(attr: str, cond: ValueCondition) -> None:
        v = _serialize_value_condition(cond)
        if not isinstance(v, dict):
            payload[f"{prefix}{attr}"] = v
            return
        payload[f"{prefix}{attr}_mode"] = v.get("mode")
        payload[f"{prefix}{attr}_value"] = v.get("value")
        payload[f"{prefix}{attr}_allow_missing"] = v.get("allow_missing")
        payload[f"{prefix}{attr}_missing_markers"] = v.get("missing_markers")
        payload[f"{prefix}{attr}_normalizer"] = v.get("normalizer")

    def _flatten_feature(attr: str, cond: FeatureCondition) -> None:
        d = _serialize_feature_condition(cond)
        payload[f"{prefix}{attr}_mode"] = d.get("mode")
        payload[f"{prefix}{attr}_allow_extra_keys"] = d.get("allow_extra_keys")
        payload[f"{prefix}{attr}_allow_missing"] = d.get("allow_missing")
        payload[f"{prefix}{attr}_missing_markers"] = d.get("missing_markers")
        payload[f"{prefix}{attr}_normalizer"] = d.get("normalizer")

        # Attempt to flatten required/forbidden when they are simple mappings of scalars.
        required = d.get("required")
        if isinstance(required, Mapping):
            if all(_is_text_scalar(val) for val in required.values()):
                for k, val in required.items():
                    payload[f"{prefix}{attr}_required_{k}"] = val
            else:
                payload[f"{prefix}{attr}_required_raw"] = json.dumps(
                    required, ensure_ascii=False
                )
        elif required is not None:
            payload[f"{prefix}{attr}_required_raw"] = json.dumps(
                required, ensure_ascii=False
            )

        forbidden = d.get("forbidden")
        if isinstance(forbidden, Mapping):
            if all(_is_text_scalar(val) for val in forbidden.values()):
                for k, val in forbidden.items():
                    payload[f"{prefix}{attr}_forbidden_{k}"] = val
            else:
                payload[f"{prefix}{attr}_forbidden_raw"] = json.dumps(
                    forbidden, ensure_ascii=False
                )
        elif forbidden is not None:
            payload[f"{prefix}{attr}_forbidden_raw"] = json.dumps(
                forbidden, ensure_ascii=False
            )

    if node_constraint.attribute_conditions:
        for attr_name, condition in node_constraint.attribute_conditions.items():
            if attr_name == "text":
                continue
            if isinstance(condition, FeatureCondition):
                _flatten_feature(attr_name, condition)
            else:
                _flatten_value(attr_name, condition)

    # If a dedicated feats_condition exists but wasn't normalised into
    # attribute_conditions, include it as flattened fields as well.
    if node_constraint.feats_condition is not None and not (
        node_constraint.attribute_conditions
        and "feats" in node_constraint.attribute_conditions
    ):
        _flatten_feature("feats", node_constraint.feats_condition)

    if node_constraint.extra_predicates:
        payload[f"{prefix}extra_predicates"] = tuple(
            getattr(predicate, "__name__", repr(predicate))
            for predicate in node_constraint.extra_predicates
        )

    return payload


def _serialize_edge_constraint(
    source_role: str,
    edge_constraint: EdgeConstraint,
) -> Dict[str, Any]:
    """Flatten edge-constraint metadata onto the source role."""
    payload: Dict[str, Any] = {
        f"{source_role}_direction": edge_constraint.direction.value,
        f"{source_role}_min_hops": edge_constraint.min_hops,
        f"{source_role}_max_hops": edge_constraint.max_hops,
    }

    if edge_constraint.attribute_conditions:
        for attr_name, condition in edge_constraint.attribute_conditions.items():
            v = _serialize_value_condition(condition)
            if not isinstance(v, dict):
                if v is not None:
                    payload[f"{source_role}_{attr_name}"] = v
                continue
            # flatten dict-valued serialization
            payload[f"{source_role}_{attr_name}_mode"] = v.get("mode")
            payload[f"{source_role}_{attr_name}_value"] = v.get("value")
            payload[f"{source_role}_{attr_name}_allow_missing"] = v.get("allow_missing")
            payload[f"{source_role}_{attr_name}_missing_markers"] = v.get(
                "missing_markers"
            )
            payload[f"{source_role}_{attr_name}_normalizer"] = v.get("normalizer")

    return payload


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

    for pattern in patterns:
        for node_constraint in pattern.node_steps:
            prefix = f"{node_constraint.role}_"
            if (
                node_constraint.attribute_conditions
                or node_constraint.feats_condition is not None
                or node_constraint.extra_predicates
            ):
                _append_unique(attribute_names, seen, f"{prefix}text")

    for pattern in patterns:
        for node_constraint in pattern.node_steps:
            prefix = f"{node_constraint.role}_"
            if node_constraint.attribute_conditions:
                for attr_name in node_constraint.attribute_conditions:
                    if attr_name == "text":  # already in one of the role span fields
                        continue
                    _append_unique(attribute_names, seen, f"{prefix}{attr_name}")
            if node_constraint.feats_condition is not None:
                _append_unique(attribute_names, seen, f"{prefix}feats")
            if node_constraint.extra_predicates:
                _append_unique(attribute_names, seen, f"{prefix}extra_predicates")

        for index, edge_constraint in enumerate(pattern.edge_steps):
            source_role = pattern.node_steps[index].role
            _append_unique(attribute_names, seen, f"{source_role}_direction")
            _append_unique(attribute_names, seen, f"{source_role}_min_hops")
            _append_unique(attribute_names, seen, f"{source_role}_max_hops")
            if edge_constraint.attribute_conditions:
                for attr_name in edge_constraint.attribute_conditions:
                    _append_unique(attribute_names, seen, f"{source_role}_{attr_name}")

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
        if node_constraint is not None and (
            node_constraint.attribute_conditions
            or node_constraint.feats_condition is not None
            or node_constraint.extra_predicates
        ):
            payload[f"{role}_text"] = getattr(node, "text", None)
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

        for index, edge_constraint in enumerate(pattern.edge_steps):
            source_role = pattern.node_steps[index].role
            payload.update(_serialize_edge_constraint(source_role, edge_constraint))

    return payload
