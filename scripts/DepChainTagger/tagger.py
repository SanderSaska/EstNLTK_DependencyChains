from typing import (
    Any,
    Dict,
    Literal,
    List,
    Optional,
    Self,
    Tuple,
    cast,
)
import hashlib
import re

from estnltk import Text
from estnltk.taggers import RelationTagger
from estnltk_core import RelationLayer

from .config import (
    DEFAULT_DEDUP_MODE_GLOBAL,
    DEFAULT_DEDUP_MODE_SENTENCE,
    DEFAULT_MAX_MATCHES_PER_SENTENCE,
    DEFAULT_MAX_TOTAL_MATCHES,
    DEFAULT_OUTPUT_LAYER_NAME,
    DEFAULT_SENTENCES_LAYER_NAME,
    DEFAULT_SYNTAX_LAYER_NAME,
)
from .orchestrator import DepTaggerOrchestrator
from .patterns import PathPattern


def _deterministic_hash(items: Tuple[Any, ...]) -> str:
    """Return a short stable hash for match identifiers."""
    serialized = str(items).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def _make_identifier(value: str) -> str:
    """Sanitise a string into a RelationLayer-safe identifier."""
    ident = re.sub(r"\W+", "_", value).strip("_")
    if not ident:
        ident = "span"
    if ident[0].isdigit():
        ident = f"_{ident}"
    return ident


def _append_unique(values: List[str], seen: set[str], item: str) -> None:
    """Append a string once while preserving first-seen order."""
    if item in seen:
        return
    seen.add(item)
    values.append(item)


def _collect_role_span_names(patterns: Tuple[PathPattern, ...]) -> Tuple[str, ...]:
    """Collect unique role names across all patterns in first-seen order."""
    span_names: List[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for node_constraint in pattern.node_steps:
            _append_unique(span_names, seen, node_constraint.role)
    return tuple(span_names)


def _collect_output_attribute_names(patterns: Tuple[PathPattern, ...]) -> Tuple[str, ...]:
    """Collect the flattened metadata fields needed to render all patterns."""
    attribute_names: List[str] = ["pattern_name", "matched_text"]
    seen: set[str] = set(attribute_names)

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

        for index, edge_constraint in enumerate(pattern.edge_steps):
            source_role = pattern.node_steps[index].role
            _append_unique(attribute_names, seen, f"{source_role}_direction")
            _append_unique(attribute_names, seen, f"{source_role}_min_hops")
            _append_unique(attribute_names, seen, f"{source_role}_max_hops")
            if edge_constraint.attribute_conditions:
                for attr_name in edge_constraint.attribute_conditions:
                    _append_unique(attribute_names, seen, f"{source_role}_{attr_name}")

    return tuple(attribute_names)


def _build_match_annotation_payload(
    match: Any,
    patterns_by_name: Dict[str, PathPattern],
    span_names: Tuple[str, ...],
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
        start = getattr(node, "start", None)
        end = getattr(node, "end", None)
        if start is None or end is None:
            raise ValueError(
                f"Cannot add relation span for role {role!r}: node is missing start/end offsets."
            )
        payload[role] = (int(start), int(end))

    for node_constraint in pattern.node_steps:
        prefix = f"{node_constraint.role}_"
        if node_constraint.attribute_conditions:
            for attr_name, condition in node_constraint.attribute_conditions.items():
                if condition.mode is not None and condition.mode.name == "WILDCARD":
                    continue
                if (
                    condition.mode.name == "EXACT"
                    and condition.allow_missing is False
                    and condition.normalizer is None
                    and condition.missing_markers == (None, "", "_")
                ):
                    payload[f"{prefix}{attr_name}"] = condition.value
                else:
                    payload[f"{prefix}{attr_name}"] = {
                        "mode": condition.mode.value,
                        "value": condition.value,
                        "allow_missing": condition.allow_missing,
                        "missing_markers": condition.missing_markers,
                        "normalizer": getattr(condition.normalizer, "__name__", None),
                    }

        if node_constraint.feats_condition is not None:
            feats_condition = node_constraint.feats_condition
            payload[f"{prefix}feats"] = {
                "mode": feats_condition.mode.value,
                "required": feats_condition.required,
                "forbidden": feats_condition.forbidden,
                "allow_extra_keys": feats_condition.allow_extra_keys,
                "allow_missing": feats_condition.allow_missing,
                "missing_markers": feats_condition.missing_markers,
                "normalizer": getattr(feats_condition.normalizer, "__name__", None),
            }

        if node_constraint.extra_predicates:
            payload[f"{prefix}extra_predicates"] = tuple(
                getattr(predicate, "__name__", repr(predicate))
                for predicate in node_constraint.extra_predicates
            )

    for index, edge_constraint in enumerate(pattern.edge_steps):
        source_role = pattern.node_steps[index].role
        payload[f"{source_role}_direction"] = edge_constraint.direction.value
        payload[f"{source_role}_min_hops"] = edge_constraint.min_hops
        payload[f"{source_role}_max_hops"] = edge_constraint.max_hops
        if edge_constraint.attribute_conditions:
            for attr_name, condition in edge_constraint.attribute_conditions.items():
                if condition.mode.name == "WILDCARD":
                    continue
                if (
                    condition.mode.name == "EXACT"
                    and condition.allow_missing is False
                    and condition.normalizer is None
                    and condition.missing_markers == (None, "", "_")
                ):
                    payload[f"{source_role}_{attr_name}"] = condition.value
                else:
                    payload[f"{source_role}_{attr_name}"] = {
                        "mode": condition.mode.value,
                        "value": condition.value,
                        "allow_missing": condition.allow_missing,
                        "missing_markers": condition.missing_markers,
                        "normalizer": getattr(condition.normalizer, "__name__", None),
                    }

    return payload


class DepChainTagger(RelationTagger):
    """EstNLTK RelationTagger wrapper for dependency-chain matches.

    This tagger takes a set of `PathPattern` objects, finds all matches in the input text, and produces a RelationLayer where each relation corresponds to one match. The relation's attributes include token-level information for each role in the match, as well as metadata about the pattern and sentence. The tagger supports configurable deduplication strategies to control the number of matches emitted per sentence and across the entire text.
    """

    conf_param = [
        "patterns",
        "output_layer",
        "output_span_names",
        "output_attributes",
        "sentence_match_dedup_mode",
        "max_matches_per_sentence",
        "allow_role_node_overlap",
        "global_dedup_mode",
        "max_total_matches",
        "_depchain_tagger",
        "_pattern_by_name",
        # "_pattern_span_name_by_pattern",
    ]

    def __init__(
        self: Self,
        patterns: Tuple[PathPattern, ...],
        output_layer: str = DEFAULT_OUTPUT_LAYER_NAME,
        output_attributes: Optional[Tuple[str, ...]] = None,
        sentence_match_dedup_mode: Literal["none", "exact", "role_based"] = cast(
            Literal["none", "exact", "role_based"], DEFAULT_DEDUP_MODE_SENTENCE
        ),
        max_matches_per_sentence: int = DEFAULT_MAX_MATCHES_PER_SENTENCE,
        allow_role_node_overlap: bool = False,
        global_dedup_mode: Literal["none", "exact", "role_based"] = cast(
            Literal["none", "exact", "role_based"], DEFAULT_DEDUP_MODE_GLOBAL
        ),
        max_total_matches: int = DEFAULT_MAX_TOTAL_MATCHES,
    ) -> None:

        self.input_layers = (DEFAULT_SYNTAX_LAYER_NAME, DEFAULT_SENTENCES_LAYER_NAME)
        self.output_layer = output_layer
        self.output_span_names = _collect_role_span_names(patterns)
        self.output_attributes = (
            output_attributes
            if output_attributes is not None
            else _collect_output_attribute_names(patterns)
        )

        self._depchain_tagger = DepTaggerOrchestrator(
            patterns=patterns,
            sentence_match_dedup_mode=sentence_match_dedup_mode,
            max_matches_per_sentence=max_matches_per_sentence,
            allow_role_node_overlap=allow_role_node_overlap,
            global_dedup_mode=global_dedup_mode,
            max_total_matches=max_total_matches,
        )

        seen_pattern_names: set[str] = set()
        for pattern in patterns:
            if pattern.name in seen_pattern_names:
                raise ValueError(
                    f"Duplicate pattern name encountered: {pattern.name!r}"
                )
            seen_pattern_names.add(pattern.name)

        self._pattern_by_name: Dict[str, PathPattern] = {
            pattern.name: pattern for pattern in patterns
        }

    def _make_layer_template(self: Self) -> RelationLayer:
        return RelationLayer(
            name=self.output_layer,
            span_names=self.output_span_names,
            attributes=self.output_attributes,
            display_order=tuple(self.output_span_names) + tuple(self.output_attributes),
            text_object=None,
            ambiguous=True,
        )

    def _make_layer(
        self: Self,
        text: Text,
        layers: Any,
        status: Optional[Dict[str, Any]] = None,
    ) -> RelationLayer:
        layer = self._make_layer_template()
        layer.text_object = text

        if (
            layers is None
            or DEFAULT_SYNTAX_LAYER_NAME not in layers
            or DEFAULT_SENTENCES_LAYER_NAME not in layers
        ):
            return layer

        sentences_layer = layers[DEFAULT_SENTENCES_LAYER_NAME]

        try:
            top_level_syntax = (
                layers.get(DEFAULT_SYNTAX_LAYER_NAME)
                if hasattr(layers, "get")
                else None
            )
            sentence_syntax_layers: List[Any] = []
            sentence_spans = [(s.start, s.end) for s in sentences_layer]
            for sent in sentences_layer:
                try:
                    sent_syntax = sent[DEFAULT_SYNTAX_LAYER_NAME]
                except Exception:
                    if top_level_syntax is None:
                        raise KeyError(DEFAULT_SYNTAX_LAYER_NAME)
                    sent_syntax = [
                        ann
                        for ann in top_level_syntax
                        if getattr(ann, "start", None) is not None
                        and ann.start >= sent.start
                        and ann.end <= sent.end
                    ]
                sentence_syntax_layers.append(sent_syntax)

            matches = self._depchain_tagger.tag_sentence_layers(
                sentence_syntax_layers=sentence_syntax_layers,
                sentence_spans=sentence_spans,
            )

            for match in matches:
                try:
                    self._add_match_to_layer(layer, match)
                except Exception as exc:
                    if status is not None:
                        status.setdefault("errors", []).append(
                            f"Error adding match to layer: {str(exc)}"
                        )

        except Exception as exc:
            if status is not None:
                status.setdefault("errors", []).append(
                    f"DepChainTagger matching failed: {str(exc)}"
                )
            return layer

        return layer

    def _add_match_to_layer(self: Self, layer: Any, match: Any) -> None:
        """Add one matched relation row for the whole pattern match."""
        if match.pattern_name not in self._pattern_by_name:
            raise KeyError(f"Unknown pattern name: {match.pattern_name!r}")

        annotation_payload = _build_match_annotation_payload(
            match=match,
            patterns_by_name=self._pattern_by_name,
            span_names=self.output_span_names,
        )
        layer.add_annotation(annotation_payload)
