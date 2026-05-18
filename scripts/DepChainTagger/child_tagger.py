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

from .child_matcher import DepChildMatcher
from .config import (
    DEFAULT_ANCHOR_ROLE,
    DEFAULT_DEDUP_MODE_GLOBAL,
    DEFAULT_DEDUP_MODE_SENTENCE,
    DEFAULT_MAX_MATCHES_PER_SENTENCE,
    DEFAULT_MAX_TOTAL_MATCHES,
    DEFAULT_OUTPUT_ATTRIBUTES,
    DEFAULT_OUTPUT_LAYER_NAME,
    DEFAULT_OUTPUT_SPAN_NAMES,
    DEFAULT_SENTENCES_LAYER_NAME,
    DEFAULT_SYNTAX_LAYER_NAME,
)
from .orchestrator import DepTaggerOrchestrator
from .patterns import PathPattern
from .types import DirectionMode


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


class DepChildTagger(RelationTagger):
    """EstNLTK RelationTagger wrapper for direct-child matches.

    This tagger mirrors `DepChainTagger`, but delegates matching to
    `DepChildMatcher`.  The output format stays compatible with the same
    decorator and relation-layer structure.
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
        "_depchild_tagger",
        "_pattern_by_name",
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
        self.output_span_names = DEFAULT_OUTPUT_SPAN_NAMES
        self.output_attributes = (
            output_attributes
            if output_attributes is not None
            else DEFAULT_OUTPUT_ATTRIBUTES
        )

        self._depchild_tagger = DepTaggerOrchestrator(
            patterns=patterns,
            matcher=DepChildMatcher(
                patterns=patterns,
                dedup_mode=sentence_match_dedup_mode,
                max_matches_per_sentence=max_matches_per_sentence,
                allow_role_node_overlap=allow_role_node_overlap,
            ),
            sentence_match_dedup_mode=sentence_match_dedup_mode,
            max_matches_per_sentence=max_matches_per_sentence,
            allow_role_node_overlap=allow_role_node_overlap,
            global_dedup_mode=global_dedup_mode,
            max_total_matches=max_total_matches,
        )

        seen_pattern_names: set[str] = set()
        for pattern in patterns:
            # Validate that all edge constraints use DOWN direction only
            for edge in pattern.edge_steps:
                # EdgeConstraint.direction is an enum; reject anything not DOWN
                if getattr(edge, "direction", None) != DirectionMode.DOWN:
                    raise ValueError(
                        f"DepChildTagger only supports DOWN edge constraints; pattern {pattern.name!r} has edge with direction {getattr(edge, 'direction', None)!r}."
                    )
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

            matches = self._depchild_tagger.tag_sentence_layers(
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
                    f"DepChildTagger matching failed: {str(exc)}"
                )
            return layer

        return layer

    def _add_match_to_layer(self: Self, layer: Any, match: Any) -> None:
        """Add one matched relation and one annotation per matched token."""
        pattern = self._pattern_by_name.get(match.pattern_name)
        emit_roles = (
            tuple(pattern.emit_roles)
            if pattern and pattern.emit_roles
            else tuple(match.role_to_node.keys())
        )

        anchor_role = (
            pattern.anchor_role
            if pattern and pattern.anchor_role in match.role_to_node
            else None
        )
        if anchor_role is None:
            anchor_role = (
                DEFAULT_ANCHOR_ROLE
                if DEFAULT_ANCHOR_ROLE in match.role_to_node
                else next(iter(match.role_to_token_id.keys()), None)
            )

        nodes = [
            match.role_to_node[role]
            for role in emit_roles
            if role in match.role_to_node
        ]
        if not nodes:
            return

        token_spans: Dict[str, Tuple[int, int]] = {}
        for role in emit_roles:
            node = match.role_to_node.get(role)
            if node is None:
                continue
            start = getattr(node, "start", None)
            end = getattr(node, "end", None)
            if start is None or end is None:
                raise ValueError(
                    f"Cannot add relation span for role {role!r}: node is missing start/end offsets."
                )
            token_spans[role] = (int(start), int(end))

        for role in emit_roles:
            node = match.role_to_node.get(role)
            if node is None:
                continue
            token_span = token_spans.get(role)
            annotation_payload: Dict[str, Any] = {
                "text": token_span,
                "pattern_name": match.pattern_name,
                "matched_text": getattr(match, "matched_text", None),
                "upostag": getattr(node, "upostag", None),
                "xpostag": getattr(node, "xpostag", None),
                "feats": getattr(node, "feats", None),
                "lemma": getattr(node, "lemma", None),
                "deprel": getattr(node, "deprel", None),
                "role": role,
                "is_anchor": role == anchor_role,
            }
            layer.add_annotation(annotation_payload)
