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

from estnltk import Text
from estnltk.taggers import RelationTagger
from estnltk_core import RelationLayer

from .child_matcher import DepChildMatcher
from .config import (
    DEFAULT_DEDUP_MODE_GLOBAL,
    DEFAULT_DEDUP_MODE_SENTENCE,
    DEFAULT_MAX_MATCHES_PER_SENTENCE,
    DEFAULT_MAX_TOTAL_MATCHES,
    DEFAULT_OUTPUT_LAYER_NAME,
    DEFAULT_SENTENCES_LAYER_NAME,
    DEFAULT_SYNTAX_LAYER_NAME,
)
from .dep_chain_tagger import AnnotationDecorator
from .orchestrator import DepTaggerOrchestrator
from .patterns import PathPattern
from .output_utils import (
    build_match_annotation_payload,
    collect_output_attribute_names,
    collect_role_span_names,
)
from .types import DirectionMode


class DepChildTagger(RelationTagger):
    """EstNLTK RelationTagger wrapper for direct-child matches.

    Args:
        patterns: Dependency patterns to evaluate for direct-child matches.
        syntax_layer: Name of the syntax layer used as input.
        sentences_layer: Name of the sentence layer used to split the text.
        output_layer: Name of the output relation layer.
        output_attributes: Optional relation attributes to expose.
        annotation_decorator: Optional annotation decorator that can update
            or filter relation payloads before insertion.
        include_pattern_constraints: Whether to flatten pattern constraint
            metadata into the output relation rows.
        max_matches_per_sentence: Maximum number of matches to keep per
            sentence before global aggregation.
        allow_role_node_overlap: Whether the matcher may reuse the same node
            for multiple roles in one pattern.
        max_total_matches: Global cap on the number of accepted matches.
        sentence_match_dedup_mode: Deduplication mode applied within a
            sentence.
        global_dedup_mode: Deduplication mode applied across sentences.

    This tagger mirrors `DepChainTagger`, but delegates matching to
    `DepChildMatcher`. The output format stays compatible with the same
    decorator and relation-layer structure.
    """

    conf_param = [
        "patterns",
        "syntax_layer",
        "sentences_layer",
        "output_layer",
        "output_span_names",
        "output_attributes",
        "annotation_decorator",
        "include_pattern_constraints",
        "max_matches_per_sentence",
        "allow_role_node_overlap",
        "max_total_matches",
        "sentence_match_dedup_mode",
        "global_dedup_mode",
        "_depchild_tagger",
        "_pattern_by_name",
    ]

    def __init__(
        self: Self,
        patterns: Tuple[PathPattern, ...],
        syntax_layer: str = DEFAULT_SYNTAX_LAYER_NAME,
        sentences_layer: str = DEFAULT_SENTENCES_LAYER_NAME,
        output_layer: str = DEFAULT_OUTPUT_LAYER_NAME,
        output_attributes: Optional[Tuple[str, ...]] = None,
        annotation_decorator: Optional[AnnotationDecorator] = None,
        include_pattern_constraints: bool = False,
        max_matches_per_sentence: int = DEFAULT_MAX_MATCHES_PER_SENTENCE,
        allow_role_node_overlap: bool = False,
        max_total_matches: int = DEFAULT_MAX_TOTAL_MATCHES,
        sentence_match_dedup_mode: Literal["none", "exact", "role_based"] = cast(
            Literal["none", "exact", "role_based"], DEFAULT_DEDUP_MODE_SENTENCE
        ),
        global_dedup_mode: Literal["none", "exact", "role_based"] = cast(
            Literal["none", "exact", "role_based"], DEFAULT_DEDUP_MODE_GLOBAL
        ),
    ) -> None:

        self.syntax_layer = syntax_layer
        self.sentences_layer = sentences_layer
        self.input_layers = (self.syntax_layer, self.sentences_layer)
        self.output_layer = output_layer
        self.include_pattern_constraints = include_pattern_constraints
        self.output_span_names = collect_role_span_names(patterns)
        self.output_attributes = (
            output_attributes
            if output_attributes is not None
            else collect_output_attribute_names(
                patterns, self.include_pattern_constraints
            )
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
        self.annotation_decorator = annotation_decorator or AnnotationDecorator()

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
            or self.syntax_layer not in layers
            or self.sentences_layer not in layers
        ):
            return layer

        sentences_layer = layers[self.sentences_layer]

        try:
            top_level_syntax = (
                layers.get(self.syntax_layer) if hasattr(layers, "get") else None
            )
            sentence_syntax_layers: List[Any] = []
            sentence_spans = [(s.start, s.end) for s in sentences_layer]
            for sent in sentences_layer:
                try:
                    sent_syntax = sent[self.syntax_layer]
                except Exception:
                    if top_level_syntax is None:
                        raise KeyError(self.syntax_layer)
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
        """Add one matched relation row for the whole pattern match."""
        if match.pattern_name not in self._pattern_by_name:
            raise KeyError(f"Unknown pattern name: {match.pattern_name!r}")

        if layer.text_object is None:
            raise RuntimeError("Relation layer is missing the source text object.")

        annotation_payload = build_match_annotation_payload(
            match=match,
            patterns_by_name=self._pattern_by_name,
            span_names=self.output_span_names,
            include_pattern_constraints=self.include_pattern_constraints,
        )
        decorated_payload = self.annotation_decorator.decorate(
            text=cast(Text, layer.text_object),
            base_span=dict(match.role_to_node),
            annotation=annotation_payload,
        )
        if decorated_payload is None:
            return

        layer.add_annotation(decorated_payload)
