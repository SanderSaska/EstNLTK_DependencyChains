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
from .output_utils import (
    build_match_annotation_payload,
    collect_output_attribute_names,
    collect_role_span_names,
)


class DepChainTagger(RelationTagger):
    """EstNLTK RelationTagger wrapper for dependency-chain matches.

    This tagger takes a set of `PathPattern` objects, finds all matches in the input text, and produces a RelationLayer where each relation corresponds to one match. The relation's attributes include token-level information for each role in the match, as well as metadata about the pattern and sentence. The tagger supports configurable deduplication strategies to control the number of matches emitted per sentence and across the entire text.
    """

    conf_param = [
        "patterns",
        "output_layer",
        "output_span_names",
        "output_attributes",
        "include_pattern_constraints",
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
        include_pattern_constraints: bool = False,
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
        self.include_pattern_constraints = include_pattern_constraints
        self.output_span_names = collect_role_span_names(patterns)
        self.output_attributes = (
            output_attributes
            if output_attributes is not None
            else collect_output_attribute_names(
                patterns, self.include_pattern_constraints
            )
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

        annotation_payload = build_match_annotation_payload(
            match=match,
            patterns_by_name=self._pattern_by_name,
            span_names=self.output_span_names,
            include_pattern_constraints=self.include_pattern_constraints,
        )
        layer.add_annotation(annotation_payload)
