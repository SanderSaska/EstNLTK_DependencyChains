# DepChildTagger class testing
from typing import Any, cast

import pytest

from scripts.DepChainTagger.child_matcher import DepChildMatcher
from scripts.DepChainTagger.dep_child_tagger import DepChildTagger
from scripts.DepChainTagger.conditions import (
    EdgeConstraint,
    NodeConstraint,
    ValueCondition,
)
from scripts.DepChainTagger.patterns import PathPattern
from scripts.DepChainTagger.types import ConditionMode, DirectionMode


def build_child_pattern(name: str) -> PathPattern:
    parent_node = NodeConstraint(
        role="parent",
        attribute_conditions={"upostag": ValueCondition(ConditionMode.WILDCARD)},
    )
    child_node = NodeConstraint(
        role="child",
        attribute_conditions={"upostag": ValueCondition(ConditionMode.WILDCARD)},
    )
    edge_constraint = EdgeConstraint(
        direction=DirectionMode.DOWN,
        attribute_conditions={"deprel": ValueCondition(ConditionMode.WILDCARD)},
        min_hops=1,
        max_hops=1,
    )
    return PathPattern(
        name=name,
        node_steps=(parent_node, child_node),
        edge_steps=(edge_constraint,),
        anchor_role="parent",
        emit_roles=("parent", "child"),
    )


def test_constructor_wires_child_matcher():
    pattern = build_child_pattern("base_p")
    tagger = DepChildTagger(patterns=(pattern,))

    # Check: DepChildTagger builds and wires an internal DepChildMatcher
    assert tagger._depchild_tagger is not None
    # Check: the internal wrapper exposes a matcher instance
    assert tagger._depchild_tagger.matcher is not None
    # Check: the matcher is an instance of DepChildMatcher
    assert isinstance(tagger._depchild_tagger.matcher, DepChildMatcher)


def test_constructor_validation():
    pattern = build_child_pattern("base_p")

    # Check: constructor enforces tuple type for patterns
    with pytest.raises(TypeError):
        DepChildTagger(patterns=cast(Any, [pattern]))

    # Check: duplicate patterns are rejected
    with pytest.raises(ValueError):
        DepChildTagger(patterns=(pattern, pattern))


def test_constructor_accepts_custom_input_layer_names():
    pattern = build_child_pattern("base_p")

    tagger = DepChildTagger(
        patterns=(pattern,),
        syntax_layer="v172_stanza_syntax",
        sentences_layer="sentences",
    )

    assert tagger.syntax_layer == "v172_stanza_syntax"
    assert tagger.sentences_layer == "sentences"
    assert tagger.input_layers == ("v172_stanza_syntax", "sentences")
