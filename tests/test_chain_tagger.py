# DepChainTagger class testing
from typing import Any, cast

import pytest

from scripts.DepChainTagger.matcher import DepChainMatcher
from scripts.DepChainTagger.dep_chain_tagger import DepChainTagger
from scripts.DepChainTagger.conditions import (
    EdgeConstraint,
    NodeConstraint,
    ValueCondition,
)
from scripts.DepChainTagger.patterns import PathPattern
from scripts.DepChainTagger.types import ConditionMode, DirectionMode


def build_chain_pattern(name: str) -> PathPattern:
    parent_node = NodeConstraint(
        role="parent",
        attribute_conditions={"upostag": ValueCondition(ConditionMode.WILDCARD)},
    )
    child_node = NodeConstraint(
        role="child",
        attribute_conditions={"upostag": ValueCondition(ConditionMode.WILDCARD)},
    )
    edge_constraint = EdgeConstraint(
        direction=DirectionMode.UP,
        attribute_conditions={"deprel": ValueCondition(ConditionMode.WILDCARD)},
        min_hops=1,
        max_hops=1,
    )
    return PathPattern(
        name=name,
        node_steps=(parent_node, child_node),
        edge_steps=(edge_constraint,),
        anchor_role="child",
        emit_roles=("parent", "child"),
    )


def test_constructor_wires_chain_matcher():
    pattern = build_chain_pattern("base_p")
    tagger = DepChainTagger(patterns=(pattern,))

    # Check: DepChainTagger builds and wires an internal DepTaggerOrchestrator
    assert tagger._depchain_tagger is not None
    # Check: the internal wrapper exposes a matcher instance
    assert tagger._depchain_tagger.matcher is not None
    # Check: the matcher is an instance of DepChainMatcher
    assert isinstance(tagger._depchain_tagger.matcher, DepChainMatcher)


def test_constructor_validation():
    pattern = build_chain_pattern("base_p")

    # Check: constructor enforces tuple type for patterns
    with pytest.raises(TypeError):
        DepChainTagger(patterns=cast(Any, [pattern]))

    # Check: duplicate patterns are rejected
    with pytest.raises(ValueError):
        DepChainTagger(patterns=(pattern, pattern))
