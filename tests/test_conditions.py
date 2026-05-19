# ValueCondition class testing
from types import SimpleNamespace

from scripts.DepChainTagger.conditions import (
    EdgeConstraint,
    FeatureCondition,
    NodeConstraint,
    ValueCondition,
)
from scripts.DepChainTagger.types import (
    ConditionMode,
    DirectionMode,
    EdgeContext,
    NodePredicate,
)
import pytest


def test_valuecondition_exact_and_negation() -> None:
    exact = ValueCondition(mode=ConditionMode.EXACT, value="NOUN")
    assert exact.matches("NOUN")
    assert not exact.matches("VERB")

    neg = ValueCondition(mode=ConditionMode.NEGATION, value="VERB")
    assert neg.matches("NOUN")
    assert not neg.matches("VERB")


def test_valuecondition_membership_and_not_membership() -> None:
    membership = ValueCondition(mode=ConditionMode.MEMBERSHIP, value=("NOUN", "ADJ"))
    assert membership.matches("NOUN")
    assert not membership.matches("VERB")

    not_membership = ValueCondition(
        mode=ConditionMode.NOT_MEMBERSHIP, value=("NOUN", "ADJ")
    )
    assert not_membership.matches("VERB")
    assert not not_membership.matches("NOUN")


def test_valuecondition_wildcard_and_missing() -> None:
    wildcard = ValueCondition(mode=ConditionMode.WILDCARD, value=None)
    assert wildcard.matches("anything")
    assert wildcard.matches(None)

    exact_missing = ValueCondition(
        mode=ConditionMode.EXACT, value="NOUN", allow_missing=True
    )
    assert exact_missing.matches(None)


def test_valuecondition_normalizer_and_constructor_validation() -> None:
    exact_norm = ValueCondition(
        mode=ConditionMode.EXACT,
        value="Noun",
        normalizer=lambda x: x.lower() if isinstance(x, str) else x,
    )
    assert exact_norm.matches("NOUN")

    with pytest.raises(ValueError):
        ValueCondition(mode=ConditionMode.EXACT, value=None)
    with pytest.raises(ValueError):
        ValueCondition(mode=ConditionMode.WILDCARD, value="not-none")
    with pytest.raises(ValueError):
        ValueCondition(mode=ConditionMode.NOT_MEMBERSHIP, value=None)
    with pytest.raises(TypeError):
        ValueCondition(
            mode=ConditionMode.EXACT, value="NOUN", normalizer="not-callable"
        )


# FeatureCondition class testing
def test_featurecondition_exact_required_and_forbidden() -> None:
    exact = FeatureCondition(
        mode=ConditionMode.EXACT,
        required={"Case": "Gen", "Number": "Sing"},
        forbidden={"Polarity": "Neg"},
        allow_extra_keys=False,
    )
    assert exact.matches({"Case": "Gen", "Number": "Sing", "Polarity": "Pos"})
    assert not exact.matches({"Case": "Nom", "Number": "Sing", "Polarity": "Pos"})
    assert not exact.matches({"Case": "Gen", "Number": "Sing", "Polarity": "Neg"})
    assert not exact.matches({"Case": "Gen", "Polarity": "Pos"})


def test_featurecondition_allow_missing_and_extra_keys() -> None:
    exact_allow_missing = FeatureCondition(
        mode=ConditionMode.EXACT,
        required={"Case": "Gen", "Number": "Sing"},
        forbidden={"Polarity": "Neg"},
        allow_missing=True,
        allow_extra_keys=True,
    )
    assert exact_allow_missing.matches({"Case": "Gen"})

    exact_no_extra = FeatureCondition(
        mode=ConditionMode.EXACT,
        required={"Case": "Gen"},
        forbidden={"Polarity": "Neg"},
        allow_extra_keys=False,
    )
    assert not exact_no_extra.matches({"Case": "Gen", "Other": "X"})

    exact_with_extra = FeatureCondition(
        mode=ConditionMode.EXACT,
        required={"Case": "Gen"},
        forbidden={"Polarity": "Neg"},
        allow_extra_keys=True,
    )
    assert exact_with_extra.matches({"Case": "Gen", "Other": "X"})


def test_featurecondition_negation_and_wildcard() -> None:
    neg = FeatureCondition(
        mode=ConditionMode.NEGATION,
        required={"Case": "Gen", "Number": "Sing"},
        forbidden={"Polarity": "Neg"},
    )
    assert not neg.matches({"Case": "Gen", "Number": "Sing"})
    assert neg.matches({"Case": "Gen", "Number": "Plur"})
    assert not neg.matches({"Case": "Gen", "Number": "Plur", "Polarity": "Neg"})

    wildcard = FeatureCondition(mode=ConditionMode.WILDCARD)
    assert wildcard.matches({"anything": "goes"})
    assert wildcard.matches(None)


def test_featurecondition_normalizer_and_constructor_validation() -> None:
    norm_cond = FeatureCondition(
        mode=ConditionMode.EXACT,
        required={"Case": "gEn"},
        forbidden={"Polarity": "nEg"},
        normalizer=lambda x: x.lower() if isinstance(x, str) else x,
        allow_extra_keys=True,
    )
    assert norm_cond.matches({"Case": "GEN", "Polarity": "pos"})

    with pytest.raises(ValueError):
        FeatureCondition(mode=ConditionMode.EXACT)
    with pytest.raises(ValueError):
        FeatureCondition(mode=ConditionMode.WILDCARD, required={"Case": "Gen"})
    with pytest.raises(TypeError):
        FeatureCondition(
            mode=ConditionMode.EXACT,
            required={"Case": "Gen"},
            normalizer="not-callable",
        )
    with pytest.raises(TypeError):
        FeatureCondition(mode=ConditionMode.EXACT, required=[("Case", "Gen")])


# NodeConstraint class testing
def make_node(
    text: str = "test",
    upostag: str = "NOUN",
    xpostag: str = "S",
    lemma: str = "kass",
    deprel: str = "nmod",
    feats: dict | None = None,
) -> SimpleNamespace:
    """Create a lightweight span-like node object for NodeConstraint tests."""
    return SimpleNamespace(
        text=text,
        upostag=upostag,
        xpostag=xpostag,
        lemma=lemma,
        deprel=deprel,
        feats={} if feats is None else feats,
    )


def test_nodeconstraint_happy_path_and_scalar_mismatch() -> None:
    node = make_node(
        upostag="NOUN",
        xpostag="S",
        lemma="lendur",
        deprel="nmod",
        feats={"sg": "sg", "n": "n"},
    )

    constraint = NodeConstraint(
        role="target",
        attribute_conditions={
            "upostag": ValueCondition(ConditionMode.EXACT, "NOUN"),
            "xpostag": ValueCondition(ConditionMode.EXACT, "S"),
            "lemma": ValueCondition(ConditionMode.EXACT, "lendur"),
            "deprel": ValueCondition(ConditionMode.EXACT, "nmod"),
        },
        feats_condition=FeatureCondition(
            mode=ConditionMode.EXACT,
            required={"sg": "sg", "n": "n"},
            allow_extra_keys=True,
        ),
    )
    assert constraint.matches(node)

    wrong_node = make_node(upostag="VERB", xpostag="V", lemma="andma", deprel="root")
    assert not constraint.matches(wrong_node)


def test_nodeconstraint_feature_mismatch_and_predicates() -> None:
    constraint = NodeConstraint(
        role="target",
        attribute_conditions={
            "upostag": ValueCondition(ConditionMode.EXACT, "NOUN"),
            "xpostag": ValueCondition(ConditionMode.EXACT, "S"),
            "lemma": ValueCondition(ConditionMode.EXACT, "lendur"),
            "deprel": ValueCondition(ConditionMode.EXACT, "nmod"),
        },
        feats_condition=FeatureCondition(
            mode=ConditionMode.EXACT,
            required={"sg": "sg", "n": "n"},
            allow_extra_keys=True,
        ),
    )
    feature_mismatch_node = make_node(
        upostag="NOUN",
        xpostag="S",
        lemma="lendur",
        deprel="nmod",
        feats={"sg": "sg", "g": "g"},
    )
    assert not constraint.matches(feature_mismatch_node)

    pred_ok: NodePredicate = lambda n: n.text.startswith("l")
    pred_fail: NodePredicate = lambda n: n.text.endswith("z")
    pred_constraint_ok = NodeConstraint(role="pred_test", extra_predicates=(pred_ok,))
    pred_constraint_fail = NodeConstraint(
        role="pred_test", extra_predicates=(pred_fail,)
    )
    predicate_node = make_node(text="lendur")
    assert pred_constraint_ok.matches(predicate_node)
    assert not pred_constraint_fail.matches(predicate_node)


def test_nodeconstraint_selectivity_describe_and_validation() -> None:
    unconstrained = NodeConstraint(role="a")
    exact_upos = NodeConstraint(
        role="b",
        attribute_conditions={"upostag": ValueCondition(ConditionMode.EXACT, "NOUN")},
    )
    exact_upos_plus_feats = NodeConstraint(
        role="c",
        attribute_conditions={"upostag": ValueCondition(ConditionMode.EXACT, "NOUN")},
        feats_condition=FeatureCondition(
            mode=ConditionMode.EXACT, required={"sg": "sg"}, allow_extra_keys=True
        ),
    )
    assert unconstrained.score_selectivity() < exact_upos.score_selectivity()
    assert exact_upos.score_selectivity() < exact_upos_plus_feats.score_selectivity()

    constraint = NodeConstraint(
        role="target",
        attribute_conditions={"upostag": ValueCondition(ConditionMode.EXACT, "NOUN")},
        feats_condition=FeatureCondition(
            mode=ConditionMode.EXACT, required={"sg": "sg"}, allow_extra_keys=True
        ),
    )
    desc = constraint.describe()
    assert "Role: target" in desc
    assert "Attribute 'upostag':" in desc and "Feats:" in desc

    with pytest.raises(TypeError):
        NodeConstraint(role="")
    with pytest.raises(TypeError):
        NodeConstraint(
            role="bad_upos", attribute_conditions={"upostag": "not-a-valuecondition"}
        )
    with pytest.raises(TypeError):
        NodeConstraint(role="bad_feats", feats_condition="not-a-featurecondition")
    with pytest.raises(TypeError):
        NodeConstraint(role="bad_preds", extra_predicates=[lambda n: True])
    with pytest.raises(TypeError):
        NodeConstraint(role="bad_pred_member", extra_predicates=("not-callable",))
    with pytest.raises(ValueError):
        NodeConstraint(
            role="bad_dict",
            attribute_conditions={
                "misc": ValueCondition(ConditionMode.EXACT, {"Case": "Gen"})
            },
        )


# EdgeConstraint class testing
def make_edge_context(
    direction: DirectionMode,
    deprel: str | None,
    hops: int,
    crosses_sentence: bool,
) -> EdgeContext:
    """Build an EdgeContext instance for tests."""
    ctx = EdgeContext(
        direction=direction,
        deprel=deprel,
        hops=hops,
        crosses_sentence=crosses_sentence,
    )
    return ctx


def test_edgeconstraint_up_and_direction_mismatch() -> None:
    c_up = EdgeConstraint(
        direction=DirectionMode.UP,
        attribute_conditions={"deprel": ValueCondition(ConditionMode.EXACT, "nmod")},
        min_hops=1,
        max_hops=2,
    )
    ctx_up_ok = make_edge_context(DirectionMode.UP, "nmod", 1, False)
    assert c_up.matches(ctx_up_ok)

    ctx_wrong_dir = make_edge_context(DirectionMode.DOWN, "nmod", 1, False)
    assert not c_up.matches(ctx_wrong_dir)


def test_edgeconstraint_both_direction_and_deprel_hops() -> None:
    c_both = EdgeConstraint(
        direction=DirectionMode.BOTH,
        attribute_conditions={"deprel": ValueCondition(ConditionMode.EXACT, "obl")},
        min_hops=1,
        max_hops=3,
    )
    assert c_both.matches(make_edge_context(DirectionMode.UP, "obl", 2, False))
    assert c_both.matches(make_edge_context(DirectionMode.DOWN, "obl", 2, False))
    c_up = EdgeConstraint(
        direction=DirectionMode.UP,
        attribute_conditions={"deprel": ValueCondition(ConditionMode.EXACT, "nmod")},
        min_hops=1,
        max_hops=2,
    )
    assert not c_up.matches(make_edge_context(DirectionMode.UP, "obl", 1, False))
    assert not c_up.matches(make_edge_context(DirectionMode.UP, "nmod", 0, False))
    assert not c_up.matches(make_edge_context(DirectionMode.UP, "nmod", 3, False))


def test_edgeconstraint_cross_sentence_policy_describe_and_validation() -> None:
    c_up = EdgeConstraint(
        direction=DirectionMode.UP,
        attribute_conditions={"deprel": ValueCondition(ConditionMode.EXACT, "nmod")},
        min_hops=1,
        max_hops=2,
    )

    desc = c_up.describe()
    assert "Direction: up" in desc
    assert "Attribute 'deprel':" in desc
    assert "Hops:" in desc

    with pytest.raises(TypeError):
        EdgeConstraint(direction="up")
    with pytest.raises(TypeError):
        EdgeConstraint(
            direction=DirectionMode.UP,
            attribute_conditions={"deprel": "not-a-valuecondition"},
        )
    with pytest.raises(ValueError):
        EdgeConstraint(
            direction=DirectionMode.UP,
            attribute_conditions={
                "meta": ValueCondition(ConditionMode.EXACT, {"a": 1})
            },
        )
    with pytest.raises(ValueError):
        EdgeConstraint(direction=DirectionMode.UP, min_hops=-1)
    with pytest.raises(ValueError):
        EdgeConstraint(direction=DirectionMode.UP, max_hops=-1)
    with pytest.raises(ValueError):
        EdgeConstraint(direction=DirectionMode.UP, min_hops=3, max_hops=1)


# End of condition tests (individual pytest functions replace the grouped runner)
