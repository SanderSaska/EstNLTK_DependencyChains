# SyntaxGraphIndex class testing
import pytest
from types import SimpleNamespace

from scripts.DepChainTagger.graph import SyntaxGraphIndex
from scripts.DepChainTagger.types import DirectionMode


def make_ann(
    token_id: int,
    head: int,
    text: str,
    upostag: str = "NOUN",
    xpostag: str = "S",
    deprel: str = "nmod",
    lemma: str = "_",
    feats: dict | None = None,
) -> SimpleNamespace:
    """Create a lightweight stanza-like annotation object for unit tests."""
    return SimpleNamespace(
        id=token_id,
        head=head,
        text=text,
        upostag=upostag,
        xpostag=xpostag,
        deprel=deprel,
        lemma=lemma,
        feats={} if feats is None else feats,
    )


def test_syntaxgraphindex_basics() -> None:
    """Basic behaviour: construction, node accessors and token order."""
    layer_ok = [
        make_ann(1, 0, "root", deprel="root"),
        make_ann(2, 1, "left_child", deprel="nmod"),
        make_ann(3, 1, "right_child", deprel="obl"),
    ]
    graph = SyntaxGraphIndex(layer_ok, sentence_id=0, sentence_span=(0, 15))

    assert graph.sent_id == 0
    assert graph.sentence_span == (0, 15)
    assert graph.token_order == [1, 2, 3]
    assert graph.has_node(2)
    assert not graph.has_node(99)
    assert graph.get_node(1).text == "root"
    assert graph.get_parent(1) is None
    assert graph.get_parent(2).id == 1

    assert [node.id for node in graph.get_children(1)] == [2, 3]
    assert [node.id for node in graph.get_root_nodes()] == [1]
    assert graph._validate_tree()


def test_syntaxgraphindex_iter_edges_and_nodes() -> None:
    """Iteration helpers: iter_nodes and iter_edges return expected sequences."""
    layer_ok = [
        make_ann(1, 0, "root", deprel="root"),
        make_ann(2, 1, "left_child", deprel="nmod"),
        make_ann(3, 1, "right_child", deprel="obl"),
    ]
    graph = SyntaxGraphIndex(layer_ok)

    assert [node.id for node in graph.iter_nodes()] == [1, 2, 3]

    edges_up = [
        (node.id, parent.id, direction)
        for node, parent, direction in graph.iter_edges(DirectionMode.UP)
    ]
    assert edges_up == [(2, 1, DirectionMode.UP), (3, 1, DirectionMode.UP)]

    edges_down = [
        (node.id, child.id, direction)
        for node, child, direction in graph.iter_edges(DirectionMode.DOWN)
    ]
    assert edges_down == [(1, 2, DirectionMode.DOWN), (1, 3, DirectionMode.DOWN)]


def test_syntaxgraphindex_duplicate_ids_raises() -> None:
    layer_duplicate_ids = [
        make_ann(1, 0, "root"),
        make_ann(1, 1, "duplicate"),
    ]
    with pytest.raises(ValueError):
        SyntaxGraphIndex(layer_duplicate_ids)


def test_syntaxgraphindex_missing_head_raises() -> None:
    layer_missing_head = [
        make_ann(1, 0, "root"),
        make_ann(2, 99, "orphan"),
    ]
    with pytest.raises(ValueError):
        SyntaxGraphIndex(layer_missing_head)


def test_syntaxgraphindex_cycle_raises() -> None:
    layer_cycle = [
        make_ann(1, 2, "a"),
        make_ann(2, 1, "b"),
    ]
    with pytest.raises(ValueError):
        SyntaxGraphIndex(layer_cycle)
