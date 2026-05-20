import estnltk
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Iterable,
    Self,
    Any,
)

from .types import DirectionMode
from .config import SEED
import matplotlib.pyplot as plt
import networkx as nx


class SyntaxGraphIndex:
    """
    A class to represent the dependency syntax graph of a sentence, indexed by token IDs.

    The graph is built from an estnltk Layer containing the sentence annotations, and provides methods to access nodes, parents, children, and edges in the dependency graph.

    ## Attributes:
    - **sentences_layer** (`estnltk.Layer`): The layer containing the stanza syntax annotations for the sentence from which the graph index is built.
    - **nodes_by_id** (`Dict[int, estnltk.Span]`): A mapping from token IDs to their corresponding estnltk Span annotations.
    - **parent_by_id** (`Dict[int, Optional[int]]`): A mapping from token IDs to their parent token IDs in the dependency graph.
    - **children_by_id** (`Dict[int, List[int]]`): A mapping from token IDs to a list of their child token IDs in the dependency graph.
    - **token_order** (`List[int]`): A list of token IDs in the order they appear in the sentence.
    - **sent_id** (`Optional[int]`): The ID of the sentence being indexed.
    - **sentence_span** (`Optional[Tuple[int, int]]`): The character span of the sentence in the original text.

    ## Methods:
    - :func:`~SyntaxGraphIndex.__init__`: Initializes the graph index from the given sentences layer.
    - :func:`~SyntaxGraphIndex.build_from_layer`: Builds the graph index from the provided sentences layer.
    - :func:`~SyntaxGraphIndex.get_node`: Retrieves the estnltk Span annotation for a given token ID.
    - :func:`~SyntaxGraphIndex.get_parent`: Retrieves the parent node of a given token ID in the dependency graph.
    - :func:`~SyntaxGraphIndex.get_children`: Retrieves the child nodes of a given token ID in the dependency graph.
    - :func:`~SyntaxGraphIndex.iter_nodes`: Iterates over all nodes in the graph in the order they appear in the sentence.
    - :func:`~SyntaxGraphIndex.iter_edges`: Iterates over all edges in the graph, optionally filtering by direction (up, down, or both).
    - :func:`~SyntaxGraphIndex.get_root_nodes`: Retrieves the root nodes of the dependency graph (nodes with no parent).
    - :func:`~SyntaxGraphIndex.has_node`: Checks if a given token ID exists in the graph.
    """

    stanza_syntax: estnltk.Layer
    nodes_by_id: Dict[int, estnltk.Span]
    parent_by_id: Dict[int, Optional[int]]
    children_by_id: Dict[int, List[int]]
    token_order: List[int]
    sent_id: Optional[int]
    sentence_span: Optional[Tuple[int, int]]
    lookup_cache: Dict[Tuple, Any]

    def __init__(
        self: Self,
        stanza_syntax_layer: estnltk.Layer,
        sentence_id: Optional[int] = None,
        sentence_span: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initializes the SyntaxGraphIndex from the given sentences layer.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being initialized.
            stanza_syntax_layer (estnltk.Layer): The layer containing the stanza syntax annotations for the sentence from which to build the graph index.
            sentence_id (Optional[int], optional): The ID of the sentence being indexed. Defaults to None.
            sentence_span (Optional[Tuple[int, int]], optional): The character span of the sentence in the original text. Defaults to None.
        """

        # Initialize the graph index from the given sentences layer
        self.stanza_syntax: estnltk.Layer = stanza_syntax_layer
        self.nodes_by_id: Dict[int, estnltk.Span] = {}
        self.parent_by_id: Dict[int, Optional[int]] = {}
        self.children_by_id: Dict[int, List[int]] = {}
        self.token_order: List[int] = []
        self.sent_id: Optional[int] = sentence_id
        self.sentence_span: Optional[Tuple[int, int]] = sentence_span
        self.lookup_cache = {}

        # Build the graph index from the sentences layer
        self.build_from_layer(self.stanza_syntax)

        # Validate the graph structure (optional, can be commented out if not needed)
        if not self._validate_tree():
            raise ValueError(
                "The provided stanza syntax layer does not form a valid tree structure."
            )

    def build_from_layer(self: Self, stanza_syntax: estnltk.Layer) -> None:
        """
        Builds the graph index from the provided syntax layer.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being built.
            stanza_syntax (estnltk.Layer): The layer containing the stanza syntax annotations for the sentence from which to build the graph index.
        """
        # Build the graph index from the provided sentences layer
        for ann in stanza_syntax:
            if ann.id in self.nodes_by_id:
                raise ValueError(f"Duplicate token id encountered: {ann.id}")
            self.nodes_by_id[ann.id] = ann
            self.parent_by_id[ann.id] = ann.head
            self.children_by_id[ann.id] = []
            self.token_order.append(ann.id)

        # Populate the children_by_id mapping based on the parent_by_id mapping
        for ann in stanza_syntax:
            if ann.head == 0:
                continue
            if ann.head not in self.children_by_id:
                raise ValueError(
                    f"Invalid head reference: token {ann.id} points to missing head {ann.head}."
                )
            self.children_by_id[ann.head].append(ann.id)

    def get_node(self: Self, token_id: int) -> Optional[estnltk.Span]:
        """
        Gets the estnltk Span annotation for a given token ID.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.
            token_id (int): The ID of the token for which to retrieve the annotation.

        Returns:
            Optional[estnltk.Span]: The estnltk Span annotation corresponding to the given token ID, or None if the token ID does not exist in the graph index.
        """
        return self.nodes_by_id.get(token_id)

    def get_parent(self: Self, token_id: int) -> Optional[estnltk.Span]:
        """
        Gets the parent node of a given token ID in the dependency graph.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.
            token_id (int): The ID of the token for which to retrieve the parent node.

        Returns:
            Optional[estnltk.Span]: The estnltk Span annotation corresponding to the parent node of the given token ID in the dependency graph, or None if the token ID does not exist in the graph index or if it is a root node (with no parent).
        """
        parent_id = self.parent_by_id.get(token_id)
        if parent_id is not None:
            return self.nodes_by_id.get(parent_id)
        return None

    def get_children(self: Self, token_id: int) -> List[estnltk.Span]:
        """
        Gets the child nodes of a given token ID in the dependency graph.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.
            token_id (int): The ID of the token for which to retrieve the child nodes.

        Returns:
            List[estnltk.Span]: A list of estnltk Span annotations corresponding to the child nodes of the given token ID in the dependency graph. If the token ID does not exist in the graph index or has no children, an empty list is returned.
        """
        child_ids = self.children_by_id.get(token_id, [])
        return [self.nodes_by_id[child_id] for child_id in child_ids]

    def iter_nodes(self: Self) -> Iterable[estnltk.Span]:
        """
        Iterates over all nodes in the graph in the order they appear in the sentence.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being iterated over.

        Returns:
            Iterable[estnltk.Span]: An iterator that yields estnltk Span annotations for each node in the graph, in the order they appear in the sentence.

        Yields:
            estnltk.Span: The estnltk Span annotation for each node in the graph, yielded in the order they appear in the sentence.
        """
        for token_id in self.token_order:
            yield self.nodes_by_id[token_id]

    def iter_edges(
        self: Self, direction: DirectionMode = DirectionMode.BOTH
    ) -> Iterable[Tuple[Optional[estnltk.Span], Optional[estnltk.Span], DirectionMode]]:
        """
        Iterates over all edges in the graph, optionally filtering by direction (up, down, or both).

        Args:
            self (Self): The instance of the SyntaxGraphIndex being iterated over.
            direction (DirectionMode, optional): The direction of edges to iterate over. Can be DirectionMode.UP for parent-child edges, DirectionMode.DOWN for child-parent edges, or DirectionMode.BOTH for all edges. Defaults to DirectionMode.BOTH.

        Returns:
            Iterable[Tuple[Optional[estnltk.Span], Optional[estnltk.Span], str]]: _description_

        Yields:
            Iterator[Iterable[Tuple[Optional[estnltk.Span], Optional[estnltk.Span], str]]]: _description_
        """
        for token_id in self.token_order:
            node = self.nodes_by_id[token_id]
            parent_id = self.parent_by_id.get(token_id)
            if parent_id is not None and parent_id != 0:
                parent_node = self.nodes_by_id.get(parent_id)
                if direction in [DirectionMode.BOTH, DirectionMode.UP]:
                    # Move from id to head (up the tree)
                    yield (node, parent_node, DirectionMode.UP)
                if direction in [DirectionMode.BOTH, DirectionMode.DOWN]:
                    # Move from head to id (down the tree)
                    if parent_id != 0:
                        # Skip the root node which has no parent (head = 0)
                        yield (parent_node, node, DirectionMode.DOWN)

    def get_root_nodes(self: Self) -> List[estnltk.Span]:
        """
        Gets the root nodes of the dependency graph (nodes with no parent).

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.

        Returns:
            List[estnltk.Span]: A list of estnltk Span annotations corresponding to the root nodes of the dependency graph (nodes with no parent). If there are no root nodes, an empty list is returned.
        """
        root_nodes = []
        for token_id in self.token_order:
            if self.parent_by_id.get(token_id) == 0:  # Root nodes have head = 0
                root_nodes.append(self.nodes_by_id[token_id])
        return root_nodes

    def has_node(self: Self, token_id: int) -> bool:
        """
        Checks if a given token ID exists in the graph.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.
            token_id (int): The ID of the token to check for existence in the graph.

        Returns:
            bool: True if the given token ID exists in the graph index, False otherwise.
        """
        return token_id in self.nodes_by_id

    def _format_node_label(
        self: Self, token_id: int, with_node_labels: bool = False
    ) -> str:
        """
        Build a human-readable node label for graph visualisation.

        The label includes the surface form, the universal part-of-speech tag,
        and the morphological features.
        """
        node = self.nodes_by_id[token_id]
        word = getattr(node, "text", "") or ""
        upostag = getattr(node, "upostag", None)
        feats = getattr(node, "feats", None)
        feats_text = "_" if not feats else str(feats)
        if with_node_labels:
            return f"{word}\n{upostag or '_'}\n{feats_text}"
        return f"{word}"

    def _format_edge_label(
        self: Self, child_id: int, with_edge_labels: bool = True
    ) -> str:
        """
        Build the dependency-label text for an edge.

        The dependency relation is stored on the child node in Stanza syntax.
        """
        child_node = self.nodes_by_id[child_id]
        if with_edge_labels:
            return str(getattr(child_node, "deprel", "") or "")
        return ""

    def to_networkx_graph(
        self: Self, with_node_labels: bool = False, with_edge_labels: bool = True
    ) -> nx.DiGraph:
        """
        Convert the indexed tree into a NetworkX directed graph.

        Returns:
            networkx.DiGraph: A directed graph with parent-to-child edges.

        Raises:
            ImportError: If NetworkX is not installed.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "SyntaxGraphIndex.to_networkx_graph() requires the 'networkx' package."
            ) from exc

        graph = nx.DiGraph()

        for token_id in self.token_order:
            node = self.nodes_by_id[token_id]
            graph.add_node(
                token_id,
                label=self._format_node_label(
                    token_id, with_node_labels=with_node_labels
                ),
                text=getattr(node, "text", None),
                upostag=getattr(node, "upostag", None),
                feats=getattr(node, "feats", None),
            )

        for child_id in self.token_order:
            parent_id = self.parent_by_id.get(child_id)
            if parent_id is None or parent_id == 0:
                continue
            graph.add_edge(
                parent_id,
                child_id,
                label=self._format_edge_label(
                    child_id, with_edge_labels=with_edge_labels
                ),
            )

        return graph

    def visualize(
        self: Self,
        ax: Optional[Any] = None,
        figsize: Tuple[int, int] = (12, 8),
        layout: str = "dot",
        with_node_labels: bool = False,
        with_edge_labels: bool = True,
        node_size: int = 2600,
        font_size: int = 8,
        title: Optional[str] = None,
        show: bool = True,
    ):
        """
        Visualise the dependency tree as a readable graph.

        Args:
            ax (Optional[Axes], optional): Existing matplotlib axis to draw on.
            layout (str, optional): Layout strategy. Supported values are
                "spring" and "dot" (uses Graphviz if available, otherwise
                falls back to a spring layout).
            with_labels (bool, optional): Whether to draw node labels.
            node_size (int, optional): Matplotlib node size.
            font_size (int, optional): Font size used for node and edge labels.
            title (Optional[str], optional): Optional plot title.
            show (bool, optional): Whether to call ``plt.show()``.

        Returns:
            matplotlib.figure.Figure: The created figure.

        Raises:
            ImportError: If NetworkX or Matplotlib is not installed.
            ValueError: If an unsupported layout is requested.
        """

        graph = self.to_networkx_graph(
            with_node_labels=with_node_labels, with_edge_labels=with_edge_labels
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        if layout == "bipartite":
            positions = nx.bipartite_layout(graph, nodes=graph.nodes())
        elif layout == "kamada_kawai":
            positions = nx.kamada_kawai_layout(graph)
        elif layout == "planar":
            positions = nx.planar_layout(graph)
        elif layout == "random":
            positions = nx.random_layout(graph, seed=SEED)
        elif layout == "spectral":
            positions = nx.spectral_layout(graph)
        elif layout == "spring":
            positions = nx.spring_layout(graph, seed=SEED)
        elif layout == "shell":
            positions = nx.shell_layout(graph)
        else:
            try:
                positions = nx.nx_agraph.graphviz_layout(graph, prog=layout)
            except Exception:
                print(
                    f"Graphviz layout '{layout}' is not available. Falling back to spring layout."
                )
                positions = nx.spring_layout(graph, seed=SEED)

        node_labels = nx.get_node_attributes(graph, "label")
        edge_labels = nx.get_edge_attributes(graph, "label")

        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            arrows=True,
            arrowstyle="->",
            arrowsize=18,
            width=1.3,
            edge_color="#666666",
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_size=node_size,
            node_color="#dfeaf5",
            edgecolors="#4a6fa5",
            linewidths=1.2,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=node_labels,
            ax=ax,
            font_size=font_size,
            font_family="sans-serif",
        )
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            ax=ax,
            font_size=font_size,
            label_pos=0.5,
        )

        ax.set_axis_off()
        ax.set_title(title or self._build_visualization_title())
        fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _build_visualization_title(self: Self) -> str:
        """
        Create a default title for the visualisation.
        """
        sentence_id = (
            f"Sentence {self.sent_id}" if self.sent_id is not None else "Sentence"
        )
        if self.sentence_span is None:
            return f"{sentence_id} dependency tree"
        return f"{sentence_id} dependency tree {self.sentence_span}"

    def _validate_tree(self: Self) -> bool:
        """
        Validates that the graph forms a proper tree structure. Checks for cycles, missing heads, and orphan references.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being validated.

        Returns:
            bool: True if the graph forms a valid tree structure, False otherwise. A valid tree structure means that there are no cycles in the parent-child relationships, all nodes have a valid head (except for root nodes), and there are no orphan references (nodes that reference a non-existent head).
        """
        # Check for cycles using a depth-first search
        visited = set()

        def _dfs(node_id: int, parent_id: Optional[int]) -> bool:
            """
            Performs a depth-first search to detect cycles in the graph.

            Args:
                node_id (int): The ID of the current node being visited in the depth-first search.
                parent_id (Optional[int]): The ID of the parent node of the current node in the depth-first search. This is used to avoid false positive cycle detection when traversing back to the parent node.

            Returns:
                bool: True if no cycles are detected in the graph, False if a cycle is detected. A cycle is detected if a node is visited more than once during the depth-first search, indicating that there is a circular reference in the parent-child relationships of the graph.
            """
            if node_id in visited:
                return False  # Cycle detected
            visited.add(node_id)
            for child_id in self.children_by_id.get(node_id, []):
                if child_id == parent_id:
                    continue  # Skip the parent node to avoid false positive cycle detection
                if not _dfs(child_id, node_id):
                    return False
            return True

        # Check for valid heads and orphan references
        for token_id in self.token_order:
            parent_id = self.parent_by_id.get(token_id)
            if (
                parent_id is not None  # A head is specified
                and parent_id != 0  # Root nodes have head = 0, so we allow that
                and parent_id
                not in self.nodes_by_id  # The specified head does not exist in the graph
            ):
                return False  # Orphan reference detected (node references a non-existent head)

        # Check for cycles starting from root nodes
        root_nodes = self.get_root_nodes()
        if not root_nodes:
            return False

        for root_node in root_nodes:
            root_id = int(getattr(root_node, "id"))
            if not _dfs(root_id, None):
                return False  # Cycle detected

        # Every node must be reachable from some root.
        if len(visited) != len(self.token_order):
            return False

        return True
