import itertools
import random
from collections import defaultdict

import networkx as nx
from pyformlang.cfg import CFG, Variable

GRAMMARS = [
    "S -> a S b | S S | $",
    "S -> a S b | a b",
    "S -> $",
    "S -> S a | $",
    "S -> N B\nB -> $\nN -> a",
    "S -> a S b S | $",
    "S -> A B\nA -> a A | $\nB -> b B | b",
    "S -> S | a b",
    "S -> S S | S | a",
    "S -> A b | b\nA -> $\nC -> c C | $",
]


def reference_cfpq(cfg: CFG, graph: nx.MultiDiGraph) -> set[tuple[int, int]]:
    """Least fixpoint of the grammar rules read as equations on node relations;
    independent of any normal form."""

    edges = defaultdict(set)
    for source, target, label in graph.edges(data="label"):
        edges[label].add((source, target))
    relation = defaultdict(set)

    def compose(left, right):
        ends = defaultdict(set)
        for source, target in right:
            ends[source].add(target)
        return {(source, end) for source, middle in left for end in ends[middle]}

    changed = True
    while changed:
        changed = False
        for production in cfg.productions:
            pairs = {(node, node) for node in graph.nodes}
            for symbol in production.body:
                step = (
                    relation[symbol]
                    if isinstance(symbol, Variable)
                    else edges[symbol.value]
                )
                pairs = compose(pairs, step)
            if not pairs <= relation[production.head]:
                relation[production.head] |= pairs
                changed = True
    return relation[cfg.start_symbol]


def build_chain(word: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(0)
    for index, label in enumerate(word):
        graph.add_edge(index, index + 1, label=label)
    return graph


def random_graph(seed: int) -> nx.MultiDiGraph:
    rng = random.Random(seed)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(7))
    for source, target in itertools.product(range(7), repeat=2):
        if rng.random() < 0.2:
            graph.add_edge(source, target, label=rng.choice("ab"))
    return graph
