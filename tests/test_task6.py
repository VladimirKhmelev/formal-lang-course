import itertools
import random
from collections import defaultdict

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable

from project.task4 import ms_bfs_based_rpq
from project.task6 import cfg_to_weak_normal_form, hellings_based_cfpq

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


@pytest.mark.parametrize("text", GRAMMARS)
def test_weak_normal_form_has_only_allowed_rules(text):
    grammar = cfg_to_weak_normal_form(CFG.from_text(text))

    for production in grammar.productions:
        body = production.body
        assert (
            len(body) == 0
            or (len(body) == 1 and body[0] in grammar.terminals)
            or (len(body) == 2 and all(s in grammar.variables for s in body))
        ), production


@pytest.mark.parametrize("text", GRAMMARS)
def test_weak_normal_form_preserves_language(text):
    cfg = CFG.from_text(text)
    grammar = cfg_to_weak_normal_form(cfg)

    for length in range(6):
        for word in itertools.product("ab", repeat=length):
            chain = build_chain("".join(word))
            expected = (0, length) in reference_cfpq(cfg, chain)
            assert ((0, length) in reference_cfpq(grammar, chain)) == expected, word


def random_graph(seed: int) -> nx.MultiDiGraph:
    rng = random.Random(seed)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(7))
    for source, target in itertools.product(range(7), repeat=2):
        if rng.random() < 0.2:
            graph.add_edge(source, target, label=rng.choice("ab"))
    return graph


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("text", GRAMMARS)
def test_hellings_agrees_with_reference_on_cyclic_graphs(seed, text):
    cfg = CFG.from_text(text)
    graph = random_graph(seed)

    assert hellings_based_cfpq(cfg, graph) == reference_cfpq(cfg, graph)


def test_hellings_ignores_unlabeled_edges_and_variable_like_labels():
    graph = build_chain("Sb")
    graph.add_edge(0, 2)

    assert hellings_based_cfpq(CFG.from_text("S -> S | b"), graph) == {(1, 2)}


def test_hellings_counts_brackets_unlike_regular_query():
    graph = build_chain("aabbb")

    assert hellings_based_cfpq(CFG.from_text("S -> a S b | a b"), graph) == {
        (1, 3),
        (0, 4),
    }


def test_hellings_on_cycle_with_balanced_brackets():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="b")

    assert hellings_based_cfpq(
        CFG.from_text("S -> a S b S | $"), graph, {0, 1}, {0, 1}
    ) == {(0, 0), (1, 1)}


def test_hellings_treats_missing_start_and_final_as_all_nodes():
    graph = build_chain("ab")

    assert hellings_based_cfpq(CFG.from_text("S -> a | b"), graph) == {
        (0, 1),
        (1, 2),
    }
    assert hellings_based_cfpq(CFG.from_text("S -> a | b"), graph, None, {2}) == {
        (1, 2)
    }
    assert hellings_based_cfpq(
        CFG.from_text("S -> a | b"), graph, set(), set()
    ) == hellings_based_cfpq(CFG.from_text("S -> a | b"), graph)


def test_hellings_accepts_digraph():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    assert hellings_based_cfpq(CFG.from_text("S -> a b"), graph) == {(0, 2)}


def test_hellings_uses_grammar_start_symbol():
    cfg = CFG.from_text("P -> a P | b\nS -> a", start_symbol=Variable("P"))

    assert hellings_based_cfpq(cfg, build_chain("aab")) == {(0, 3), (1, 3), (2, 3)}


@pytest.mark.parametrize(
    "text,regex",
    [("S -> a S | $", "a*"), ("S -> a b S | $", "(a b)*"), ("S -> a | b S", "b* a")],
)
def test_hellings_agrees_with_ms_bfs_on_regular_grammars(text, regex):
    graph = random_graph(0)
    graph.add_edge(7, 0, label="a")
    graph.add_edge(3, 3, label="b")
    starts, finals = {0, 2, 3, 5}, {1, 3, 4, 7}

    assert hellings_based_cfpq(
        CFG.from_text(text), graph, starts, finals
    ) == ms_bfs_based_rpq(regex, graph, starts, finals)


@pytest.mark.parametrize("reverse", [False, True])
def test_hellings_combines_facts_on_both_sides(reverse):
    edges = list(enumerate("acd"))
    graph = nx.MultiDiGraph()
    for index, label in reversed(edges) if reverse else edges:
        graph.add_edge(index, index + 1, label=label)
    cfg = CFG.from_text("S -> A B\nA -> a\nB -> C D\nC -> c\nD -> d")

    assert hellings_based_cfpq(cfg, graph) == {(0, 3)}
