import networkx as nx
import pytest
from cfpq_helpers import GRAMMARS, build_chain, random_graph, reference_cfpq
from pyformlang.cfg import CFG, Variable

from project.task7 import matrix_based_cfpq


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("text", GRAMMARS)
def test_matrix_agrees_with_reference_on_cyclic_graphs(seed, text):
    cfg = CFG.from_text(text)
    graph = random_graph(seed)

    assert matrix_based_cfpq(cfg, graph) == reference_cfpq(cfg, graph)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("text", GRAMMARS)
def test_matrix_agrees_with_reference_on_start_and_final_subsets(seed, text):
    cfg = CFG.from_text(text)
    graph = random_graph(seed)
    starts, finals = {0, 2, 3, 6}, {1, 2, 4, 6}

    assert matrix_based_cfpq(cfg, graph, starts, finals) == {
        (source, target)
        for source, target in reference_cfpq(cfg, graph)
        if source in starts and target in finals
    }


def test_matrix_counts_brackets():
    graph = build_chain("aabbb")

    assert matrix_based_cfpq(CFG.from_text("S -> a S b | a b"), graph) == {
        (1, 3),
        (0, 4),
    }


def test_matrix_needs_several_iterations_on_long_derivation():
    graph = build_chain("aaaabbbb")

    assert matrix_based_cfpq(CFG.from_text("S -> a S b | a b"), graph) == {
        (3, 5),
        (2, 6),
        (1, 7),
        (0, 8),
    }


def test_matrix_empty_word_on_isolated_nodes():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])

    assert matrix_based_cfpq(CFG.from_text("S -> a S | $"), graph) == {
        (0, 0),
        (1, 1),
        (2, 2),
    }


def test_matrix_treats_missing_and_empty_sets_as_all_nodes():
    cfg = CFG.from_text("S -> a | b")
    graph = build_chain("ab")

    assert matrix_based_cfpq(cfg, graph) == {(0, 1), (1, 2)}
    assert matrix_based_cfpq(cfg, graph, set(), set()) == {(0, 1), (1, 2)}
    assert matrix_based_cfpq(cfg, graph, None, {2}) == {(1, 2)}


def test_matrix_accepts_digraph():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    assert matrix_based_cfpq(CFG.from_text("S -> a b"), graph) == {(0, 2)}


def test_matrix_uses_grammar_start_symbol():
    cfg = CFG.from_text("P -> a P | b\nS -> a", start_symbol=Variable("P"))

    assert matrix_based_cfpq(cfg, build_chain("aab")) == {(0, 3), (1, 3), (2, 3)}


def test_matrix_ignores_unlabeled_edges_and_variable_like_labels():
    graph = build_chain("Sb")
    graph.add_edge(0, 2)

    assert matrix_based_cfpq(CFG.from_text("S -> S | b"), graph) == {(1, 2)}


def test_matrix_grammar_without_matching_labels():
    assert matrix_based_cfpq(CFG.from_text("S -> c d"), build_chain("ab")) == set()


def test_matrix_returns_node_names_not_matrix_indices():
    graph = nx.MultiDiGraph()
    graph.add_edge("x", "y", label="a")
    graph.add_edge(30, 10, label="b")
    graph.add_edge("y", 30, label="b")

    assert matrix_based_cfpq(CFG.from_text("S -> a b | b"), graph) == {
        ("x", 30),
        ("y", 30),
        (30, 10),
    }


def test_matrix_rule_whose_head_is_its_own_operand():
    cfg = CFG.from_text("S -> a S | a")
    graph = build_chain("a" * 30)

    assert matrix_based_cfpq(cfg, graph) == {
        (source, target) for source in range(31) for target in range(source + 1, 31)
    }
