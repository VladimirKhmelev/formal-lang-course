from copy import deepcopy

import networkx as nx
import pytest
from cfpq_helpers import GRAMMARS, build_chain, random_graph, reference_cfpq
from pyformlang.cfg import CFG, Production, Terminal, Variable
from pyformlang.rsa import RecursiveAutomaton

from project.task8 import cfg_to_rsm, ebnf_to_rsm
from project.task9 import gll_based_cfpq


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    "text",
    GRAMMARS
    + [
        "S -> A\nA -> B a | b\nB -> A | $",
        "S -> A B\nA -> S | $\nB -> b B | $",
        "S -> A A\nA -> B | a\nB -> A | $",
    ],
)
@pytest.mark.parametrize("restricted", [False, True])
def test_gll_agrees_with_reference_on_cyclic_graphs(seed, text, restricted):
    cfg = CFG.from_text(text)
    graph = random_graph(seed)
    starts = {0, 2, 3, 6} if restricted else set(graph.nodes)
    finals = {1, 2, 4, 6} if restricted else set(graph.nodes)

    assert gll_based_cfpq(cfg_to_rsm(cfg), graph, starts, finals) == {
        (source, target)
        for source, target in reference_cfpq(cfg, graph)
        if source in starts and target in finals
    }


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    "text,ebnf",
    [
        ("S -> a S | b S | $", "S -> (a | b)*"),
        ("S -> a S b S | $", "S -> (a S b)*"),
        ("S -> A S | $\nA -> a b", "S -> A*\nA -> a b"),
    ],
)
def test_gll_on_ebnf_agrees_with_reference(seed, text, ebnf):
    graph = random_graph(seed)

    assert gll_based_cfpq(ebnf_to_rsm(ebnf), graph) == reference_cfpq(
        CFG.from_text(text), graph
    )


@pytest.mark.parametrize("body", ["$", "a"])
def test_gll_replays_completed_call_for_later_return_address(body):
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 0, label="a")
    rsm = ebnf_to_rsm(f"S -> A A\nA -> {body}")

    assert gll_based_cfpq(rsm, graph) == {(0, 0)}


def test_gll_shares_calls_without_mixing_start_vertices():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 2, label="a")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    rsm = ebnf_to_rsm("S -> a A\nA -> b")

    assert gll_based_cfpq(rsm, graph, {0, 1}, {3}) == {(0, 3), (1, 3)}


def test_gll_keeps_nested_call_results_after_matching_a_terminal():
    rsm = ebnf_to_rsm("S -> A c\nA -> a B\nB -> b")

    assert gll_based_cfpq(rsm, build_chain("abc"), {0}, {3}) == {(0, 3)}


def test_gll_endpoint_filters_do_not_restrict_intermediate_calls():
    rsm = ebnf_to_rsm("S -> A B\nA -> a\nB -> b")

    assert gll_based_cfpq(rsm, build_chain("ab"), {0}, {2}) == {(0, 2)}


@pytest.mark.parametrize("text", ["S -> S", "S -> A\nA -> S", "S -> A A\nA -> S"])
def test_gll_terminates_on_unproductive_recursive_calls(text):
    assert gll_based_cfpq(cfg_to_rsm(CFG.from_text(text)), build_chain("a")) == set()


def test_gll_nullable_recursive_calls_on_isolated_vertices():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    rsm = ebnf_to_rsm("S -> A B\nA -> A | $\nB -> B | $")

    assert gll_based_cfpq(rsm, graph) == {(0, 0), (1, 1), (2, 2)}
    assert gll_based_cfpq(rsm, graph, {0, 1}, {1, 2}) == {(1, 1)}


def test_gll_continues_after_a_final_box_state():
    assert gll_based_cfpq(ebnf_to_rsm("S -> a*"), build_chain("aa"), {0}) == {
        (0, 0),
        (0, 1),
        (0, 2),
    }


def test_gll_handles_deep_calls_without_python_recursion():
    length = 1500
    rsm = ebnf_to_rsm("S -> a S | b")

    assert gll_based_cfpq(rsm, build_chain("a" * length + "b"), {0}, {length + 1}) == {
        (0, length + 1)
    }


def test_gll_treats_missing_and_empty_endpoint_sets_as_all_nodes():
    rsm = ebnf_to_rsm("S -> a | b")
    graph = build_chain("ab")

    assert gll_based_cfpq(rsm, graph) == {(0, 1), (1, 2)}
    assert gll_based_cfpq(rsm, graph, set(), set()) == {(0, 1), (1, 2)}
    assert gll_based_cfpq(rsm, graph, None, {2}) == {(1, 2)}


def test_gll_keeps_parallel_edges_and_node_names():
    graph = nx.MultiDiGraph()
    graph.add_edge("x", 30, label="a")
    graph.add_edge("x", 30, label="b")
    graph.add_edge(30, "y", label="b")
    graph.add_edge(30, "y", label="b")
    rsm = ebnf_to_rsm("S -> a b | b")

    assert gll_based_cfpq(rsm, graph) == {("x", "y"), ("x", 30), (30, "y")}


def test_gll_accepts_digraph():
    graph = nx.DiGraph(build_chain("ab"))

    assert gll_based_cfpq(ebnf_to_rsm("S -> a b"), graph) == {(0, 2)}


def test_gll_uses_initial_label_and_distinguishes_literal_from_call():
    start = Variable("P")
    cfg = CFG(
        start_symbol=start,
        productions={
            Production(start, [Terminal("P"), start]),
            Production(start, [Terminal("b")]),
        },
    )

    assert gll_based_cfpq(cfg_to_rsm(cfg), build_chain("PPb")) == {
        (0, 3),
        (1, 3),
        (2, 3),
    }


def test_gll_ignores_unlabeled_edges_and_variable_like_labels():
    graph = build_chain("Sb")
    graph.add_edge(0, 2)

    assert gll_based_cfpq(ebnf_to_rsm("S -> S | b"), graph) == {(1, 2)}


@pytest.mark.parametrize("text", ["S -> a", "S -> $"])
def test_gll_on_empty_graph(text):
    assert gll_based_cfpq(ebnf_to_rsm(text), nx.DiGraph()) == set()


def test_gll_on_empty_rsm():
    assert gll_based_cfpq(RecursiveAutomaton(), build_chain("a")) == set()


def test_gll_rejects_rsm_without_entry_box():
    rsm = RecursiveAutomaton()
    rsm.add_box(ebnf_to_rsm("S -> a").get_box("S"))

    with pytest.raises(ValueError, match="initial label"):
        gll_based_cfpq(rsm, build_chain("a"))


def test_gll_does_not_modify_inputs_and_can_reuse_rsm():
    rsm = ebnf_to_rsm("S -> a S b | $")
    original_rsm = deepcopy(rsm)
    graph = build_chain("aabb")
    original_graph = deepcopy(graph)
    starts, finals = {0, 1}, {3, 4}

    assert gll_based_cfpq(rsm, graph, starts, finals) == {(0, 4), (1, 3)}
    assert gll_based_cfpq(rsm, build_chain("ab"), {0}, {2}) == {(0, 2)}
    assert rsm == original_rsm
    assert rsm.initial_label == original_rsm.initial_label
    assert nx.utils.graphs_equal(graph, original_graph)
    assert starts == {0, 1}
    assert finals == {3, 4}
