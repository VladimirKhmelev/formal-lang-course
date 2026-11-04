from copy import deepcopy

import networkx as nx
import pytest
from cfpq_helpers import GRAMMARS, build_chain, random_graph, reference_cfpq
from pyformlang.cfg import CFG, Production, Terminal, Variable
from pyformlang.finite_automaton import Symbol
from pyformlang.regular_expression import MisformedRegexError
from pyformlang.rsa import RecursiveAutomaton

from project.task8 import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq


def test_cfg_to_rsm_preserves_start_symbol_and_production_bodies():
    cfg = CFG.from_text("P -> a P b | A | $\nA -> a b", start_symbol=Variable("P"))

    rsm = cfg_to_rsm(cfg)

    assert rsm.initial_label == Symbol("P")
    assert rsm.labels == {Symbol("P"), Symbol("A")}
    assert rsm.get_box("P").dfa.accepts(["a", "P", "b"])
    assert rsm.get_box("P").dfa.accepts(["A"])
    assert rsm.get_box("P").dfa.accepts([])
    assert not rsm.get_box("P").dfa.accepts(["a", "b"])
    assert rsm.get_box("A").dfa.accepts(["a", "b"])


def test_cfg_to_rsm_keeps_regex_metacharacters_literal():
    start = Variable("S")
    cfg = CFG(
        start_symbol=start,
        productions={Production(start, [Terminal("a*"), Terminal("|")])},
    )
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a*")
    graph.add_edge(1, 2, label="|")

    rsm = cfg_to_rsm(cfg)

    assert rsm.get_box("S").dfa.accepts(["a*", "|"])
    assert not rsm.get_box("S").dfa.accepts(["a", "|"])
    assert tensor_based_cfpq(rsm, graph) == {(0, 2)}


def test_cfg_to_rsm_distinguishes_terminal_from_same_named_start_symbol():
    start = Variable("S")
    cfg = CFG(start_symbol=start, productions={Production(start, [Terminal("S")])})
    graph = build_chain("S")

    rsm = cfg_to_rsm(cfg)

    assert rsm.initial_label != Symbol("S")
    assert tensor_based_cfpq(rsm, graph) == {(0, 1)}


@pytest.mark.parametrize("seed", range(6))
def test_cfg_to_rsm_distinguishes_literals_from_recursive_calls(seed):
    start, recursive = Variable("S"), Variable("a")
    cfg = CFG(
        start_symbol=start,
        productions={
            Production(start, [recursive]),
            Production(recursive, [Terminal("a"), recursive, Terminal("b")]),
            Production(recursive, []),
        },
    )
    graph = random_graph(seed)

    assert tensor_based_cfpq(cfg_to_rsm(cfg), graph) == reference_cfpq(cfg, graph)


def test_cfg_to_rsm_call_labels_do_not_collide_with_existing_symbols():
    start, other = Variable("S"), Variable("__RSM_CALL_0")
    cfg = CFG(
        start_symbol=start,
        productions={
            Production(start, [Terminal("S"), other]),
            Production(other, [Terminal("__RSM_CALL_1")]),
        },
    )
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="S")
    graph.add_edge(1, 2, label="__RSM_CALL_1")
    graph.add_edge(2, 3, label="__RSM_CALL_0")
    rsm = cfg_to_rsm(cfg)

    assert rsm.initial_label.value not in {
        "S",
        "__RSM_CALL_0",
        "__RSM_CALL_1",
    }
    assert tensor_based_cfpq(rsm, graph) == {(0, 2)}


def test_cfg_to_rsm_rejects_grammar_without_start_symbol():
    with pytest.raises(ValueError, match="start symbol"):
        cfg_to_rsm(CFG())


@pytest.mark.parametrize("text", ["S -> A", "S -> S", "S -> A\nA -> S"])
def test_cfg_to_rsm_handles_variables_without_terminal_derivations(text):
    rsm = cfg_to_rsm(CFG.from_text(text))

    assert tensor_based_cfpq(rsm, build_chain("ab")) == set()


def test_cfg_to_rsm_handles_grammar_without_productions():
    rsm = cfg_to_rsm(CFG(start_symbol=Variable("P")))

    assert rsm.initial_label == Symbol("P")
    assert tensor_based_cfpq(rsm, build_chain("ab")) == set()


def test_ebnf_to_rsm_combines_repeated_heads_and_parses_regexes():
    rsm = ebnf_to_rsm("S -> (a | b)* C\nS -> $\nC -> c c*")

    assert rsm.initial_label == Symbol("S")
    assert rsm.labels == {Symbol("S"), Symbol("C")}
    assert rsm.get_box("S").dfa.accepts(["a", "b", "C"])
    assert rsm.get_box("S").dfa.accepts([])
    assert not rsm.get_box("S").dfa.accepts(["a", "c"])
    assert rsm.get_box("C").dfa.accepts(["c", "c"])
    assert not rsm.get_box("C").dfa.accepts([])


@pytest.mark.parametrize(
    "text", ["", "garbage", "S -> a\nbroken", "S -> a -> b", "P -> a"]
)
def test_ebnf_to_rsm_rejects_empty_or_malformed_rules(text):
    with pytest.raises(ValueError):
        ebnf_to_rsm(text)


def test_ebnf_to_rsm_rejects_malformed_regex():
    with pytest.raises(MisformedRegexError):
        ebnf_to_rsm("S -> (a")


@pytest.mark.parametrize("text", ["S ->", "S -> $", "\n  S -> $\n"])
def test_ebnf_to_rsm_preserves_empty_productions(text):
    assert tensor_based_cfpq(ebnf_to_rsm(text), build_chain("a")) == {(0, 0), (1, 1)}


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("text", GRAMMARS)
@pytest.mark.parametrize("restricted", [False, True])
def test_tensor_agrees_with_reference_on_cyclic_graphs(seed, text, restricted):
    cfg = CFG.from_text(text)
    graph = random_graph(seed)
    starts = {0, 2, 3, 6} if restricted else set(graph.nodes)
    finals = {1, 2, 4, 6} if restricted else set(graph.nodes)

    assert tensor_based_cfpq(cfg_to_rsm(cfg), graph, starts, finals) == {
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
        (
            "S -> A B\nA -> a A | $\nB -> b B | $",
            "S -> A B\nA -> a*\nB -> b*",
        ),
        ("S -> A S | $\nA -> a b", "S -> A*\nA -> a b"),
        ("S -> a B | $\nB -> b S", "S -> (a B)*\nB -> b S"),
    ],
)
def test_ebnf_tensor_agrees_with_reference(seed, text, ebnf):
    graph = random_graph(seed)

    assert tensor_based_cfpq(ebnf_to_rsm(ebnf), graph) == reference_cfpq(
        CFG.from_text(text), graph
    )


@pytest.mark.parametrize("text", ["S -> a", "S -> $"])
def test_tensor_on_empty_graph(text):
    assert tensor_based_cfpq(cfg_to_rsm(CFG.from_text(text)), nx.DiGraph()) == set()


def test_tensor_on_empty_rsm():
    assert tensor_based_cfpq(RecursiveAutomaton(), build_chain("a")) == set()


def test_tensor_rejects_rsm_without_entry_box():
    rsm = RecursiveAutomaton()
    rsm.add_box(ebnf_to_rsm("S -> a").get_box("S"))

    with pytest.raises(ValueError, match="initial label"):
        tensor_based_cfpq(rsm, build_chain("a"))


def test_tensor_empty_word_on_isolated_nodes():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    rsm = cfg_to_rsm(CFG.from_text("S -> A B\nA -> $\nB -> $"))

    assert tensor_based_cfpq(rsm, graph) == {(0, 0), (1, 1), (2, 2)}
    assert tensor_based_cfpq(rsm, graph, {0, 1}, {1, 2}) == {(1, 1)}


def test_tensor_counts_brackets_across_several_iterations():
    graph = build_chain("aaaabbbb")
    rsm = cfg_to_rsm(CFG.from_text("S -> a S b | a b"))

    assert tensor_based_cfpq(rsm, graph) == {
        (3, 5),
        (2, 6),
        (1, 7),
        (0, 8),
    }


def test_tensor_keeps_states_of_different_boxes_separate():
    graph = build_chain("ab")
    rsm = ebnf_to_rsm("S -> A B\nA -> a\nB -> b")

    assert tensor_based_cfpq(rsm, graph) == {(0, 2)}
    assert tensor_based_cfpq(rsm, build_chain("ba")) == set()


def test_tensor_treats_missing_and_empty_sets_as_all_nodes():
    rsm = cfg_to_rsm(CFG.from_text("S -> a | b"))
    graph = build_chain("ab")

    assert tensor_based_cfpq(rsm, graph) == {(0, 1), (1, 2)}
    assert tensor_based_cfpq(rsm, graph, set(), set()) == {(0, 1), (1, 2)}
    assert tensor_based_cfpq(rsm, graph, None, {2}) == {(1, 2)}
    assert tensor_based_cfpq(rsm, graph, {0}, None) == {(0, 1)}


def test_tensor_accepts_digraph_and_parallel_edges():
    rsm = cfg_to_rsm(CFG.from_text("S -> a b"))
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    assert tensor_based_cfpq(rsm, graph) == {(0, 2)}

    graph = nx.MultiDiGraph(graph)
    graph.add_edge(0, 1, label="b")
    graph.add_edge(0, 1, label="a")
    assert tensor_based_cfpq(rsm, graph) == {(0, 2)}


def test_tensor_uses_rsm_initial_label():
    cfg = CFG.from_text("P -> a P | b\nS -> a", start_symbol=Variable("P"))

    assert tensor_based_cfpq(cfg_to_rsm(cfg), build_chain("aab")) == {
        (0, 3),
        (1, 3),
        (2, 3),
    }


def test_tensor_ignores_unlabeled_edges_and_variable_like_labels():
    graph = build_chain("Sb")
    graph.add_edge(0, 2)
    rsm = cfg_to_rsm(CFG.from_text("S -> S | b"))

    assert tensor_based_cfpq(rsm, graph) == {(1, 2)}


def test_tensor_returns_node_names_not_matrix_indices():
    graph = nx.MultiDiGraph()
    graph.add_edge("x", "y", label="a")
    graph.add_edge("y", 30, label="b")
    graph.add_edge(30, 10, label="b")
    graph.add_node("isolated")
    rsm = cfg_to_rsm(CFG.from_text("S -> a b | b | $"))

    assert tensor_based_cfpq(rsm, graph) == {
        ("x", 30),
        ("y", 30),
        (30, 10),
        *((node, node) for node in graph.nodes),
    }


def test_tensor_does_not_modify_inputs_and_can_reuse_rsm():
    cfg = CFG.from_text("S -> a S b | $")
    original_cfg = deepcopy(cfg)
    rsm = cfg_to_rsm(cfg)
    original_rsm = deepcopy(rsm)
    graph = build_chain("aabb")
    original_graph = deepcopy(graph)
    starts, finals = {0, 1}, {3, 4}

    assert tensor_based_cfpq(rsm, graph, starts, finals) == {(0, 4), (1, 3)}
    assert tensor_based_cfpq(rsm, build_chain("ab"), {0}, {2}) == {(0, 2)}
    assert cfg.productions == original_cfg.productions
    assert cfg.start_symbol == original_cfg.start_symbol
    assert rsm.initial_label == original_rsm.initial_label
    assert rsm == original_rsm
    assert nx.utils.graphs_equal(graph, original_graph)
    assert starts == {0, 1}
    assert finals == {3, 4}
