import cfpq_data
import networkx as nx
import pytest
from pyformlang.finite_automaton import State
from pyformlang.regular_expression import MisformedRegexError

from project.task1 import get_graph_info
from project.task2 import graph_to_nfa, regex_to_dfa


@pytest.mark.parametrize(
    "regex,accepted,rejected",
    [
        ("a b*", [["a"], ["a", "b"], ["a", "b", "b"]], [[], ["b"], ["a", "a"]]),
        ("(a|b)*c", [["c"], ["a", "b", "a", "c"]], [[], ["a", "b"]]),
        ("a*", [[], ["a"], ["a", "a", "a"]], [["b"]]),
    ],
)
def test_regex_to_dfa_language(regex, accepted, rejected):
    dfa = regex_to_dfa(regex)

    assert dfa.is_deterministic()

    for word in accepted:
        assert dfa.accepts(word)

    for word in rejected:
        assert not dfa.accepts(word)


@pytest.mark.parametrize("regex", ["a b*", "(a|b)*c", "a*", "(a b) | (a c)"])
def test_regex_to_dfa_is_minimal(regex):
    dfa = regex_to_dfa(regex)

    assert len(dfa.states) == len(dfa.minimize().states)


@pytest.mark.parametrize("regex", ["a(", "*a", "(a|b"])
def test_regex_to_dfa_malformed_regex_raises(regex):
    with pytest.raises(MisformedRegexError):
        regex_to_dfa(regex)


def build_path_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    return graph


def test_graph_to_nfa_with_explicit_start_and_final():
    nfa = graph_to_nfa(build_path_graph(), {0}, {2})

    assert nfa.accepts(["a", "b"])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["b"])
    assert nfa.start_states == {State(0)}
    assert nfa.final_states == {State(2)}


def test_graph_to_nfa_defaults_to_all_states_when_not_specified():
    nfa = graph_to_nfa(build_path_graph(), set(), set())

    assert nfa.start_states == {State(0), State(1), State(2)}
    assert nfa.final_states == {State(0), State(1), State(2)}
    assert nfa.accepts(["a", "b"])
    assert nfa.accepts(["b"])
    assert nfa.accepts([])


def test_graph_to_nfa_keeps_isolated_nodes():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.states == {State(0), State(1), State(2)}
    assert nfa.accepts([])
    assert not nfa.accepts(["a"])


def test_graph_to_nfa_self_loop_accepts_empty_word():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 0, label="a")

    nfa = graph_to_nfa(graph, {0}, {0})

    assert nfa.accepts([])
    assert nfa.accepts(["a"])
    assert nfa.accepts(["a", "a"])


def test_regex_to_dfa_matches_graph_to_nfa_on_star_loop():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 1, label="b")

    dfa = regex_to_dfa("(a b)*")
    nfa = graph_to_nfa(graph, {1}, {1})

    words = [[], ["a"], ["b"], ["a", "b"], ["a", "b", "a", "b"], ["a", "a"]]
    for word in words:
        assert dfa.accepts(word) == nfa.accepts(word)


def test_graph_to_nfa_matches_get_graph_info_on_real_dataset_graph():
    info = get_graph_info("generations")
    graph_path = cfpq_data.download("generations")
    graph = cfpq_data.graph_from_csv(graph_path)

    nfa = graph_to_nfa(graph, set(), set())

    assert len(nfa.states) == info.number_of_nodes
    assert nfa.get_number_transitions() == info.number_of_edges
    assert {str(symbol) for symbol in nfa.symbols} == info.labels
