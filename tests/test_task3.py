import networkx as nx
import pytest
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata, tensor_based_rpq


@pytest.mark.parametrize(
    "regex,accepted,rejected",
    [
        ("a b*", [["a"], ["a", "b", "b"]], [[], ["b"], ["b", "a"]]),
        ("(a|b)*c", [["c"], ["a", "b", "c"]], [[], ["a"], ["c", "c"]]),
        ("a*", [[], ["a"], ["a", "a"]], [["b"], ["a", "b"]]),
    ],
)
def test_accepts_matches_regex_language(regex, accepted, rejected):
    fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    for word in accepted:
        assert fa.accepts(word)

    for word in rejected:
        assert not fa.accepts(word)


def test_accepts_rejects_unknown_symbol():
    fa = AdjacencyMatrixFA(regex_to_dfa("a b"))

    assert not fa.accepts(["a", "z"])


def test_accepts_on_nondeterministic_automaton():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 2, label="a")
    graph.add_edge(2, 3, label="b")

    fa = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {1, 3}))

    assert fa.accepts(["a"])
    assert fa.accepts(["a", "b"])
    assert not fa.accepts(["b"])


@pytest.mark.parametrize(
    "regex,word",
    [("a", ["a"]), ("a b", ["a", "b"]), ("(a|b)*", []), ("a* b", ["b"])],
)
def test_non_empty_language_has_accepted_word(regex, word):
    fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    assert fa.accepts(word)
    assert not fa.is_empty()


def test_is_empty_for_automaton_without_final_states():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transition(State(0), Symbol("a"), State(1))
    nfa.add_start_state(State(0))

    assert AdjacencyMatrixFA(nfa).is_empty()


def test_is_empty_for_unreachable_final_state():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_node(2)

    fa = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {2}))

    assert fa.is_empty()


def test_intersect_automata_accepts_common_words():
    fa1 = AdjacencyMatrixFA(regex_to_dfa("(a|b)*"))
    fa2 = AdjacencyMatrixFA(regex_to_dfa("a*"))

    intersection = intersect_automata(fa1, fa2)

    assert intersection.accepts(["a", "a"])
    assert not intersection.accepts(["b"])


def test_intersect_automata_indexes_every_pair_of_states():
    dfa1 = regex_to_dfa("a b")
    dfa2 = regex_to_dfa("(a|b)*")

    intersection = intersect_automata(AdjacencyMatrixFA(dfa1), AdjacencyMatrixFA(dfa2))

    assert sorted(intersection.state_to_index.values()) == list(
        range(intersection.states_count)
    )
    start_pairs = {
        State((s1.value, s2.value))
        for s1 in dfa1.start_states
        for s2 in dfa2.start_states
    }
    assert {
        intersection.state_to_index[state] for state in start_pairs
    } == intersection.start_states


def test_intersect_automata_of_disjoint_languages_is_empty():
    fa1 = AdjacencyMatrixFA(regex_to_dfa("a a"))
    fa2 = AdjacencyMatrixFA(regex_to_dfa("b b"))

    assert intersect_automata(fa1, fa2).is_empty()


def test_intersect_automata_without_shared_symbols_has_no_matrices():
    fa1 = AdjacencyMatrixFA(regex_to_dfa("a"))
    fa2 = AdjacencyMatrixFA(regex_to_dfa("b"))

    intersection = intersect_automata(fa1, fa2)

    assert intersection.matrices == {}
    assert intersection.states_count == fa1.states_count * fa2.states_count


def test_empty_automaton_is_empty():
    fa = AdjacencyMatrixFA()

    assert fa.states_count == 0
    assert fa.is_empty()
    assert not fa.accepts(["a"])


def test_accepts_takes_symbols_as_well_as_strings():
    fa = AdjacencyMatrixFA(regex_to_dfa("a b"))

    assert fa.accepts([Symbol("a"), Symbol("b")])
    assert fa.accepts(["a", "b"])


def build_cycle_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="a")
    return graph


def test_tensor_based_rpq_on_cycle():
    graph = build_cycle_graph()

    assert tensor_based_rpq("a", graph, {0}, {1, 2}) == {(0, 1)}
    assert tensor_based_rpq("a b", graph, {0}, {2}) == {(0, 2)}


def test_tensor_based_rpq_empty_word_connects_node_to_itself():
    graph = build_cycle_graph()

    assert tensor_based_rpq("a*", graph, {1}, {1}) == {(1, 1)}


def test_tensor_based_rpq_without_matching_paths():
    graph = build_cycle_graph()

    assert tensor_based_rpq("b b", graph, {0}, {1, 2}) == set()


def test_tensor_based_rpq_distinguishes_parallel_edges_by_label():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")

    assert tensor_based_rpq("a", graph, {0}, {1}) == {(0, 1)}
    assert tensor_based_rpq("b", graph, {0}, {1}) == {(0, 1)}
    assert tensor_based_rpq("c", graph, {0}, {1}) == set()


def test_tensor_based_rpq_follows_self_loop_twice():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 0, label="a")

    assert tensor_based_rpq("a a", graph, {0}, {0}) == {(0, 0)}


def test_tensor_based_rpq_keeps_components_separate():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(2, 3, label="a")

    assert tensor_based_rpq("a", graph, {0, 2}, {1, 3}) == {(0, 1), (2, 3)}


@pytest.mark.parametrize("regex", ["(a|b)* a", "a b a", "b*", "(a b)*"])
def test_tensor_based_rpq_agrees_with_pyformlang_intersection(regex):
    graph = build_cycle_graph()
    graph.add_edge(1, 1, label="b")
    nodes = set(graph.nodes)

    expected = {
        (start, final)
        for start in nodes
        for final in nodes
        if not (graph_to_nfa(graph, {start}, {final}) & regex_to_dfa(regex)).is_empty()
    }

    assert tensor_based_rpq(regex, graph, nodes, nodes) == expected
