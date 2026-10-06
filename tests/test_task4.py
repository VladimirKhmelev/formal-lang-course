import random

import cfpq_data
import networkx as nx
import pytest

from project.task3 import tensor_based_rpq
from project.task4 import ms_bfs_based_rpq


def build_cycle_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="a")
    graph.add_edge(1, 1, label="b")
    return graph


def test_ms_bfs_keeps_start_nodes_apart():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(2, 3, label="b")

    assert ms_bfs_based_rpq("a", graph, {0, 2}, {1, 3}) == {(0, 1)}
    assert ms_bfs_based_rpq("b", graph, {0, 2}, {1, 3}) == {(2, 3)}


def test_ms_bfs_start_nodes_share_reachable_node():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 2, label="a")
    graph.add_edge(1, 2, label="a")

    assert ms_bfs_based_rpq("a", graph, {0, 1}, {2}) == {(0, 2), (1, 2)}


def test_ms_bfs_empty_word_connects_final_start_node_to_itself():
    graph = build_cycle_graph()

    assert ms_bfs_based_rpq("a*", graph, {1}, {1}) == {(1, 1)}


def test_ms_bfs_empty_word_requires_start_node_to_be_final():
    graph = build_cycle_graph()

    assert ms_bfs_based_rpq("b*", graph, {0}, {1, 2}) == set()


def test_ms_bfs_follows_loop_several_times():
    graph = build_cycle_graph()

    assert ms_bfs_based_rpq("a b b b", graph, {0}, {2}) == {(0, 2)}

    graph.remove_edge(1, 1)
    assert ms_bfs_based_rpq("a b b b", graph, {0}, {2}) == set()


def test_ms_bfs_returns_to_start_through_cycle():
    graph = build_cycle_graph()

    assert ms_bfs_based_rpq("a b a", graph, {0}, {0}) == {(0, 0)}


def test_ms_bfs_treats_empty_start_and_final_sets_as_all_nodes():
    graph = build_cycle_graph()

    assert ms_bfs_based_rpq("a", graph, set(), set()) == {(0, 1), (2, 0)}


def test_ms_bfs_regex_with_several_final_states():
    graph = build_cycle_graph()

    assert ms_bfs_based_rpq("a | a b", graph, {0}, set()) == {(0, 1), (0, 2)}


def test_ms_bfs_on_empty_graph():
    assert ms_bfs_based_rpq("a*", nx.MultiDiGraph(), set(), set()) == set()


def test_ms_bfs_without_matching_paths():
    graph = build_cycle_graph()

    assert ms_bfs_based_rpq("b a b", graph, {0, 1, 2}, {0, 1, 2}) == set()


def test_ms_bfs_regex_symbols_absent_from_graph():
    graph = build_cycle_graph()

    assert ms_bfs_based_rpq("c", graph, {0}, {1}) == set()


def test_ms_bfs_empty_language():
    graph = build_cycle_graph()
    nodes = set(graph.nodes)

    assert ms_bfs_based_rpq("", graph, nodes, nodes) == set()


@pytest.mark.parametrize(
    "regex", ["a", "a b", "(a|b)* a", "b*", "(a b)*", "a b* a", "$"]
)
def test_ms_bfs_agrees_with_tensor_based_rpq(regex):
    graph = build_cycle_graph()
    nodes = set(graph.nodes)

    for starts in ({0}, {1, 2}, nodes):
        assert ms_bfs_based_rpq(regex, graph, starts, nodes) == tensor_based_rpq(
            regex, graph, starts, nodes
        )


@pytest.mark.parametrize("seed", range(10))
def test_ms_bfs_agrees_with_tensor_based_rpq_on_random_graph(seed):
    rng = random.Random(seed)
    graph = cfpq_data.labeled_binomial_graph(
        30, 0.1, labels=["a", "b", "c"], choice=rng.choice, seed=seed
    )
    nodes = sorted(graph.nodes)
    starts = set(rng.sample(nodes, rng.randint(1, len(nodes))))
    finals = set(rng.sample(nodes, rng.randint(1, len(nodes))))

    for regex in ["(a|b)* c", "a b* c*", "(a b | c)*", "a | b | c"]:
        assert ms_bfs_based_rpq(regex, graph, starts, finals) == tensor_based_rpq(
            regex, graph, starts, finals
        )


@pytest.fixture(scope="module")
def generations_graph() -> nx.MultiDiGraph:
    return cfpq_data.graph_from_csv(cfpq_data.download("generations"))


@pytest.mark.parametrize(
    "regex", ["type", "rest* first", "(first | rest)* type", "onProperty type*"]
)
def test_ms_bfs_agrees_with_tensor_based_rpq_on_dataset_graph(generations_graph, regex):
    starts = set(random.Random(0).sample(sorted(generations_graph.nodes), 30))
    nodes = set(generations_graph.nodes)

    assert ms_bfs_based_rpq(
        regex, generations_graph, starts, nodes
    ) == tensor_based_rpq(regex, generations_graph, starts, nodes)
