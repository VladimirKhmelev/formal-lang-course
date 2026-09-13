from pathlib import Path

import pydot
import pytest
from networkx import is_isomorphic
from networkx.drawing.nx_pydot import read_dot

from project.task1 import create_and_save_two_cycles_graph, get_graph_info

RESOURCES = Path(__file__).parent / "resources" / "task1"


def test_get_graph_info_from_unknown_name():
    with pytest.raises(FileNotFoundError):
        get_graph_info("this name does not exist")


@pytest.mark.parametrize(
    "graph_name,number_of_nodes,number_of_edges,labels",
    [
        pytest.param(
            "skos",
            144,
            252,
            {
                "comment",
                "contributor",
                "creator",
                "definition",
                "description",
                "disjointWith",
                "domain",
                "example",
                "first",
                "inverseOf",
                "isDefinedBy",
                "label",
                "range",
                "rest",
                "scopeNote",
                "seeAlso",
                "subClassOf",
                "subPropertyOf",
                "title",
                "type",
                "unionOf",
            },
            id="skos",
        ),
        pytest.param(
            "generations",
            129,
            273,
            {
                "type",
                "first",
                "rest",
                "onProperty",
                "intersectionOf",
                "equivalentClass",
                "someValuesFrom",
                "hasValue",
                "hasSex",
                "hasChild",
                "hasParent",
                "inverseOf",
                "sameAs",
                "hasSibling",
                "oneOf",
                "range",
                "versionInfo",
            },
            id="generations",
        ),
        pytest.param("bzip", 632, 556, {"d", "a"}, id="bzip"),
        pytest.param("wc", 332, 269, {"d", "a"}, id="wc"),
        pytest.param("pr", 815, 692, {"d", "a"}, id="pr"),
    ],
)
def test_get_graph_info(graph_name, number_of_nodes, number_of_edges, labels):
    info = get_graph_info(graph_name)

    assert info.number_of_nodes == number_of_nodes
    assert info.number_of_edges == number_of_edges
    assert info.labels == labels


@pytest.mark.parametrize(
    "first_cycle_nodes,second_cycle_nodes", [(0, 0), (1, 0), (0, 1)]
)
def test_create_and_save_two_cycles_graph_zero_sized_cycle(
    tmp_path, first_cycle_nodes, second_cycle_nodes
):
    with pytest.raises(IndexError):
        create_and_save_two_cycles_graph(
            first_cycle_nodes, second_cycle_nodes, ("a", "b"), tmp_path / "graph.dot"
        )


@pytest.mark.parametrize(
    "first_cycle_nodes,second_cycle_nodes,labels",
    [
        (3, 2, ("a", "b")),
        (5, 5, ("x", "y")),
        (1, 1, ("first", "second")),
    ],
)
def test_create_and_save_two_cycles_graph(
    tmp_path, first_cycle_nodes, second_cycle_nodes, labels
):
    output_path = tmp_path / "graph.dot"

    create_and_save_two_cycles_graph(
        first_cycle_nodes, second_cycle_nodes, labels, output_path
    )

    assert output_path.exists()

    (loaded_graph,) = pydot.graph_from_dot_file(str(output_path))

    expected_nodes = first_cycle_nodes + second_cycle_nodes + 1
    expected_edges = first_cycle_nodes + second_cycle_nodes + 2

    assert len(loaded_graph.get_nodes()) == expected_nodes
    assert len(loaded_graph.get_edges()) == expected_edges

    edge_labels = {edge.get_attributes()["label"] for edge in loaded_graph.get_edges()}
    assert edge_labels == set(labels)


@pytest.mark.parametrize(
    "first_cycle_nodes,second_cycle_nodes,labels,expected_dot_file",
    [
        (2, 4, ("x", "y"), "two_cycles_2_4.dot"),
        (6, 2, ("p", "q"), "two_cycles_6_2.dot"),
        (4, 9, ("foo", "bar"), "two_cycles_4_9.dot"),
    ],
)
def test_create_and_save_two_cycles_graph_matches_expected_dot(
    tmp_path, first_cycle_nodes, second_cycle_nodes, labels, expected_dot_file
):
    output_path = tmp_path / "graph.dot"

    create_and_save_two_cycles_graph(
        first_cycle_nodes, second_cycle_nodes, labels, output_path
    )

    expected_graph = read_dot(RESOURCES / expected_dot_file)
    actual_graph = read_dot(output_path)

    assert is_isomorphic(
        expected_graph,
        actual_graph,
        node_match=lambda n1, n2: dict(n1) == dict(n2),
        edge_match=lambda e1, e2: dict(e1) == dict(e2),
    )
