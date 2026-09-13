from dataclasses import dataclass
from pathlib import Path

import cfpq_data
import networkx as nx


@dataclass
class GraphInfo:
    """Basic statistics about a graph: number of nodes, edges and edge labels."""

    number_of_nodes: int
    number_of_edges: int
    labels: set[str]


def get_graph_info(graph_name: str) -> GraphInfo:
    """Loads a graph by name from the CFPQ_Data dataset and returns its statistics."""

    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)

    return GraphInfo(
        number_of_nodes=graph.number_of_nodes(),
        number_of_edges=graph.number_of_edges(),
        labels=set(cfpq_data.get_sorted_labels(graph)),
    )


def create_and_save_two_cycles_graph(
    first_cycle_nodes: int,
    second_cycle_nodes: int,
    labels: tuple[str, str],
    output_path: str | Path,
) -> None:
    """Builds a labeled two-cycles graph and saves it to `output_path` in DOT format."""

    graph = cfpq_data.labeled_two_cycles_graph(
        first_cycle_nodes, second_cycle_nodes, labels=labels
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pydot_graph = nx.nx_pydot.to_pydot(graph)
    pydot_graph.write_raw(str(output_path))
