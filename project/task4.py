import numpy as np
from networkx import MultiDiGraph
from scipy.sparse import csr_array, eye_array, kron

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """Solves RPQ for several start nodes at once by a breadth-first search over
    the product of the graph and the regex DFA, expressed in sparse matrices."""

    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_fa = AdjacencyMatrixFA(
        graph_to_nfa(graph, set(start_nodes), set(final_nodes))
    )

    starts = list(graph_fa.start_states)
    regex_size = regex_fa.states_count

    rows, cols = [], []
    for block, start in enumerate(starts):
        for state in regex_fa.start_states:
            rows.append(block * regex_size + state)
            cols.append(start)
    front = csr_array(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(len(starts) * regex_size, graph_fa.states_count),
        dtype=bool,
    )

    steps = [
        (
            graph_fa.matrices[symbol],
            kron(eye_array(len(starts), dtype=bool), regex_matrix.T, format="csr"),
        )
        for symbol, regex_matrix in regex_fa.matrices.items()
        if symbol in graph_fa.matrices
    ]

    visited = front.toarray()
    while front.nnz:
        reached = csr_array(front.shape, dtype=bool)
        for graph_matrix, regex_step in steps:
            reached = reached + regex_step @ (front @ graph_matrix)
        reached = reached.tocoo()

        new = ~visited[reached.row, reached.col]
        rows, cols = reached.row[new], reached.col[new]
        visited[rows, cols] = True
        front = csr_array(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=front.shape,
            dtype=bool,
        )

    finals = sorted(graph_fa.final_states)
    accepted = visited.reshape(len(starts), regex_size, graph_fa.states_count)[
        :, sorted(regex_fa.final_states)
    ].any(axis=1)[:, finals]

    index_to_node = {
        index: state.value for state, index in graph_fa.state_to_index.items()
    }
    blocks, positions = np.nonzero(accepted)

    return {
        (index_to_node[starts[block]], index_to_node[finals[position]])
        for block, position in zip(blocks.tolist(), positions.tolist())
    }
