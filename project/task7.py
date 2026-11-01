import networkx as nx
import numpy as np
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import Symbol
from scipy.sparse import csr_array, eye_array

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA
from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Returns pairs of nodes connected by a path whose labels form a word
    derivable from the start symbol, using one reachability matrix per variable."""

    grammar = cfg_to_weak_normal_form(cfg)
    graph_fa = AdjacencyMatrixFA(
        graph_to_nfa(graph, set(start_nodes or ()), set(final_nodes or ()))
    )
    size = graph_fa.states_count

    matrices = {
        variable: csr_array((size, size), dtype=bool) for variable in grammar.variables
    }
    binary_rules = []
    for production in grammar.productions:
        head, body = production.head, production.body
        if not body:
            matrices[head] += eye_array(size, dtype=bool, format="csr")
        elif len(body) == 1:
            label_matrix = graph_fa.matrices.get(Symbol(body[0].value))
            if label_matrix is not None:
                matrices[head] += label_matrix
        else:
            binary_rules.append((head, *body))

    # A rule is recomputed only if one of its operands grew since its last run.
    changed = set(matrices)
    while changed:
        recent, changed = changed, set()
        for head, left, right in binary_rules:
            if {left, right} & (recent | changed):
                updated = matrices[head] + matrices[left] @ matrices[right]
                if updated.nnz != matrices[head].nnz:
                    matrices[head] = updated
                    changed.add(head)

    found = matrices[grammar.start_symbol].tocoo()
    accepted = np.isin(found.row, list(graph_fa.start_states)) & np.isin(
        found.col, list(graph_fa.final_states)
    )
    index_to_node = {
        index: state.value for state, index in graph_fa.state_to_index.items()
    }
    return {
        (index_to_node[source], index_to_node[target])
        for source, target in zip(
            found.row[accepted].tolist(), found.col[accepted].tolist()
        )
    }
