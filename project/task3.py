from collections.abc import Iterable

import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy.sparse import csr_array, eye_array, kron

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    """Finite automaton stored as a boolean decomposition of its adjacency matrix:
    one sparse matrix per symbol, plus start and final state indices."""

    def __init__(self, automaton: NondeterministicFiniteAutomaton | None = None):
        if automaton is None:
            self.states_count = 0
            self.state_to_index = {}
            self.start_states = set()
            self.final_states = set()
            self.matrices = {}
            return

        states = list(automaton.states)
        self.states_count = len(states)
        self.state_to_index = {state: index for index, state in enumerate(states)}
        self.start_states = {self.state_to_index[s] for s in automaton.start_states}
        self.final_states = {self.state_to_index[s] for s in automaton.final_states}

        transitions: dict[Symbol, list[tuple[int, int]]] = {}
        for source, symbol, target in automaton:
            transitions.setdefault(symbol, []).append(
                (self.state_to_index[source], self.state_to_index[target])
            )

        self.matrices = {
            symbol: self._build_matrix(edges, self.states_count)
            for symbol, edges in transitions.items()
        }

    @staticmethod
    def _build_matrix(edges: list[tuple[int, int]], size: int) -> csr_array:
        rows, cols = zip(*edges)
        data = np.ones(len(edges), dtype=bool)
        return csr_array((data, (rows, cols)), shape=(size, size), dtype=bool)

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current = np.zeros(self.states_count, dtype=bool)
        current[list(self.start_states)] = True

        for symbol in word:
            key = symbol if isinstance(symbol, Symbol) else Symbol(symbol)
            matrix = self.matrices.get(key)
            if matrix is None:
                return False
            current = current @ matrix
            if not current.any():
                return False

        return any(current[final] for final in self.final_states)

    def transitive_closure(self) -> csr_array:
        closure = eye_array(self.states_count, dtype=bool, format="csr")
        for matrix in self.matrices.values():
            closure = closure + matrix
        closure = closure.astype(bool)

        nnz = -1
        while closure.nnz != nnz:
            nnz = closure.nnz
            closure = (closure @ closure).astype(bool)
        return closure

    def is_empty(self) -> bool:
        if not self.start_states or not self.final_states:
            return True

        closure = self.transitive_closure()
        return closure[list(self.start_states)][:, list(self.final_states)].nnz == 0


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Intersects two automata by taking the tensor product of their
    boolean decompositions."""

    result = AdjacencyMatrixFA()
    result.states_count = automaton1.states_count * automaton2.states_count
    size2 = automaton2.states_count

    result.state_to_index = {
        State((state1.value, state2.value)): index1 * size2 + index2
        for state1, index1 in automaton1.state_to_index.items()
        for state2, index2 in automaton2.state_to_index.items()
    }

    result.matrices = {
        symbol: kron(matrix, automaton2.matrices[symbol], format="csr").astype(bool)
        for symbol, matrix in automaton1.matrices.items()
        if symbol in automaton2.matrices
    }

    result.start_states = {
        s1 * size2 + s2
        for s1 in automaton1.start_states
        for s2 in automaton2.start_states
    }
    result.final_states = {
        f1 * size2 + f2
        for f1 in automaton1.final_states
        for f2 in automaton2.final_states
    }
    return result


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """Returns pairs of graph nodes connected by a path whose labels form
    a word from the language of the given regular expression."""

    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_fa = AdjacencyMatrixFA(
        graph_to_nfa(graph, set(start_nodes), set(final_nodes))
    )

    intersection = intersect_automata(graph_fa, regex_fa)
    closure = intersection.transitive_closure().tocoo()

    reachable = np.isin(closure.row, list(intersection.start_states)) & np.isin(
        closure.col, list(intersection.final_states)
    )

    # Intersection states are (graph node, regex state) pairs.
    index_to_node = {
        index: state.value[0] for state, index in intersection.state_to_index.items()
    }

    return {
        (index_to_node[start], index_to_node[final])
        for start, final in zip(
            closure.row[reachable].tolist(), closure.col[reachable].tolist()
        )
    }
