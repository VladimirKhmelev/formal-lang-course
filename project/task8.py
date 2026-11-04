import networkx as nx
import numpy as np
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import csr_array

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Builds one box per variable directly from its production bodies.

    Each body is a separate path from the box's start state to a final state;
    an empty production makes the start state final. No normal form is needed.
    Box labels are renamed when they conflict with literal terminal values.
    A grammar without a start symbol raises ValueError.
    """

    if cfg.start_symbol is None:
        raise ValueError("CFG must have a start symbol")

    variables = cfg.variables | {cfg.start_symbol}
    terminal_values = {terminal.value for terminal in cfg.terminals}
    used_labels = terminal_values | {variable.value for variable in variables}
    call_labels = {}
    counter = 0
    for variable in variables:
        label = variable.value
        if label in terminal_values:
            while f"__RSM_CALL_{counter}" in used_labels:
                counter += 1
            label = f"__RSM_CALL_{counter}"
            used_labels.add(label)
        call_labels[variable] = Symbol(label)

    automata = {
        variable: NondeterministicFiniteAutomaton(start_state={State(0)})
        for variable in variables
    }
    for index, production in enumerate(cfg.productions):
        automaton = automata[production.head]
        current = State(0)
        for position, symbol in enumerate(production.body):
            target = State((index, position))
            label = (
                call_labels[symbol]
                if isinstance(symbol, Variable)
                else Symbol(symbol.value)
            )
            automaton.add_transition(current, label, target)
            current = target
        automaton.add_final_state(current)

    return RecursiveAutomaton(
        initial_label=call_labels[cfg.start_symbol],
        boxes={
            Box(automaton, call_labels[variable])
            for variable, automaton in automata.items()
        },
    )


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Builds an RSM with entry S from EBNF rules of the form A -> regex.

    Each right-hand side defines its box's DFA, with box names denoting recursive
    calls. Blank lines are ignored; an empty right-hand side denotes epsilon.
    """

    # The library silently skips lines without an arrow instead of rejecting them.
    if any(line.strip() and "->" not in line for line in ebnf.splitlines()):
        raise ValueError("EBNF rules must have the form head -> regex")
    return RecursiveAutomaton.from_text(ebnf)


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Evaluates an RSM by tensor intersection with the graph until a fixpoint.

    Accepted paths inside each box add graph edges labeled by that box's
    variable. Reflexive closure also handles boxes accepting the empty word.
    Missing or empty endpoint sets mean all graph nodes. Inputs are not modified.
    A nonempty RSM whose initial label has no box raises ValueError.
    """

    if not rsm.boxes:
        return set()
    if rsm.initial_label not in rsm.boxes:
        raise ValueError("RSM initial label must refer to an existing box")

    graph_fa = AdjacencyMatrixFA(
        graph_to_nfa(graph, set(start_nodes or ()), set(final_nodes or ()))
    )
    size = graph_fa.states_count
    if not size:
        return set()

    # Box labels denote calls, so graph edges cannot supply their derivations.
    for label in rsm.labels:
        graph_fa.matrices[label] = csr_array((size, size), dtype=bool)

    # State names may repeat across boxes; prefix them with their box labels.
    automaton = NondeterministicFiniteAutomaton(
        states={
            State((label.value, state.value))
            for label, box in rsm.boxes.items()
            for state in box.dfa.states
        }
    )
    for label, box in rsm.boxes.items():
        for source, symbol, target in box.dfa:
            automaton.add_transition(
                State((label.value, source.value)),
                symbol,
                State((label.value, target.value)),
            )
    rsm_fa = AdjacencyMatrixFA(automaton)
    boundaries = {
        label: (
            [
                rsm_fa.state_to_index[State((label.value, state.value))]
                for state in box.dfa.start_states
            ],
            [
                rsm_fa.state_to_index[State((label.value, state.value))]
                for state in box.dfa.final_states
            ],
        )
        for label, box in rsm.boxes.items()
    }

    while True:
        closure = intersect_automata(rsm_fa, graph_fa).transitive_closure()
        changed = False
        for label, (starts, finals) in boundaries.items():
            updated = graph_fa.matrices[label]
            for start in starts:
                for final in finals:
                    updated = (
                        updated
                        + closure[
                            start * size : (start + 1) * size,
                            final * size : (final + 1) * size,
                        ]
                    )
            if updated.nnz != graph_fa.matrices[label].nnz:
                graph_fa.matrices[label] = updated
                changed = True
        if not changed:
            break

    found = graph_fa.matrices[rsm.initial_label].tocoo()
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
