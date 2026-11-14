from collections import defaultdict

import networkx as nx
from pyformlang.rsa import RecursiveAutomaton

from project.task2 import graph_to_nfa


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Evaluates an RSM with GLL descriptors and a graph-structured stack.

    A descriptor is (graph vertex, box state, GSS node). Each GSS node
    (box label, entry vertex) shares a call; its edges store return states and
    caller nodes. Completed returns are replayed when a new edge is added.
    Endpoint sets filter the query, not the intermediate calls; None and empty
    sets select all graph nodes. Inputs are not modified.
    """

    if not rsm.boxes:
        return set()
    if rsm.initial_label not in rsm.boxes:
        raise ValueError("RSM initial label must refer to an existing box")

    nfa = graph_to_nfa(graph, set(start_nodes or ()), set(final_nodes or ()))
    starts = {state.value for state in nfa.start_states}
    finals = {state.value for state in nfa.final_states}
    adjacency = defaultdict(set)
    for source, symbol, target in nfa:
        adjacency[source.value, symbol].add(target.value)

    automata = {label: box.dfa for label, box in rsm.boxes.items()}
    terminal_moves = defaultdict(list)
    call_moves = defaultdict(list)
    for label, automaton in automata.items():
        for source, symbol, target in automaton:
            moves = call_moves if symbol in automata else terminal_moves
            moves[label, source].append((symbol, target))

    pending = []
    seen = set()
    callers = {}
    completed = defaultdict(set)

    def add(vertex, state, stack):
        descriptor = (vertex, state, stack)
        if descriptor not in seen:
            seen.add(descriptor)
            pending.append(descriptor)

    def create(label, vertex):
        stack = (label, vertex)
        if stack not in callers:
            callers[stack] = set()
            for state in automata[label].start_states:
                add(vertex, state, stack)
        return stack

    def link(callee, return_state, caller):
        edge = (return_state, caller)
        if edge not in callers[callee]:
            callers[callee].add(edge)
            for end in completed[callee]:
                add(end, return_state, caller)

    def pop(stack, vertex):
        if vertex not in completed[stack]:
            completed[stack].add(vertex)
            for return_state, caller in callers[stack]:
                add(vertex, return_state, caller)

    # Initialize one root call for each requested start vertex.
    for start in starts:
        create(rsm.initial_label, start)

    while pending:
        vertex, state, stack = pending.pop()
        label, _ = stack

        for symbol, target in terminal_moves[label, state]:
            for end in adjacency.get((vertex, symbol), ()):
                add(end, target, stack)

        for symbol, target in call_moves[label, state]:
            callee = create(symbol, vertex)
            link(callee, target, stack)

        if state in automata[label].final_states:
            pop(stack, vertex)

    return {
        (start, end)
        for start in starts
        for end in completed[rsm.initial_label, start] & finals
    }
