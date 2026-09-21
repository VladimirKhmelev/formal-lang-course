from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Builds a minimal DFA equivalent to the given regular expression."""

    enfa = Regex(regex).to_epsilon_nfa()
    return enfa.to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NondeterministicFiniteAutomaton:
    """Builds an NFA from a graph. Nodes not specified as start/final states
    default to all nodes being both start and final when neither set is given."""

    nfa = NondeterministicFiniteAutomaton(
        states={State(node) for node in graph.nodes},
        start_state={State(s) for s in start_states or graph.nodes},
        final_states={State(s) for s in final_states or graph.nodes},
    )

    for u, v, label in graph.edges(data="label"):
        if label is not None:
            nfa.add_transition(State(u), Symbol(label), State(v))

    return nfa
