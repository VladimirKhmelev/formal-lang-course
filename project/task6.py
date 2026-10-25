from collections import defaultdict

import networkx as nx
from pyformlang.cfg import CFG, Production

from project.task2 import graph_to_nfa


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Converts a grammar to weak Chomsky normal form: rules A -> B C, A -> a and
    A -> epsilon."""

    # pyformlang's to_normal_form skips normalization for rules like S -> S,
    # so unit productions are eliminated first.
    productions = set(cfg.eliminate_unit_productions().to_normal_form().productions)
    if cfg.generate_epsilon():
        productions.add(Production(cfg.start_symbol, []))
    return CFG(start_symbol=cfg.start_symbol, productions=productions)


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Returns pairs of nodes connected by a path whose labels form a word
    derivable from the start symbol of the grammar."""

    grammar = cfg_to_weak_normal_form(cfg)
    nfa = graph_to_nfa(graph, set(start_nodes or ()), set(final_nodes or ()))

    epsilon_heads = set()
    terminal_heads = defaultdict(set)
    heads_by_left = defaultdict(list)
    heads_by_right = defaultdict(list)
    for production in grammar.productions:
        body = production.body
        if not body:
            epsilon_heads.add(production.head)
        elif len(body) == 1:
            terminal_heads[body[0].value].add(production.head)
        else:
            heads_by_left[body[0]].append((production.head, body[1]))
            heads_by_right[body[1]].append((production.head, body[0]))

    queue = []
    ends_from = defaultdict(set)
    starts_to = defaultdict(set)

    def derive(variable, source, target):
        if target not in ends_from[source, variable]:
            ends_from[source, variable].add(target)
            starts_to[target, variable].add(source)
            queue.append((variable, source, target))

    for source, symbol, target in nfa:
        for head in terminal_heads[symbol.value]:
            derive(head, source.value, target.value)
    for state in nfa.states:
        for head in epsilon_heads:
            derive(head, state.value, state.value)

    while queue:
        variable, source, target = queue.pop()
        for head, right in heads_by_left[variable]:
            for end in list(ends_from[target, right]):
                derive(head, source, end)
        for head, left in heads_by_right[variable]:
            for start in list(starts_to[source, left]):
                derive(head, start, target)

    finals = {state.value for state in nfa.final_states}
    return {
        (state.value, target)
        for state in nfa.start_states
        for target in ends_from[state.value, grammar.start_symbol] & finals
    }
