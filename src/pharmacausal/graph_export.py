"""Convert causal-learn's internal graph representations into a readable
edge table, and pull out the drug<->event edges specifically (the ones the
DrugBank validation step and the visualization step actually care about).

PC output: `CausalGraph.G.graph` is an (n, n) ndarray.
    graph[j, i] == 1 and graph[i, j] == -1  =>  i --> j
    graph[i, j] == graph[j, i] == -1        =>  i --- j   (undirected)
    graph[i, j] == graph[j, i] == 1         =>  i <-> j   (conflicting orientation)

FCI output: a list of `Edge` objects on a PAG (partial ancestral graph).
    Each edge has two endpoints, each one of TAIL / ARROW / CIRCLE.
    o-> means "uncertain whether there's a latent common cause, but this
    node is not caused by the other" -- the hallmark FCI mark that has no
    PC equivalent, and the reason FCI's output can represent "we can't rule
    out an unmeasured confounder here" instead of forcing an edge to look
    more certain than the data support.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def pc_graph_to_edges(node_names: list[str], graph) -> pd.DataFrame:
    rows = []
    n = len(node_names)
    for i in range(n):
        for j in range(i + 1, n):
            gij, gji = graph[i, j], graph[j, i]
            if gij == 0 and gji == 0:
                continue
            if gij == 1 and gji == -1:
                relation = f"{node_names[j]} --> {node_names[i]}"
            elif gij == -1 and gji == 1:
                relation = f"{node_names[i]} --> {node_names[j]}"
            elif gij == -1 and gji == -1:
                relation = f"{node_names[i]} --- {node_names[j]}"
            elif gij == 1 and gji == 1:
                relation = f"{node_names[i]} <-> {node_names[j]}"
            else:
                relation = f"{node_names[i]} ?? {node_names[j]} (raw {gij},{gji})"
            rows.append({"node1": node_names[i], "node2": node_names[j], "relation": relation})
    return pd.DataFrame(rows)


_ENDPOINT_SYMBOL = {"TAIL": "-", "ARROW": ">", "CIRCLE": "o", "NULL": " "}
_LEFT_MARK = {"-": "-", ">": "<", "o": "o"}


def fci_edges_to_df(edges) -> pd.DataFrame:
    rows = []
    for e in edges:
        n1, n2 = e.get_node1().get_name(), e.get_node2().get_name()
        end1 = _ENDPOINT_SYMBOL.get(e.get_endpoint1().name, "?")
        end2 = _ENDPOINT_SYMBOL.get(e.get_endpoint2().name, "?")
        relation = f"{n1} {_LEFT_MARK.get(end1, '?')}--{end2} {n2}"
        rows.append({"node1": n1, "node2": n2, "end1": end1, "end2": end2, "relation": relation})
    return pd.DataFrame(rows)


def drug_event_edges(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Filter an edge table down to rows connecting a drug__ node to an event__ node."""
    def is_drug_event(row):
        n1, n2 = row["node1"], row["node2"]
        return (n1.startswith("drug__") and n2.startswith("event__")) or \
               (n2.startswith("drug__") and n1.startswith("event__"))
    mask = edge_df.apply(is_drug_event, axis=1)
    out = edge_df[mask].copy()

    def split(row):
        drug = row["node1"] if row["node1"].startswith("drug__") else row["node2"]
        event = row["node1"] if row["node1"].startswith("event__") else row["node2"]
        return pd.Series({"drug": drug.replace("drug__", ""), "event": event.replace("event__", "")})

    if len(out):
        out[["drug", "event"]] = out.apply(split, axis=1)
    return out
