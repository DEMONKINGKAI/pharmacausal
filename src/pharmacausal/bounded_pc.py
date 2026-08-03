"""A depth-capped fork of causal-learn's PC skeleton discovery.

Why this exists: causal-learn's `pc()` runs an *unbounded* conditioning-set
search -- at each depth d it tests every pair of still-adjacent nodes against
every size-d subset of one node's other neighbors, and only stops once every
node's remaining degree is below the current depth. With 143 variables and
N in the hundreds of thousands, two things compound against convergence:

1. Combinatorics: if the skeleton hasn't sparsified enough by depth ~3-4,
   the number of size-d neighbor subsets per pair explodes.
2. Statistical power: chi-square/G-test independence tests get *more*
   powerful to detect tiny, practically-meaningless dependencies as N grows.
   At N in the hundreds of thousands, almost any two variables that share
   even a weak common cause (e.g. everything correlates a little with
   `n_drugs_bin`) will reject the independence null at conventional alpha,
   so the skeleton stays dense for longer than it would on a smaller,
   cleaner dataset. This is a real property of the data/method combination,
   not just an engineering inconvenience -- worth stating plainly rather
   than hiding behind a silent runtime workaround.

The fix applied here is the standard practical compromise in constraint-based
discovery on many-variable data: cap the maximum conditioning-set size
(`max_depth`). This trades completeness (conditional independencies that
only appear at higher-order conditioning sets are missed, so some edges
that should be removed will remain) for tractability. It is a documented
limitation of the resulting graph, discussed in the writeup rather than
buried in a config default.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import List

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import Meek, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from causallearn.utils.PCUtils.Helper import append_value


def skeleton_discovery_bounded(
    data: ndarray, alpha: float, indep_test: CIT, max_depth: int,
    stable: bool = True, show_progress: bool = True, node_names: List[str] | None = None,
    background_knowledge: BackgroundKnowledge | None = None,
) -> CausalGraph:
    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    depth_t0 = time.time()
    while cg.max_degree() - 1 > depth and depth < max_depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
                pbar.set_description(f"Depth={depth}/{max_depth}, node {x}")
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                if background_knowledge is not None and (
                    background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                    and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])
                ):
                    edge_removal.append((x, y))
                    edge_removal.append((y, x))
                    append_value(cg.sepset, x, y, ())
                    append_value(cg.sepset, y, x, ())
                    continue
                sepsets = set()
                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))
                            edge_removal.append((y, x))
                            for s in S:
                                sepsets.add(s)
                if (x, y) in edge_removal or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))
        if show_progress:
            pbar.refresh()
        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

        n_remaining = int((cg.G.graph != 0).sum() / 2)
        print(f"  [depth {depth} done in {time.time() - depth_t0:.1f}s] "
              f"{n_remaining} edges remain, max_degree={cg.max_degree()}", flush=True)
        depth_t0 = time.time()

    if show_progress:
        pbar.close()
    return cg


def pc_bounded(
    data: ndarray, alpha: float, indep_test_name: str, max_depth: int,
    stable: bool = True, uc_rule: int = 0, uc_priority: int = 2,
    show_progress: bool = True, node_names: List[str] | None = None,
    background_knowledge: BackgroundKnowledge | None = None,
) -> CausalGraph:
    start = time.time()
    indep_test = CIT(data, indep_test_name)
    cg_1 = skeleton_discovery_bounded(
        data, alpha, indep_test, max_depth, stable=stable,
        show_progress=show_progress, node_names=node_names,
        background_knowledge=background_knowledge,
    )
    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)
    cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
    cg = Meek.meek(cg_2, background_knowledge=background_knowledge)
    cg.PC_elapsed = time.time() - start
    return cg
