"""Prepare the discrete discovery matrix and run PC + FCI over it.

Missing-data strategy
----------------------
`age_years` (37% missing) and `sex_female` (19% missing) are the only
columns with missingness -- every drug/event/indication flag is 0/1 by
construction (crosstab fill), and the remaining confounders have no nulls.
We use listwise deletion on the two columns that do (55% of cases, 232k
rows, survive together), rather than adding missingness-indicator columns.

The reason: a missingness indicator for `age`/`sex` would encode *reporting
behavior* (which report types bother to fill in demographics) as if it were
a clinical variable, and it would very plausibly become a confounding hub in
the discovered graph -- entangled with `report_expedited` and
`reporter_us` -- for reasons that have nothing to do with drugs or
adverse events. Dropping incomplete rows is more conservative and easier to
interpret, at the cost of some power and of implicitly restricting the
discovery population to "cases where age and sex were recorded," which is
itself a selection step worth naming in the writeup, not hiding.

Discretization
--------------
causal-learn's discrete tests (chisq / gsq) need integer-coded categorical
data. `age_years` and the two drug-count confounders are continuous/count
with long tails (`n_drugs` max is 1024), so we bin them into a handful of
ordinal categories. Every other column is already binary.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Tiered background knowledge: an edge is forbidden from a higher tier into a
# lower one. This encodes temporal/structural precedence that PC/FCI cannot
# infer from conditional independence alone -- e.g. without this, nothing
# stops the algorithm from orienting a reaction as a cause of the drug that
# produced it (observed on an early smoke test: "Headache --> ABALOPARATIDE").
#   0: demographics / report metadata -- exogenous, cause everything
#   1: indication + polypharmacy counts -- can follow demographics, precede
#      the prescribing decision
#   2: drug exposure flags
#   3: adverse event flags -- terminal; nothing can be *caused by* an event
#      in this graph, since every event here is, by construction, something
#      reported alongside (i.e. temporally at or after) the drug exposures
#      on the same case
_TIER_0 = {"age_bin", "sex_female", "reporter_us", "report_expedited"}
_TIER_1_EXTRA = {"n_drugs_bin", "n_suspect_drugs_bin"}


def node_tier(name: str) -> int:
    if name in _TIER_0:
        return 0
    if name in _TIER_1_EXTRA or name.startswith("indication__"):
        return 1
    if name.startswith("drug__"):
        return 2
    if name.startswith("event__"):
        return 3
    raise ValueError(f"unrecognized column, no tier assigned: {name}")


def build_background_knowledge(node_names: list[str]):
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    bk = BackgroundKnowledge()
    for name in node_names:
        bk.add_node_to_tier(GraphNode(name), node_tier(name))
    return bk

AGE_BINS = [-0.1, 18, 45, 65, 75, 200]
AGE_LABELS = ["0-17", "18-44", "45-64", "65-74", "75+"]

DRUGCOUNT_BINS = [0, 1, 2, 5, 10, 100000]
DRUGCOUNT_LABELS = ["1", "2", "3-5", "6-10", "11+"]


def build_discovery_matrix(year: int, quarter: int, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    label = f"{year}q{quarter}"
    path = out_dir / label
    fm = pd.read_parquet(path / "feature_matrix.parquet")

    n_before = len(fm)
    fm = fm.dropna(subset=["age_years", "sex_female"]).copy()
    print(f"[{label}] listwise deletion on age/sex: {n_before:,} -> {len(fm):,} rows "
          f"({100 * len(fm) / n_before:.1f}% retained)")

    fm["age_bin"] = pd.cut(fm["age_years"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    fm["n_drugs_bin"] = pd.cut(fm["n_drugs"], bins=DRUGCOUNT_BINS, labels=DRUGCOUNT_LABELS).astype(str)
    fm["n_suspect_drugs_bin"] = pd.cut(fm["n_suspect_drugs"], bins=DRUGCOUNT_BINS, labels=DRUGCOUNT_LABELS).astype(str)
    fm["sex_female"] = fm["sex_female"].astype(int)

    drop_cols = ["primaryid", "caseid", "age_years", "n_drugs", "n_suspect_drugs"]
    dm = fm.drop(columns=drop_cols)

    # integer-code the three ordinal/categorical bins; everything else is
    # already 0/1 and stays as-is.
    for col in ["age_bin", "n_drugs_bin", "n_suspect_drugs_bin"]:
        cats = AGE_LABELS if col == "age_bin" else DRUGCOUNT_LABELS
        dm[col] = pd.Categorical(dm[col], categories=cats, ordered=True).codes

    out_file = path / "discovery_matrix.parquet"
    dm.to_parquet(out_file, index=False)
    print(f"[{label}] discovery matrix: {dm.shape[0]:,} rows x {dm.shape[1]} columns -> {out_file}")
    return dm


def run_pc(
    year: int, quarter: int, alpha: float = 0.001, test: str = "chisq",
    max_depth: int = 3, subsample: int | None = None, out_dir: Path = PROCESSED_DIR,
) -> Path:
    """Run PC at full variable scale.

    alpha defaults to 0.001, stricter than causal-learn's usual 0.05 default,
    because N is in the hundreds of thousands: at conventional alpha, chi-sq/
    G-test CI tests have enough power to reject the independence null for
    dependencies too small to be practically meaningful, which both slows
    convergence (denser skeleton survives longer) and would clutter the graph
    with edges reflecting statistical rather than substantive significance.

    max_depth caps the conditioning-set size PC searches -- see bounded_pc.py
    for why this is necessary at 141 variables.
    """
    label = f"{year}q{quarter}"
    path = out_dir / label
    dm = pd.read_parquet(path / "discovery_matrix.parquet")

    if subsample:
        dm = dm.sample(n=min(subsample, len(dm)), random_state=0)
        print(f"[{label}] subsampled to {len(dm):,} rows for this run")

    node_names = list(dm.columns)
    data = dm.to_numpy(dtype=np.int64)
    print(f"[{label}] PC on {data.shape[0]:,} rows x {data.shape[1]} vars, "
          f"test={test}, alpha={alpha}, max_depth={max_depth}, with tiered background knowledge")

    from bounded_pc import pc_bounded

    bk = build_background_knowledge(node_names)
    t0 = time.time()
    pc_result = pc_bounded(data, alpha=alpha, indep_test_name=test, max_depth=max_depth,
                            stable=True, node_names=node_names, show_progress=True,
                            background_knowledge=bk)
    n_edges = int((pc_result.G.graph != 0).sum() / 2)
    print(f"[{label}] PC done in {time.time() - t0:.1f}s, {n_edges} edges")

    out_file = path / f"pc_result_{test}_a{alpha}_d{max_depth}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump({"graph": pc_result, "node_names": node_names}, f)
    return out_file


def run_fci_on_pc_subset(
    year: int, quarter: int, pc_result_file: Path, alpha: float = 0.001, test: str = "chisq",
    max_depth: int = 3, max_path_length: int = 3, subsample: int | None = None,
    out_dir: Path = PROCESSED_DIR,
) -> Path:
    """Run FCI on a reduced node set: every drug/event node PC connected by a
    drug-event edge, plus all confounders.

    FCI relaxes causal sufficiency (allows latent common causes, surfaced as
    `o->`/`<->` marks) which is the more honest assumption for FAERS -- but
    profiling showed its runtime scales badly with *both* node count (15
    vars: ~1s, 30: ~5s, 50: ~86s at N=3,000) and, even more sharply, with row
    count (a 48-var PC-linked subset ran in 4.5s at N=3,000 but did not
    finish in 30 minutes at N=10,000 -- more rows change the skeleton/PDS-set
    structure FCI has to reason about, not just the per-test cost). A full
    141-variable, N=10,000+ run is impractical here.

    Two separate scoping decisions follow from that: (1) restrict FCI to
    PC's own candidate edges plus confounders, so the comparison stays
    meaningful -- for each edge PC drew under the (likely false)
    causal-sufficiency assumption, does it survive when that assumption is
    relaxed, or does FCI mark it as possibly confounded; and (2) let FCI run
    on a *smaller row sample* than PC did, since it's being used here as a
    confirmatory check on already-identified candidates rather than as the
    primary discovery pass -- it doesn't need PC's full N to do that job.
    """
    from graph_export import drug_event_edges, pc_graph_to_edges

    label = f"{year}q{quarter}"
    path = out_dir / label

    with open(pc_result_file, "rb") as f:
        pc_saved = pickle.load(f)
    pc_node_names = pc_saved["node_names"]
    pc_edges = pc_graph_to_edges(pc_node_names, pc_saved["graph"].G.graph)
    de = drug_event_edges(pc_edges)
    print(f"[{label}] PC found {len(de)} drug-event edges; building FCI subset around them")

    involved = set()
    for n in de["node1"]:
        involved.add(n)
    for n in de["node2"]:
        involved.add(n)
    confounders = [n for n in pc_node_names if node_tier(n) <= 1]
    subset_nodes = confounders + sorted(involved)
    print(f"[{label}] FCI subset: {len(confounders)} confounders + {len(involved)} drug/event nodes "
          f"= {len(subset_nodes)} total")

    dm = pd.read_parquet(path / "discovery_matrix.parquet")
    if subsample:
        dm = dm.sample(n=min(subsample, len(dm)), random_state=0)
    dm = dm[subset_nodes]
    data = dm.to_numpy(dtype=np.int64)

    bk = build_background_knowledge(subset_nodes)
    from causallearn.search.ConstraintBased.FCI import fci

    t0 = time.time()
    fci_graph, fci_edges = fci(
        data, independence_test_method=test, alpha=alpha, depth=max_depth,
        max_path_length=max_path_length, node_names=subset_nodes,
        background_knowledge=bk, show_progress=True,
    )
    print(f"[{label}] FCI done in {time.time() - t0:.1f}s, {len(fci_edges)} edges "
          f"on {data.shape[0]:,} rows x {len(subset_nodes)} vars")

    out_file = path / f"fci_result_{test}_a{alpha}_d{max_depth}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump({"graph": fci_graph, "edges": fci_edges, "node_names": subset_nodes}, f)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--quarter", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--test", type=str, default="chisq", choices=["chisq", "gsq"])
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-path-length", type=int, default=3)
    parser.add_argument("--subsample", type=int, default=None, help="row sample size for PC")
    parser.add_argument("--fci-subsample", type=int, default=None,
                         help="row sample size for the FCI confirmatory pass (independent of --subsample)")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-pc", action="store_true", help="reuse existing PC result, only run FCI")
    args = parser.parse_args()

    if not args.skip_build:
        build_discovery_matrix(args.year, args.quarter)

    label = f"{args.year}q{args.quarter}"
    pc_file = PROCESSED_DIR / label / f"pc_result_{args.test}_a{args.alpha}_d{args.max_depth}.pkl"
    if not args.skip_pc:
        pc_file = run_pc(args.year, args.quarter, alpha=args.alpha, test=args.test,
                          max_depth=args.max_depth, subsample=args.subsample)
    run_fci_on_pc_subset(args.year, args.quarter, pc_result_file=pc_file, alpha=args.alpha,
                          test=args.test, max_depth=args.max_depth,
                          max_path_length=args.max_path_length, subsample=args.fci_subsample)


if __name__ == "__main__":
    main()
