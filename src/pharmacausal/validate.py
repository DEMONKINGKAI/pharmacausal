"""Cross-reference discovered drug->event edges against SIDER as a rough
validation proxy, and report precision honestly -- including for the drugs
SIDER simply has no data on.

Why SIDER, not DrugBank: our discovered edges are drug -> adverse-event
associations. DrugBank's structured interaction data is drug-DRUG
interactions -- a different relationship type. SIDER is built by mining
side-effect terms out of drug package-insert labels, so it's a direct match
for "is this a known/labeled side effect of this drug," which is the closest
thing to ground truth this kind of validation can use.

What "precision" means here, and what it doesn't
--------------------------------------------------
This is NOT precision in a statistical-decision-theory sense. SIDER records
*labeled* side effects -- things known/established at the time the label was
written. A discovered edge that ISN'T in SIDER could be:
  (a) actually spurious -- the confounded/noise result we're worried about, or
  (b) a real signal SIDER doesn't have on file (a newer or rarer reaction,
      or a reaction only visible in post-market surveillance data like FAERS,
      which is the whole reason post-market pharmacovigilance exists), or
  (c) unmatchable, not because it's wrong, but because SIDER has no entry for
      that drug at all.

SIDER 4.1 was last updated in 2015 (see README output for citation). Several
of our candidate drugs (dupilumab 2017, semaglutide 2017, tirzepatide 2022,
bimekizumab 2023, abaloparatide 2017, evolocumab 2015) postdate or barely
predate that cutoff, and SIDER's biologics/monoclonal-antibody coverage is
weaker than its small-molecule coverage even for older drugs (rituximab,
approved 1997, has no SIDER entry). We report coverage (what fraction of
candidate drugs SIDER has *any* data for) separately from precision (of the
covered subset, what fraction match), rather than silently scoring "not in
SIDER" as "wrong."

A small alias table below fixes known USAN/INN naming mismatches (SIDER is
EMBL/European in origin and prefers INN names, e.g. "salbutamol" for the
US "albuterol") -- these are naming differences, not coverage gaps, and
conflating them with real gaps would understate SIDER coverage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "sider"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# USAN/USP (US) -> INN (SIDER/EMBL) naming differences observed in our candidate list.
DRUG_ALIASES = {
    "ALBUTEROL": "SALBUTAMOL",
}


def load_sider_lookup() -> dict[str, set[str]]:
    """Return {UPPERCASE drug name: {MedDRA PT side effects}} from SIDER 4.1."""
    names = pd.read_csv(RAW_DIR / "drug_names.tsv", sep="\t", header=None, names=["cid", "name"])
    names["name"] = names["name"].str.upper()
    cid_to_names = names.groupby("cid")["name"].apply(set).to_dict()

    se = pd.read_csv(RAW_DIR / "meddra_all_se.tsv", sep="\t", header=None,
                      names=["cid_flat", "cid_stereo", "umls_label", "meddra_type",
                             "umls_meddra", "side_effect"])
    se_pt = se[se["meddra_type"] == "PT"]

    lookup: dict[str, set[str]] = {}
    for cid, group in se_pt.groupby("cid_flat"):
        drug_names = cid_to_names.get(cid, set())
        pts = set(group["side_effect"].str.upper())
        for dn in drug_names:
            lookup.setdefault(dn, set()).update(pts)
    return lookup


def resolve_drug_name(name: str, sider_lookup: dict[str, set[str]]) -> str | None:
    if name in sider_lookup:
        return name
    alias = DRUG_ALIASES.get(name)
    if alias and alias in sider_lookup:
        return alias
    return None


def validate_edges(edge_df: pd.DataFrame, sider_lookup: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for _, row in edge_df.iterrows():
        drug, event = row["drug"], row["event"].upper()
        resolved = resolve_drug_name(drug, sider_lookup)
        if resolved is None:
            status = "NO_SIDER_DATA"
        elif event in sider_lookup[resolved]:
            status = "MATCH"
        else:
            status = "NOT_LABELED"
        rows.append({"drug": drug, "event": row["event"], "status": status})
    return pd.DataFrame(rows)


def summarize(result: pd.DataFrame, label: str) -> None:
    n = len(result)
    n_covered = (result["status"] != "NO_SIDER_DATA").sum()
    n_match = (result["status"] == "MATCH").sum()
    n_not_labeled = (result["status"] == "NOT_LABELED").sum()
    print(f"\n=== {label} ===")
    print(f"  total candidate edges: {n}")
    print(f"  SIDER has drug data for: {n_covered} ({100 * n_covered / n:.0f}% coverage)")
    if n_covered:
        print(f"  of those, matches a labeled side effect: {n_match} "
              f"({100 * n_match / n_covered:.0f}% precision on covered subset)")
        print(f"  of those, NOT a labeled side effect: {n_not_labeled} "
              f"({100 * n_not_labeled / n_covered:.0f}%)")
    print(f"  no SIDER data for the drug at all: {n - n_covered}")


def main() -> None:
    label = "2026q2"
    path = PROCESSED_DIR / label
    sider_lookup = load_sider_lookup()
    print(f"SIDER: {len(sider_lookup)} drug names, "
          f"{sum(len(v) for v in sider_lookup.values()):,} drug-PT pairs total")

    for source in ["pc", "fci"]:
        edges = pd.read_csv(path / f"{source}_drug_event_edges.csv")
        result = validate_edges(edges, sider_lookup)
        result.to_csv(path / f"{source}_sider_validation.csv", index=False)
        summarize(result, f"{source.upper()} drug-event edges vs SIDER")
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
