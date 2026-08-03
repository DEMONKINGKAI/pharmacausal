"""Inventory what confounder-relevant variables FAERS actually provides,
and how usable they are (missingness, coding quality, coverage).

This is deliberately not a generic pandas-profiling dump. It's scoped to the
question the causal discovery step depends on: which fields could plausibly
sit in an adjustment set for drug -> adverse-event edges, and how much of
each field is actually populated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def explore_quarter(year: int, quarter: int, out_dir: Path = PROCESSED_DIR) -> None:
    label = f"{year}q{quarter}"
    path = out_dir / label

    demo = pd.read_parquet(path / "demo.parquet")
    drug = pd.read_parquet(path / "drug.parquet")
    reac = pd.read_parquet(path / "reac.parquet")
    outc = pd.read_parquet(path / "outc.parquet")
    rpsr = pd.read_parquet(path / "rpsr.parquet")
    ther = pd.read_parquet(path / "ther.parquet")
    indi = pd.read_parquet(path / "indi.parquet")

    n_cases = len(demo)

    section(f"FAERS {label}: table sizes")
    for name, df in [("demo", demo), ("drug", drug), ("reac", reac),
                      ("outc", outc), ("rpsr", rpsr), ("ther", ther), ("indi", indi)]:
        print(f"  {name:6s} {len(df):>10,} rows")
    print(f"  canonical cases (unique caseid in demo): {n_cases:,}")
    print(f"  unique drug names (raw, uncoded): {drug['drugname'].nunique():,}")
    print(f"  unique reaction PTs: {reac['pt'].nunique():,}")

    # ---- Demographic / patient-level confounder candidates ----
    section("DEMO: candidate confounder fields, missingness")
    fields = ["age", "age_cod", "age_grp", "sex", "wt", "wt_cod",
              "occp_cod", "reporter_country", "occr_country", "rept_cod",
              "event_dt", "rept_dt", "init_fda_dt"]
    for f in fields:
        n_missing = demo[f].isna().sum()
        print(f"  {f:18s} missing {n_missing:>8,} / {n_cases:,}  ({pct(n_missing, n_cases)} missing)")

    section("DEMO: age_cod distribution (unit of `age`  -  needed to normalize to years)")
    print(demo["age_cod"].value_counts(dropna=False).to_string())

    section("DEMO: sex distribution")
    print(demo["sex"].value_counts(dropna=False).to_string())

    section("DEMO: occp_cod distribution (reporter occupation  -  reporting-quality/stimulus proxy)")
    print(demo["occp_cod"].value_counts(dropna=False).to_string())
    print("  MD=physician PH=pharmacist OT=other health professional CN=consumer LW=lawyer RN=nurse")

    section("DEMO: rept_cod distribution (report type  -  reporting-stimulus proxy)")
    print(demo["rept_cod"].value_counts(dropna=False).to_string())
    print("  EXP=expedited (serious, mfr-reported) PER=periodic DIR=direct-to-FDA 5DAY=15-day alert")

    section("DEMO: reporter_country top 15 (secular/regional confounder)")
    print(demo["reporter_country"].value_counts(dropna=False).head(15).to_string())

    # ---- RPSR: report source, very sparse by design (only certain rept_cod get one) ----
    section("RPSR: report-source coverage")
    n_with_rpsr = demo["primaryid"].isin(rpsr["primaryid"]).sum()
    print(f"  cases with an RPSR record: {n_with_rpsr:,} / {n_cases:,} ({pct(n_with_rpsr, n_cases)})")
    print(rpsr["rpsr_cod"].value_counts(dropna=False).to_string())

    # ---- DRUG: role_cod breakdown -> polypharmacy / concomitant-drug confounder proxy ----
    section("DRUG: role_cod distribution (PS=primary suspect SS=secondary suspect C=concomitant I=interacting)")
    print(drug["role_cod"].value_counts(dropna=False).to_string())

    drugs_per_case = drug.groupby("primaryid").size()
    concom_per_case = drug[drug["role_cod"] == "C"].groupby("primaryid").size()
    section("DRUG: drugs-per-case distribution (polypharmacy proxy)")
    print(drugs_per_case.describe().to_string())
    print(f"\n  cases with >=1 concomitant (non-suspect) drug: {len(concom_per_case):,} / {n_cases:,} ({pct(len(concom_per_case), n_cases)})")

    # ---- INDI: indication -> best available comorbidity/confounding-by-indication proxy ----
    section("INDI: indication coverage (proxy for underlying disease / confounding by indication)")
    n_with_indi = demo["primaryid"].isin(indi["primaryid"]).sum()
    print(f"  cases with >=1 INDI record: {n_with_indi:,} / {n_cases:,} ({pct(n_with_indi, n_cases)})")
    unknown_mask = indi["indi_pt"].str.contains("unknown indication", case=False, na=False)
    print(f"  INDI rows coded as 'Product used for unknown indication': {unknown_mask.sum():,} / {len(indi):,} ({pct(unknown_mask.sum(), len(indi))})")
    cases_only_unknown = (
        indi.groupby("primaryid")["indi_pt"]
        .apply(lambda s: s.str.contains("unknown indication", case=False, na=False).all())
    )
    print(f"  cases where EVERY indication is 'unknown': {cases_only_unknown.sum():,} / {len(cases_only_unknown):,} ({pct(cases_only_unknown.sum(), len(cases_only_unknown))})")
    section("INDI: top 20 coded indications")
    print(indi["indi_pt"].value_counts().head(20).to_string())

    # ---- THER: therapy dates -> exposure timing / duration ----
    section("THER: therapy start/end date coverage (needed for exposure-window / temporal-precedence reasoning)")
    n_with_ther = demo["primaryid"].isin(ther["primaryid"]).sum()
    print(f"  cases with >=1 THER record: {n_with_ther:,} / {n_cases:,} ({pct(n_with_ther, n_cases)})")
    for f in ["start_dt", "end_dt", "dur"]:
        miss = ther[f].isna().sum()
        print(f"  THER.{f:10s} missing {miss:,} / {len(ther):,} ({pct(miss, len(ther))})")

    # ---- OUTC: outcome severity ----
    section("OUTC: outcome code distribution (severity  -  potential collider via reporting stimulus)")
    print(outc["outc_cod"].value_counts(dropna=False).to_string())
    print("  DE=death LT=life-threatening HO=hospitalization DS=disability CA=congenital-anomaly RI=required-intervention OT=other")

    section("SUMMARY: what this means for causal discovery (see writeup for full discussion)")
    print("""
  Usable as covariates (moderate-to-good coverage): age, sex, reporter
  occupation, report type, reporter country, drug count / concomitant-drug
  count, coarse indication category.

  Structurally important but NOT a covariate you condition on casually:
  indication is entangled with the very suspect-drug edges we're trying to
  discover (it's often a cause of the prescription itself), and a large
  fraction of indication rows are uninformative ("unknown indication").

  Missing entirely: any unexposed/untreated comparison population, exact
  incidence denominators, and a case-selection model. FAERS is a reported-case
  registry, not a cohort -- selection into the dataset is itself a function of
  both drug and outcome severity (stimulated reporting, Weber effect,
  litigation waves), which is a form of collider bias no covariate in this
  file can adjust away. This bounds what "causal discovery on FAERS" can mean:
  structure among reported-case variables, not population-level causal effects.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--quarter", type=int, default=2, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    explore_quarter(args.year, args.quarter)


if __name__ == "__main__":
    main()
