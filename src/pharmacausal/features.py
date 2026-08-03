"""Build a case-level feature matrix for causal discovery from parsed FAERS tables.

Design choices, made explicit because they materially shape what the
discovered graph can and can't mean:

* **Drug identity = active ingredient (`prod_ai`), not brand name.** Raw
  `drugname` is free text (brand names, misspellings, dose strings baked in)
  and would fragment the same exposure across dozens of near-duplicate
  columns. `prod_ai` is FDA's own normalization and is what a DrugBank
  cross-reference in step 4 needs.

* **Combination products are exploded.** FAERS encodes combo ingredients as a
  single `prod_ai` string joined with `\\` (e.g. "CARBIDOPA\\LEVODOPA"). Left
  unexploded, that string becomes a spurious ~6,400th "ingredient" that never
  matches anything else. We split on `\\` and count the case as exposed to
  each component ingredient.

* **Salt/ester suffixes are stripped for grouping** (e.g. "AMLODIPINE
  BESYLATE" and "AMLODIPINE" both become "AMLODIPINE"). FAERS reporters are
  inconsistent about whether they record the salt form, so treating them as
  different drugs would undercount a single real exposure. This is a
  heuristic suffix list, not a chemistry-aware normalizer — documented as a
  known simplification.

* **Exposure is role-agnostic (PS+SS+C+I all count).** `role_cod` is the
  *reporter's* suspicion about which drug caused the event, not ground
  truth. If we only counted "primary suspect" drugs as exposures, we'd be
  baking the reporter's own causal judgment into the input data before the
  discovery algorithm ever runs -- exactly the kind of naive-correlation
  shortcut this project is trying to avoid. Concomitant drugs stay in as
  candidate causes/confounders; `role_cod` composition per case is instead
  summarized as a confounder (`n_drugs`, `n_suspect_drugs`).

* **Indication is included only as coarse binary flags for the ~10 most
  common indications**, explicitly excluding "product used for unknown
  indication". Indication is a genuine confounder candidate (proxy for
  underlying disease) but is *also* often a direct cause of the prescription
  itself -- it can sit upstream of a drug node in the true graph, which is
  fine for PC/FCI (it's still a legitimate node to condition on) but means
  it should not be silently treated as a pure "adjust and forget" covariate
  in the writeup.

* **Weight and reporter occupation are dropped as confounders** — coverage
  measured in step 1 was 17% and 28% respectively, too sparse for a binary
  presence flag to carry real signal, and imputing them would manufacture
  information FAERS doesn't have.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

N_TOP_DRUGS = 50
N_TOP_EVENTS = 75
N_TOP_INDICATIONS = 10

# Common salt/ester/hydrate suffixes FAERS reporters inconsistently append.
# Stripped so e.g. "AMLODIPINE BESYLATE" groups with "AMLODIPINE".
_SALT_SUFFIXES = [
    "HYDROCHLORIDE", "HCL", "SULFATE", "SULPHATE", "SODIUM", "POTASSIUM",
    "CALCIUM", "MESYLATE", "TARTRATE", "MALEATE", "MALATE", "FUMARATE",
    "PHOSPHATE", "CITRATE", "ACETATE", "BESYLATE", "SUCCINATE", "BROMIDE",
    "CHLORIDE", "NITRATE", "GLUCONATE", "DIHYDRATE", "MONOHYDRATE",
    "ANHYDROUS", "XINAFOATE", "FUROATE", "PROPIONATE", "VALERATE",
    "PALMITATE", "DECANOATE", "ENANTHATE", "ACEPONATE",
]
_SALT_RE = re.compile(r"\b(" + "|".join(_SALT_SUFFIXES) + r")\b")


def normalize_ingredient(raw: str) -> list[str]:
    """Split a possibly-combo prod_ai string into normalized component ingredients."""
    parts = raw.split("\\")
    out = []
    for p in parts:
        p = p.strip().upper()
        p = _SALT_RE.sub("", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def age_to_years(age: pd.Series, age_cod: pd.Series) -> pd.Series:
    age_num = pd.to_numeric(age, errors="coerce")
    factor = age_cod.map({
        "DEC": 10.0, "YR": 1.0, "MON": 1 / 12, "WK": 1 / 52.1775,
        "DY": 1 / 365.25, "HR": 1 / (365.25 * 24),
    })
    return age_num * factor


def build_feature_matrix(year: int, quarter: int, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    label = f"{year}q{quarter}"
    path = out_dir / label

    demo = pd.read_parquet(path / "demo.parquet")
    drug = pd.read_parquet(path / "drug.parquet")
    reac = pd.read_parquet(path / "reac.parquet")
    indi = pd.read_parquet(path / "indi.parquet")

    print(f"[{label}] normalizing drug ingredients...")
    drug = drug.dropna(subset=["prod_ai"]).copy()
    exploded = drug.assign(ingredient=drug["prod_ai"].map(normalize_ingredient)).explode("ingredient")
    exploded = exploded[exploded["ingredient"].str.len() > 0]

    top_drugs = (
        exploded.drop_duplicates(subset=["primaryid", "ingredient"])["ingredient"]
        .value_counts()
        .head(N_TOP_DRUGS)
        .index.tolist()
    )
    print(f"  top {N_TOP_DRUGS} ingredients by case count selected")

    top_events = reac["pt"].value_counts().head(N_TOP_EVENTS).index.tolist()
    print(f"  top {N_TOP_EVENTS} reaction PTs by case count selected")

    unknown_indi_mask = indi["indi_pt"].str.contains("unknown indication", case=False, na=False)
    known_indi = indi[~unknown_indi_mask]
    top_indications = known_indi["indi_pt"].value_counts().head(N_TOP_INDICATIONS).index.tolist()
    print(f"  top {N_TOP_INDICATIONS} known indications selected: {top_indications}")

    # ---- base frame: one row per canonical case ----
    fm = demo[["primaryid", "caseid", "age", "age_cod", "sex",
               "reporter_country", "rept_cod"]].copy()
    fm["age_years"] = age_to_years(fm["age"], fm["age_cod"])
    fm["sex_female"] = (fm["sex"] == "F").astype("Int64")
    fm.loc[fm["sex"].isna(), "sex_female"] = pd.NA
    fm["reporter_us"] = (fm["reporter_country"] == "US").astype(int)
    fm["report_expedited"] = fm["rept_cod"].isin(["EXP", "5DAY", "30DAY"]).astype(int)
    fm = fm.drop(columns=["age", "age_cod", "sex", "reporter_country", "rept_cod"])

    # ---- polypharmacy confounders ----
    n_drugs = drug.groupby("primaryid").size().rename("n_drugs")
    n_suspect = (
        drug[drug["role_cod"].isin(["PS", "SS"])]
        .groupby("primaryid").size().rename("n_suspect_drugs")
    )
    fm = fm.merge(n_drugs, on="primaryid", how="left").merge(n_suspect, on="primaryid", how="left")
    fm[["n_drugs", "n_suspect_drugs"]] = fm[["n_drugs", "n_suspect_drugs"]].fillna(0)

    # ---- drug exposure flags ----
    exposed = exploded[exploded["ingredient"].isin(top_drugs)]
    drug_flags = (
        pd.crosstab(exposed["primaryid"], exposed["ingredient"])
        .clip(upper=1)
        .add_prefix("drug__")
    )
    fm = fm.merge(drug_flags, on="primaryid", how="left")
    drug_cols = [f"drug__{d}" for d in top_drugs]
    fm[drug_cols] = fm[drug_cols].fillna(0).astype(int)

    # ---- adverse event flags ----
    reac_sel = reac[reac["pt"].isin(top_events)]
    event_flags = (
        pd.crosstab(reac_sel["primaryid"], reac_sel["pt"])
        .clip(upper=1)
        .add_prefix("event__")
    )
    fm = fm.merge(event_flags, on="primaryid", how="left")
    event_cols = [f"event__{e}" for e in top_events]
    fm[event_cols] = fm[event_cols].fillna(0).astype(int)

    # ---- indication flags ----
    indi_sel = known_indi[known_indi["indi_pt"].isin(top_indications)]
    indi_flags = (
        pd.crosstab(indi_sel["primaryid"], indi_sel["indi_pt"])
        .clip(upper=1)
        .add_prefix("indication__")
    )
    fm = fm.merge(indi_flags, on="primaryid", how="left")
    indi_cols = [f"indication__{i}" for i in top_indications]
    fm[indi_cols] = fm[indi_cols].fillna(0).astype(int)

    out_file = path / "feature_matrix.parquet"
    fm.to_parquet(out_file, index=False)

    print(f"\n[{label}] feature matrix: {fm.shape[0]:,} rows x {fm.shape[1]} columns")
    print(f"  {len(drug_cols)} drug flags, {len(event_cols)} event flags, "
          f"{len(indi_cols)} indication flags, "
          f"{fm.shape[1] - len(drug_cols) - len(event_cols) - len(indi_cols) - 2} other confounders")
    print(f"  saved to {out_file}")

    return fm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--quarter", type=int, default=2, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    build_feature_matrix(args.year, args.quarter)


if __name__ == "__main__":
    main()
