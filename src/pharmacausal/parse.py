"""Parse a FAERS quarterly ASCII extract into clean, deduplicated parquet tables.

FAERS quirks this module handles explicitly (skipping them silently would
corrupt any downstream causal analysis):

1. **Case versioning.** A `caseid` can appear under several `primaryid`s
   across quarters (and even within one quarter) as follow-up reports amend
   the original. `caseversion` increments with each amendment. Keeping every
   version double-counts the same underlying patient/event. We keep only the
   highest `caseversion` per `caseid` and treat that as the canonical case.

2. **Deleted cases.** FDA ships a `Deleted/DELETE<label>.txt` list of
   `caseid`s retracted after publication (duplicates, data-entry errors,
   sponsor retractions). These must be dropped from every table, not just
   DEMO.

3. **Child-table filtering.** DRUG/REAC/OUTC/RPSR/THER/INDI are keyed by
   `primaryid`, not `caseid`. Once we know the canonical (latest, non-deleted)
   `primaryid` per case from DEMO, every other table must be filtered to that
   same set of `primaryid`s — otherwise superseded report versions leak back
   in via the child tables even after DEMO is deduplicated.

Everything is read as `dtype=str` on purpose: FAERS ids are sometimes
zero-padded or exceed int64-safe ranges in edge cases, and several "numeric"
looking columns (dose_amt, dur) contain free text. Casting happens downstream
in feature engineering, once we know which columns we actually need.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TABLES = ["DEMO", "DRUG", "REAC", "OUTC", "RPSR", "THER", "INDI"]


def _find_ascii_dir(raw_dir: Path, label: str) -> Path:
    candidates = list((raw_dir / label).rglob("DEMO*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No DEMO*.txt found under {raw_dir / label}; run download.py first")
    return candidates[0].parent


def _find_table_file(ascii_dir: Path, table: str) -> Path:
    matches = list(ascii_dir.glob(f"{table}*.txt"))
    if not matches:
        raise FileNotFoundError(f"No {table}*.txt in {ascii_dir}")
    return matches[0]


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="$",
        dtype=str,
        engine="c",
        na_values=[""],
        keep_default_na=True,
        on_bad_lines="warn",
        encoding="latin-1",
    )


def _find_deleted_caseids(raw_dir: Path, label: str) -> set[str]:
    matches = list((raw_dir / label).rglob("DELETE*.txt"))
    if not matches:
        return set()
    deleted = pd.read_csv(matches[0], header=None, dtype=str, names=["caseid"])
    return set(deleted["caseid"].dropna().str.strip())


def parse_quarter(year: int, quarter: int, raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> Path:
    label = f"{year}q{quarter}"
    ascii_dir = _find_ascii_dir(raw_dir, label)
    out_path = out_dir / label
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[{label}] reading DEMO to establish canonical case set...")
    demo_file = _find_table_file(ascii_dir, "DEMO")
    demo = _read_table(demo_file)
    print(f"  raw DEMO rows: {len(demo):,}")

    deleted_ids = _find_deleted_caseids(raw_dir, label)
    print(f"  deleted/retracted caseids to exclude: {len(deleted_ids):,}")
    if deleted_ids:
        demo = demo[~demo["caseid"].isin(deleted_ids)]

    demo["caseversion"] = pd.to_numeric(demo["caseversion"], errors="coerce")
    demo = demo.sort_values("caseversion").drop_duplicates(subset="caseid", keep="last")
    print(f"  canonical cases after caseversion dedup: {len(demo):,}")

    keep_primaryids = set(demo["primaryid"])

    demo.to_parquet(out_path / "demo.parquet", index=False)

    for table in ["DRUG", "REAC", "OUTC", "RPSR", "THER", "INDI"]:
        f = _find_table_file(ascii_dir, table)
        df = _read_table(f)
        before = len(df)
        df = df[df["primaryid"].isin(keep_primaryids)]
        df.to_parquet(out_path / f"{table.lower()}.parquet", index=False)
        print(f"[{label}] {table}: {before:,} -> {len(df):,} rows after primaryid filter -> {out_path / (table.lower() + '.parquet')}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--quarter", type=int, default=2, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    parse_quarter(args.year, args.quarter)


if __name__ == "__main__":
    main()
