"""Download and extract a FAERS quarterly ASCII data extract.

FDA publishes non-cumulative quarterly extracts at a stable URL pattern:
    https://fis.fda.gov/content/Exports/faers_ascii_<year>q<quarter>.zip

No credentialing or API key is required. Each zip contains an `ascii/`
folder with seven $-delimited text files (DEMO, DRUG, REAC, OUTC, RPSR,
THER, INDI) plus a `Deleted*.txt` list of retracted case IDs and a
README describing the current quarter's schema.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

FAERS_URL_TEMPLATE = "https://fis.fda.gov/content/Exports/faers_ascii_{year}q{quarter}.zip"

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"


def download_file(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    tmp.rename(dest)
    return dest


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"[skip] {extract_to} already extracted")
        return extract_to

    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    return extract_to


def fetch_quarter(year: int, quarter: int, raw_dir: Path = RAW_DIR) -> Path:
    """Download + extract one quarterly extract. Returns the extraction dir."""
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"quarter must be 1-4, got {quarter}")

    url = FAERS_URL_TEMPLATE.format(year=year, quarter=quarter)
    label = f"{year}q{quarter}"
    zip_path = raw_dir / f"faers_ascii_{label}.zip"
    extract_dir = raw_dir / label

    print(f"Downloading {url}")
    download_file(url, zip_path)

    print(f"Extracting to {extract_dir}")
    extract_zip(zip_path, extract_dir)

    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--quarter", type=int, default=2, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    extract_dir = fetch_quarter(args.year, args.quarter)

    # FAERS zips nest an `ascii/` (or `ASCII/`) subfolder with varying case
    # across quarters; locate it rather than hardcoding.
    ascii_dirs = list(extract_dir.rglob("DEMO*.txt"))
    if not ascii_dirs:
        print("WARNING: no DEMO*.txt found after extraction — check zip layout", file=sys.stderr)
        sys.exit(1)

    ascii_dir = ascii_dirs[0].parent
    print(f"\nFound ASCII files in: {ascii_dir}")
    for f in sorted(ascii_dir.glob("*.txt")):
        print(f"  {f.name:20s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
