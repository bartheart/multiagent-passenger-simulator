"""
Download FAA Service Difficulty Report (SDR) data for years 2019–2023.

Usage:
    python scripts/download_data.py

Downloads to data/raw/. Safe to re-run — skips existing files.
"""

import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# FAA SDR bulk download URLs (year-by-year ZIP files)
SDR_URLS = {
    2019: "https://av-info.faa.gov/sdrx/data/SDR2019.zip",
    2020: "https://av-info.faa.gov/sdrx/data/SDR2020.zip",
    2021: "https://av-info.faa.gov/sdrx/data/SDR2021.zip",
    2022: "https://av-info.faa.gov/sdrx/data/SDR2022.zip",
    2023: "https://av-info.faa.gov/sdrx/data/SDR2023.zip",
}


def download_sdr(year: int, url: str) -> Path:
    dest_csv = RAW_DIR / f"SDR{year}.csv"
    if dest_csv.exists():
        print(f"  [skip] SDR{year}.csv already exists")
        return dest_csv

    print(f"  Downloading SDR{year}...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [warn] Could not download SDR{year}: {e}")
        return dest_csv  # may not exist; prepare_data.py will handle missing files

    zip_path = RAW_DIR / f"SDR{year}.zip"
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        # FAA ZIPs typically contain a single CSV; find it
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            print(f"  [warn] No CSV found in SDR{year}.zip")
            zip_path.unlink()
            return dest_csv
        zf.extract(csv_names[0], RAW_DIR)
        extracted = RAW_DIR / csv_names[0]
        extracted.rename(dest_csv)

    zip_path.unlink()
    print(f"  [ok] Saved {dest_csv.name} ({dest_csv.stat().st_size // 1024} KB)")
    return dest_csv


def main() -> None:
    print("Downloading FAA SDR data (2019–2023)...")
    success = 0
    for year, url in SDR_URLS.items():
        path = download_sdr(year, url)
        if path.exists():
            success += 1

    print(f"\nDone: {success}/{len(SDR_URLS)} files available in data/raw/")
    if success == 0:
        print("No SDR files downloaded. prepare_data.py will generate synthetic data.")
        sys.exit(0)


if __name__ == "__main__":
    main()
