"""
Clean, filter, and split FAA SDR data for the maintenance simulation.

Usage:
    python scripts/prepare_data.py

Outputs to data/processed/:
  - sdr_737800.parquet   — all cleaned 737-800 records
  - sdr_train.parquet    — 2019–2021 (knowledge graph seeding + health state priors)
  - sdr_test.parquet     — 2022–2023 (held-out validation set)
  - knowledge_graph.json — serialized NetworkX graph

If no SDR CSVs are found, generates synthetic data for offline use.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from maintenance_sim.config import (
    ATA_SUBSYSTEM_MAP,
    BLEED_RIGHT_KEYWORDS,
    ENGINE_RIGHT_KEYWORDS,
    PROCESSED_DIR,
    RAW_DIR,
    SEEDS_DIR,
    SUBSYSTEMS,
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUTOFF_YEAR = 2021


# ---------------------------------------------------------------------------
# Load raw SDR data
# ---------------------------------------------------------------------------

def load_raw_sdr(years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        path = RAW_DIR / f"SDR{year}.csv"
        if not path.exists():
            print(f"  [skip] SDR{year}.csv not found")
            continue
        try:
            # FAA SDRs use latin-1 encoding; try utf-8 first then fallback
            try:
                df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1", low_memory=False)
            df["source_year"] = year
            frames.append(df)
            print(f"  [ok] Loaded SDR{year}: {len(df):,} rows")
        except Exception as e:
            print(f"  [warn] Could not read SDR{year}.csv: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase with underscores."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def filter_737800(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows for Boeing 737-800."""
    make_col = next((c for c in df.columns if "make" in c and "aircraft" in c), None)
    model_col = next((c for c in df.columns if "model" in c and "aircraft" in c), None)
    if make_col is None or model_col is None:
        # Try simpler column names
        make_col = next((c for c in df.columns if c in ("aircraftmake", "make")), None)
        model_col = next((c for c in df.columns if c in ("aircraftmodel", "model")), None)

    if make_col is None or model_col is None:
        print("  [warn] Cannot find make/model columns; returning all rows")
        return df

    mask = (
        df[make_col].str.upper().str.contains("BOEING", na=False)
        & df[model_col].str.upper().str.contains("737.8", na=False, regex=True)
    )
    filtered = df[mask].copy()
    print(f"  Filtered to 737-800: {len(filtered):,} rows")
    return filtered


def extract_jasc(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 2-digit ATA chapter from JASC/ATA code column."""
    jasc_col = next(
        (c for c in df.columns if "jasc" in c or ("ata" in c and "chapter" not in c)),
        None,
    )
    if jasc_col is None:
        jasc_col = next((c for c in df.columns if "difficulty_code" in c or "partcode" in c), None)

    if jasc_col is None:
        print("  [warn] No JASC/ATA column found; cannot map to subsystems")
        df["ata_chapter"] = "00"
        return df

    df["ata_chapter"] = df[jasc_col].astype(str).str[:2].str.zfill(2)
    return df


def assign_subsystem(df: pd.DataFrame) -> pd.DataFrame:
    """Map ATA chapter → subsystem, handle L/R engine/bleed splits."""
    text_col = next(
        (c for c in df.columns if "description" in c or "narrative" in c), None
    )
    part_col = next((c for c in df.columns if "partname" in c or "part_name" in c), None)

    def _get_subsystem(row: pd.Series) -> str | None:
        chapter = str(row.get("ata_chapter", "00"))
        base = ATA_SUBSYSTEM_MAP.get(chapter)
        if base is None:
            return None

        combined = " ".join(
            str(row.get(c, "")).lower()
            for c in [text_col, part_col]
            if c is not None
        )

        if base == "engine_l":
            if any(kw in combined for kw in ENGINE_RIGHT_KEYWORDS):
                return "engine_r"
            return "engine_l"
        if base == "bleed_air_l":
            if any(kw in combined for kw in BLEED_RIGHT_KEYWORDS):
                return "bleed_air_r"
            return "bleed_air_l"
        return base

    df["subsystem"] = df.apply(_get_subsystem, axis=1)
    before = len(df)
    df = df.dropna(subset=["subsystem"])
    print(f"  Subsystem mapping: {len(df):,}/{before:,} rows matched ATA chapters")
    return df


def parse_date(df: pd.DataFrame) -> pd.DataFrame:
    date_col = next(
        (c for c in df.columns if "date" in c and "difficulty" in c),
        next((c for c in df.columns if "date" in c), None),
    )
    if date_col is None:
        df["difficulty_date"] = pd.NaT
        return df
    df["difficulty_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["difficulty_year"] = df["difficulty_date"].dt.year
    return df


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------

def generate_synthetic_sdr(n_records: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic SDR records when no real data is available."""
    print("  Generating synthetic SDR data (no real CSV files found)...")
    rng = np.random.default_rng(seed)

    subsystem_weights = {
        "engine_l": 0.18, "engine_r": 0.17,
        "bleed_air_l": 0.12, "bleed_air_r": 0.11,
        "apu": 0.10,
        "hydraulics": 0.09,
        "landing_gear": 0.10,
        "avionics": 0.08,
        "flight_controls": 0.05,
    }
    subsystems = list(subsystem_weights.keys())
    weights = list(subsystem_weights.values())

    years = rng.choice([2019, 2020, 2021, 2022, 2023], size=n_records,
                       p=[0.18, 0.18, 0.22, 0.22, 0.20])
    sampled_subsystems = rng.choice(subsystems, size=n_records, p=weights)

    # Inject cascade co-occurrences: bleed_air faults followed by engine faults
    cascade_mask = (sampled_subsystems == "bleed_air_l") & (rng.random(n_records) < 0.35)
    extra_engine_idxs = np.where(cascade_mask)[0]

    extra_rows = {
        "subsystem": ["engine_l"] * len(extra_engine_idxs),
        "difficulty_year": years[extra_engine_idxs] + (rng.integers(0, 2, len(extra_engine_idxs))),
        "difficulty_date": pd.to_datetime(
            [f"{y}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}"
             for y in years[extra_engine_idxs]]
        ),
        "source_year": years[extra_engine_idxs],
        "ata_chapter": ["72"] * len(extra_engine_idxs),
    }

    base_dates = pd.to_datetime(
        [f"{y}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}" for y in years]
    )

    df = pd.DataFrame({
        "subsystem": sampled_subsystems,
        "difficulty_year": years,
        "difficulty_date": base_dates,
        "source_year": years,
        "ata_chapter": [
            next((k for k, v in ATA_SUBSYSTEM_MAP.items() if v == s), "72")
            for s in sampled_subsystems
        ],
    })

    extra_df = pd.DataFrame(extra_rows)
    df = pd.concat([df, extra_df], ignore_index=True)
    df["difficulty_year"] = df["difficulty_year"].clip(2019, 2023).astype(int)
    print(f"  Generated {len(df):,} synthetic SDR records")
    return df


# ---------------------------------------------------------------------------
# Edge derivation from co-occurrence
# ---------------------------------------------------------------------------

def derive_sdr_edges(
    df: pd.DataFrame, window_days: int = 30, min_cooccur: int = 5, min_lift: float = 1.5
) -> list[dict]:
    """
    Compute co-occurrence lift for subsystem pairs within a time window.
    Returns list of edge dicts {source, target, weight, confidence}.
    """
    if "difficulty_date" not in df.columns or df["difficulty_date"].isna().all():
        return []

    df = df.dropna(subset=["difficulty_date", "subsystem"]).copy()
    df = df.sort_values("difficulty_date")

    # Compute marginal fault rates per subsystem
    total = len(df)
    marginal = (df["subsystem"].value_counts() / total).to_dict()

    # Find co-occurrences: for each fault on subsystem A, count B faults within window
    cooccur: dict[tuple[str, str], int] = {}
    for _, row in df.iterrows():
        window_end = row["difficulty_date"] + pd.Timedelta(days=window_days)
        window_df = df[
            (df["difficulty_date"] > row["difficulty_date"])
            & (df["difficulty_date"] <= window_end)
            & (df["subsystem"] != row["subsystem"])
        ]
        for b in window_df["subsystem"].unique():
            key = (row["subsystem"], b)
            cooccur[key] = cooccur.get(key, 0) + 1

    edges = []
    for (a, b), count in cooccur.items():
        if count < min_cooccur:
            continue
        p_b = marginal.get(b, 1e-6)
        p_ab = count / total
        lift = p_ab / p_b
        if lift >= min_lift:
            weight = min(lift / 5.0, 1.0)
            edges.append({
                "source": a, "target": b,
                "weight": round(weight, 3),
                "confidence": "sdr_derived",
                "ad_number": "",
                "ata_chapter": "",
            })

    print(f"  Derived {len(edges)} SDR co-occurrence edges (lift >= {min_lift})")
    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Preparing data...")

    # 1. Load raw SDR
    df = load_raw_sdr(list(range(2019, 2024)))
    synthetic = df.empty

    if not df.empty:
        df = normalise_columns(df)
        df = filter_737800(df)
        df = extract_jasc(df)
        df = assign_subsystem(df)
        df = parse_date(df)
    else:
        df = generate_synthetic_sdr()

    if df.empty:
        print("ERROR: No data available after processing.")
        sys.exit(1)

    # 2. Ensure required columns exist
    for col in ("subsystem", "difficulty_year", "difficulty_date"):
        if col not in df.columns:
            print(f"ERROR: Missing column: {col}")
            sys.exit(1)

    df = df.dropna(subset=["difficulty_year"])
    df["difficulty_year"] = df["difficulty_year"].astype(int)

    # 3. Save full filtered set
    sdr_path = PROCESSED_DIR / "sdr_737800.parquet"
    df.to_parquet(sdr_path, index=False)
    print(f"  Saved {sdr_path.name}: {len(df):,} rows")

    # 4. Temporal split
    train_df = df[df["difficulty_year"] <= TRAIN_CUTOFF_YEAR].copy()
    test_df = df[df["difficulty_year"] > TRAIN_CUTOFF_YEAR].copy()
    train_df.to_parquet(PROCESSED_DIR / "sdr_train.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "sdr_test.parquet", index=False)
    print(f"  Train: {len(train_df):,} rows (≤{TRAIN_CUTOFF_YEAR})")
    print(f"  Test:  {len(test_df):,} rows  (>{TRAIN_CUTOFF_YEAR})")

    # 5. Build and save knowledge graph
    print("Building knowledge graph...")
    import networkx as nx
    from networkx.readwrite import json_graph

    G = nx.DiGraph()
    for s in SUBSYSTEMS:
        G.add_node(s)

    # Load AD edges (highest confidence)
    ad_edges_path = SEEDS_DIR / "ad_edges.csv"
    import csv
    ad_edge_set: set[tuple[str, str]] = set()
    with open(ad_edges_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            src, tgt = row["source"], row["target"]
            G.add_edge(src, tgt,
                       weight=float(row["weight"]),
                       ad_number=row["ad_number"],
                       ata_chapter=row["ata_chapter"],
                       confidence="ad_mandated")
            ad_edge_set.add((src, tgt))

    print(f"  Added {len(ad_edge_set)} AD-mandated edges")

    # Add SDR-derived edges (skip if already have AD edge)
    sdr_edges = derive_sdr_edges(train_df if not synthetic else df)
    added_sdr = 0
    for e in sdr_edges:
        key = (e["source"], e["target"])
        if key not in ad_edge_set:
            G.add_edge(e["source"], e["target"],
                       weight=e["weight"],
                       confidence="sdr_derived",
                       ad_number="", ata_chapter="")
            ad_edge_set.add(key)
            added_sdr += 1

    print(f"  Added {added_sdr} SDR-derived edges")
    print(f"  Total graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    graph_path = PROCESSED_DIR / "knowledge_graph.json"
    with open(graph_path, "w") as f:
        json.dump(json_graph.node_link_data(G), f, indent=2)
    print(f"  Saved {graph_path.name}")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
