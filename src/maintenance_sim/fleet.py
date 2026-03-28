"""
Aircraft dataclass and synthetic fleet factory.

Health states are seeded from SDR fault rates where available,
or from Beta distribution priors for offline use.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from maintenance_sim.config import (
    BASE_DECAY_RATES,
    DEMO_SEEDS,
    PROCESSED_DIR,
    SEEDS_DIR,
    SUBSYSTEMS,
    SYNTHETIC_TAILS,
)


@dataclass
class Aircraft:
    tail_number: str
    total_flight_cycles: int
    manufacture_year: int
    last_c_check_cycle: int
    open_deferred_items: list[str]
    health_states: dict[str, float]   # subsystem → 0.0 (failed) to 1.0 (healthy)
    seed: int = 42

    @property
    def cycles_since_c_check(self) -> int:
        return self.total_flight_cycles - self.last_c_check_cycle

    @property
    def age_years(self) -> int:
        return 2024 - self.manufacture_year


def _compute_fault_rates(sdr_df: pd.DataFrame) -> dict[str, float]:
    """Compute per-subsystem fault rate from SDR training data."""
    if sdr_df.empty or "subsystem" not in sdr_df.columns:
        return {s: BASE_DECAY_RATES[s] * 10 for s in SUBSYSTEMS}

    total = max(len(sdr_df), 1)
    counts = sdr_df["subsystem"].value_counts()
    return {s: counts.get(s, 0) / total for s in SUBSYSTEMS}


def _health_from_fault_rate(fault_rate: float, cycles: int,
                             c_check_cycles: int, rng: np.random.Generator) -> float:
    """
    Sample initial health from a Beta distribution parameterised by fault rate.
    Higher fault rate → lower expected health.
    Age (cycles since C-check) adds additional wear.
    """
    age_penalty = min(c_check_cycles / 5000, 0.3)  # max 0.3 penalty
    mean_health = max(0.1, 0.90 - fault_rate * 8.0 - age_penalty)
    # Beta params: mean = alpha/(alpha+beta), variance ~ 0.01
    variance = 0.01
    alpha = mean_health * ((mean_health * (1 - mean_health) / variance) - 1)
    beta = (1 - mean_health) * ((mean_health * (1 - mean_health) / variance) - 1)
    alpha = max(alpha, 0.5)
    beta = max(beta, 0.5)
    return float(np.clip(rng.beta(alpha, beta), 0.05, 0.99))


def _load_fleet_metadata() -> dict[str, dict]:
    """Load synthetic_fleet.csv metadata keyed by tail number."""
    path = SEEDS_DIR / "synthetic_fleet.csv"
    metadata = {}
    if not path.exists():
        return metadata
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tail = row["tail_number"]
            metadata[tail] = {
                "total_flight_cycles": int(row["total_flight_cycles"]),
                "manufacture_year": int(row["manufacture_year"]),
                "last_c_check_cycle": int(row["last_c_check_cycle"]),
                "open_deferred_items": [
                    x.strip() for x in row["open_deferred_ata"].split(",")
                    if x.strip()
                ],
            }
    return metadata


def create_fleet(
    sdr_df: pd.DataFrame | None = None,
    seed: int = 42,
) -> list[Aircraft]:
    """
    Build 20 Aircraft instances with health states seeded from SDR fault rates.

    Five aircraft are pre-seeded with specific cascade scenarios for the demo:
      N823AN: degraded bleed_air_l → engine_l cascade
      N917AN: degraded hydraulics → flight_controls cascade
      N341AN: degraded apu + bleed_air_r
      N788AN: near-limit landing_gear
      N456UA: healthy exemplar
    """
    if sdr_df is None:
        train_path = PROCESSED_DIR / "sdr_train.parquet"
        sdr_df = pd.read_parquet(train_path) if train_path.exists() else pd.DataFrame()

    fault_rates = _compute_fault_rates(sdr_df)
    metadata = _load_fleet_metadata()
    fleet = []

    for i, tail in enumerate(SYNTHETIC_TAILS):
        rng = np.random.default_rng(seed + i)
        meta = metadata.get(tail, {
            "total_flight_cycles": 15000 + rng.integers(0, 10000),
            "manufacture_year": 2000 + rng.integers(0, 15),
            "last_c_check_cycle": 14000 + rng.integers(0, 9000),
            "open_deferred_items": [],
        })
        c_check_cycles = meta["total_flight_cycles"] - meta["last_c_check_cycle"]

        health_states = {
            s: _health_from_fault_rate(fault_rates[s], meta["total_flight_cycles"],
                                        c_check_cycles, rng)
            for s in SUBSYSTEMS
        }

        # Apply demo scenario overrides
        if tail in DEMO_SEEDS:
            for subsystem, override_health in DEMO_SEEDS[tail].items():
                health_states[subsystem] = override_health

        fleet.append(Aircraft(
            tail_number=tail,
            total_flight_cycles=meta["total_flight_cycles"],
            manufacture_year=meta["manufacture_year"],
            last_c_check_cycle=meta["last_c_check_cycle"],
            open_deferred_items=meta["open_deferred_items"],
            health_states=health_states,
            seed=seed + i,
        ))

    return fleet
