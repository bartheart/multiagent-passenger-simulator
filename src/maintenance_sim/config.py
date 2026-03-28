"""Constants, mappings, and fleet definitions."""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
SEEDS_DIR = DATA_DIR / "seeds"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = ROOT / "maintenance.db"

# --- Subsystems ---
SUBSYSTEMS = [
    "engine_l",
    "engine_r",
    "apu",
    "bleed_air_l",
    "bleed_air_r",
    "hydraulics",
    "landing_gear",
    "avionics",
    "flight_controls",
]

# --- ATA chapter prefix → subsystem ---
ATA_SUBSYSTEM_MAP: dict[str, str] = {
    "21": "bleed_air_l",   # Air Conditioning (bleed-fed)
    "36": "bleed_air_l",   # Pneumatics
    "49": "apu",           # Auxiliary Power Unit
    "71": "engine_l",      # Power Plant
    "72": "engine_l",      # Engine
    "73": "engine_l",      # Engine Fuel & Control
    "74": "engine_l",      # Ignition
    "75": "bleed_air_l",   # Air / Bleed (engine bleed)
    "76": "engine_l",      # Engine Controls
    "77": "engine_l",      # Engine Indicating
    "78": "engine_l",      # Exhaust
    "79": "engine_l",      # Oil
    "80": "engine_l",      # Starting
    "29": "hydraulics",    # Hydraulic Power
    "32": "landing_gear",  # Landing Gear
    "27": "flight_controls",  # Flight Controls
    "22": "avionics",      # Auto Flight
    "23": "avionics",      # Communications
    "24": "avionics",      # Electrical Power
    "31": "avionics",      # Indicating / Recording
    "34": "avionics",      # Navigation
    "45": "avionics",      # Central Maintenance System
}

# L/R split heuristic keywords (applied to PartName + ProblemDescription)
ENGINE_RIGHT_KEYWORDS = ["right", " r ", "#2", "eng 2", "engine 2", "-2", "no. 2", "no.2"]
BLEED_RIGHT_KEYWORDS = ["right", " r ", "rh", "r.h.", "no. 2", "no.2"]

# --- Synthetic fleet ---
SYNTHETIC_TAILS = [
    "N823AN", "N917AN", "N341AN", "N788AN", "N456UA",
    "N112SW", "N334SW", "N221WN", "N889DL", "N554DL",
    "N221UA", "N445AA", "N667UA", "N334AA", "N778DL",
    "N901SW", "N223UA", "N445DL", "N112AA", "N334DL",
]

# Demo scenario seeds: health states intentionally degraded for pitch
DEMO_SEEDS: dict[str, dict[str, float]] = {
    "N823AN": {"bleed_air_l": 0.45},                          # → engine_l cascade
    "N917AN": {"hydraulics": 0.38},                           # → flight_controls cascade
    "N341AN": {"apu": 0.42, "bleed_air_r": 0.51},            # dual degradation
    "N788AN": {"landing_gear": 0.35},                         # near-limit LG
    "N456UA": {},                                              # healthy exemplar
}

# --- Simulation ---
DEFAULT_MONTE_CARLO_BRANCHES = 200
DEFAULT_FORWARD_HORIZON = 10       # flight cycles
FAILURE_THRESHOLD = 0.15           # health below this = failed
ALERT_THRESHOLDS = {
    "nominal": 0.70,
    "watch": 0.45,
    "caution": 0.20,
}

# Base decay rates per subsystem per cycle (tuned to SDR fault rates)
BASE_DECAY_RATES: dict[str, float] = {
    "engine_l": 0.004,
    "engine_r": 0.004,
    "apu": 0.006,
    "bleed_air_l": 0.005,
    "bleed_air_r": 0.005,
    "hydraulics": 0.003,
    "landing_gear": 0.004,
    "avionics": 0.002,
    "flight_controls": 0.002,
}

NOISE_STD: dict[str, float] = {s: 0.008 for s in SUBSYSTEMS}
