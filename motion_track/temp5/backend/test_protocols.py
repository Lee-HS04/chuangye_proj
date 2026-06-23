"""
backend/test_protocols.py
--------------------------
Defines each test's:
  - Required pose for YOLO gating
  - Recording duration
  - Which biomechanics metrics to compute and display
  - Coaching thresholds and traffic-light rules
  - Display name, description, instructions shown to user

Tests:
  MUSCLE CONDITION group:
    stability  — standing balance, sway analysis
    cmj        — countermovement jump, RSI, flight time
    squat      — rep quality, symmetry, depth, fatigue drift
    gait       — walking asymmetry, cadence, coordination

  SPORTS group:
    running    — stride symmetry, cadence, ground contact, fatigue gait drift, power profile
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricSpec:
    key:         str           # matches biomechanics engine output key
    label:       str           # display label
    unit:        str           # e.g. "°", "%", "m/s", ""
    good_range:  tuple         # (min, max) — GREEN zone
    warn_range:  tuple         # (min, max) — AMBER zone; outside = RED
    higher_is_better: bool = False
    description: str = ""


@dataclass
class TestProtocol:
    id:               str
    name:             str
    group:            str           # "muscle_condition" | "sports"
    sport:            str = ""
    description:      str = ""
    instructions:     list[str] = field(default_factory=list)
    required_pose:    str = "tstand"
    record_duration:  float = 30.0
    hold_duration:    float = 3.0
    metrics:          list[MetricSpec] = field(default_factory=list)
    icon:             str = "🏃"
    # Sensors needed to run at all vs sensors that enrich data if present
    required_sensors: list[str] = field(default_factory=list)
    optional_sensors: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# METRIC SPECS (reusable)
# ─────────────────────────────────────────────

SWAY_VEL     = MetricSpec("sway_velocity",       "Sway velocity",       "m/s", (0, 0.02), (0, 0.05))
STABILITY    = MetricSpec("sway_stability",       "Stability score",     "%",   (80, 100), (60, 100),  higher_is_better=True)
SWAY_CV      = MetricSpec("sway_cv_pct",          "Sway CV",             "%",   (0, 15),   (0, 30))
ASYM_KNEE    = MetricSpec("knee_asymmetry_deg",   "Knee asymmetry",      "°",   (0, 5),    (0, 10))
ASYM_ANKLE   = MetricSpec("ankle_height_asym_m",  "Ankle height asym",   "m",   (0, 0.01), (0, 0.03))
JERK         = MetricSpec("jerk_magnitude",       "Smoothness (jerk)",   "",    (0, 0.5),  (0, 2.0),  description="Lower is smoother")
FATIGUE      = MetricSpec("fatigue_score",        "Fatigue score",       "",    (0, 25),   (0, 50))
KNEE_L       = MetricSpec("knee_l_deg",           "Knee L angle",        "°",   (0, 120),  (0, 140),  higher_is_better=True)
KNEE_R       = MetricSpec("knee_r_deg",           "Knee R angle",        "°",   (0, 120),  (0, 140),  higher_is_better=True)


# ─────────────────────────────────────────────
# ALL PROTOCOLS
# ─────────────────────────────────────────────

PROTOCOLS: dict[str, TestProtocol] = {

    # ── STABILITY ────────────────────────────────────────────────────────────
    "stability": TestProtocol(
        id             = "stability",
        name           = "Body stability",
        group          = "muscle_condition",
        icon           = "⚖️",
        description    = "Measures standing balance and postural control using pelvis and lower-limb IMU data.",
        instructions   = [
            "Stand upright with feet shoulder-width apart, arms at your sides.",
            "Look straight ahead at a fixed point.",
            "Stay as still as possible for 30 seconds.",
            "Do not shift your weight or look around.",
        ],
        required_pose    = "tstand",
        record_duration  = 30.0,
        hold_duration    = 3.0,
        required_sensors = ["pelvis"],                              # minimum to run
        optional_sensors = ["chest", "thigh_l", "thigh_r", "shin_l", "shin_r"],
        metrics          = [STABILITY, SWAY_VEL, SWAY_CV, ASYM_KNEE, JERK, FATIGUE],
    ),

    # ── CMJ ──────────────────────────────────────────────────────────────────
    "cmj": TestProtocol(
        id             = "cmj",
        name           = "Jump / explosiveness",
        group          = "muscle_condition",
        icon           = "⚡",
        description    = "Countermovement jump analysis. Measures explosive power, flight time, and reactive strength index.",
        instructions   = [
            "Stand upright with feet shoulder-width apart.",
            "When you are ready, perform 5 maximal countermovement jumps.",
            "Rest 10 seconds between each jump.",
            "Land softly with both feet.",
        ],
        required_pose    = "tstand",
        record_duration  = 60.0,
        hold_duration    = 2.0,
        required_sensors = ["pelvis"],
        optional_sensors = ["thigh_l", "thigh_r", "shin_l", "shin_r", "chest"],
        metrics          = [
            MetricSpec("rsi",        "Reactive Strength Index", "", (0.8, 3.0), (0.4, 3.0), higher_is_better=True),
            MetricSpec("rep_count",  "Jumps detected",          "", (5, 5),     (3, 7),      higher_is_better=True),
            ASYM_KNEE, ASYM_ANKLE, FATIGUE,
        ],
    ),

    # ── SQUAT ─────────────────────────────────────────────────────────────────
    "squat": TestProtocol(
        id             = "squat",
        name           = "Squat quality",
        group          = "muscle_condition",
        icon           = "🏋️",
        description    = "Analyses squat mechanics: depth, symmetry, fatigue drift over reps, and compensation patterns.",
        instructions   = [
            "Stand with feet shoulder-width apart, toes slightly turned out.",
            "Perform 10 bodyweight squats at a controlled pace.",
            "Go as deep as comfortable while keeping heels on the floor.",
            "Keep your arms straight forward for balance.",
        ],
        required_pose    = "tstand",
        record_duration  = 60.0,
        hold_duration    = 2.0,
        required_sensors = ["pelvis", "thigh_l", "thigh_r"],
        optional_sensors = ["shin_l", "shin_r", "chest"],
        metrics          = [ASYM_KNEE, KNEE_L, KNEE_R, SWAY_CV, JERK, FATIGUE,
                            MetricSpec("rep_count", "Reps completed", "", (10, 10), (5, 15), higher_is_better=True)],
    ),

    # ── GAIT ──────────────────────────────────────────────────────────────────
    "gait": TestProtocol(
        id             = "gait",
        name           = "Gait analysis",
        group          = "muscle_condition",
        icon           = "🚶",
        description    = "Walking asymmetry, step timing consistency, and coordination quality.",
        instructions   = [
            "Walk at a natural pace in a straight line.",
            "Turn around and walk back.",
            "Repeat 4 times.",
            "Do not look at your feet.",
        ],
        required_pose    = "tstand",
        record_duration  = 40.0,
        hold_duration    = 2.0,
        required_sensors = ["pelvis", "thigh_l", "thigh_r"],
        optional_sensors = ["shin_l", "shin_r", "chest"],
        metrics          = [
            ASYM_KNEE, ASYM_ANKLE,
            MetricSpec("stride_symmetry_pct", "Stride symmetry", "%",   (90, 100), (80, 100), higher_is_better=True),
            MetricSpec("cadence_spm",         "Cadence",         "spm", (90, 130), (70, 150)),
            JERK, FATIGUE,
        ],
    ),

    # ── RUNNING ───────────────────────────────────────────────────────────────
    "running": TestProtocol(
        id             = "running",
        name           = "Running analysis",
        group          = "sports",
        sport          = "running",
        icon           = "🏃",
        description    = "Comprehensive running biomechanics: stride symmetry, cadence, ground contact time, fatigue gait drift, and power transfer efficiency.",
        instructions   = [
            "Run at your comfortable training pace.",
            "Keep a straight line along a flat surface.",
            "Run for at least 60 seconds.",
            "Do not alter your natural stride for this test.",
        ],
        required_pose    = "tstand",
        record_duration  = 60.0,
        hold_duration    = 2.0,
        required_sensors = ["pelvis", "thigh_l", "thigh_r", "shin_l", "shin_r"],
        optional_sensors = ["chest"],
        metrics          = [
            MetricSpec("stride_symmetry_pct",     "Stride symmetry",          "%",  (92, 100), (85, 100), higher_is_better=True),
            MetricSpec("cadence_spm",              "Cadence",                  "spm",(170, 190),(155, 210)),
            MetricSpec("gct_asymmetry_ms",         "Ground contact asymmetry", "ms", (0, 10),   (0, 25)),
            MetricSpec("vertical_oscillation_cm",  "Vertical oscillation",     "cm", (6, 10),   (4, 14)),
            MetricSpec("running_smoothness",       "Running smoothness",       "%",  (70, 100), (50, 100), higher_is_better=True),
            MetricSpec("fatigue_gait_drift",       "Fatigue gait drift",       "",   (0, 0.01), (0, 0.05)),
            ASYM_KNEE,
        ],
    ),
}


def get_protocol(test_id: str) -> TestProtocol:
    p = PROTOCOLS.get(test_id)
    if p is None:
        raise KeyError(f"Unknown test protocol: {test_id!r}")
    return p


def get_by_group(group: str) -> list[TestProtocol]:
    return [p for p in PROTOCOLS.values() if p.group == group]


def can_run(test_id: str, connected: list[str]) -> bool:
    """True if all required sensors for this test are in the connected list."""
    p = PROTOCOLS.get(test_id)
    if p is None:
        return False
    return all(s in connected for s in p.required_sensors)


def get_missing_sensors(test_id: str, connected: list[str]) -> list[str]:
    """Returns the list of required sensors that are not currently connected."""
    p = PROTOCOLS.get(test_id)
    if p is None:
        return []
    return [s for s in p.required_sensors if s not in connected]


def metric_grade(spec: MetricSpec, value: float) -> str:
    """Returns 'green', 'amber', or 'red' for a metric value."""
    lo_g, hi_g = spec.good_range
    lo_w, hi_w = spec.warn_range
    if lo_g <= value <= hi_g:
        return "green"
    if lo_w <= value <= hi_w:
        return "amber"
    return "red"