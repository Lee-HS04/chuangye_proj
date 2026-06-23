"""
backend/profile.py
------------------
Stores and retrieves user profiles containing:
  - Limb lengths from GVHMR calibration
  - Body measurements (height, weight, age — user-entered)
  - IMU zero-reference offsets (per sensor)
  - Calibration history with timestamps
  - Test history summaries

Profiles saved as JSON under outputs/profiles/<profile_id>.json
A "current_profile.json" symlink/pointer tracks the active profile.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

PROFILES_DIR = Path(__file__).parent.parent / "outputs" / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_PTR = PROFILES_DIR / "_current.json"   # points to active profile id


# ─────────────────────────────────────────────
# DEFAULT PROFILE TEMPLATE
# ─────────────────────────────────────────────

def _empty_profile(name: str = "Default") -> dict:
    return {
        "id":          str(uuid.uuid4()),
        "name":        name,
        "created_at":  time.time(),
        "updated_at":  time.time(),

        # User-entered body info
        "body": {
            "height_cm":   None,
            "weight_kg":   None,
            "age":         None,
            "gender":      None,   # "male" | "female" | "other" | None
            "sport":       None,
        },

        # GVHMR-measured limb lengths (metres)
        "limb_lengths": {
            "thigh_l":       None,
            "shin_l":        None,
            "thigh_r":       None,
            "shin_r":        None,
            "hip_width":     None,
            "pelvis_height": None,
        },

        # Derived measurements (computed from limb lengths)
        "derived": {
            "leg_length_l_cm":    None,   # thigh + shin in cm
            "leg_length_r_cm":    None,
            "leg_symmetry_pct":   None,   # 100 = perfect
        },

        # IMU zero-reference quaternions {sensor_id: [w, x, y, z]}
        "imu_offsets": {},

        # Calibration history (list of past calibration events)
        "calibration_history": [],

        # Test history (summary of past sessions)
        "test_history": [],
    }


# ─────────────────────────────────────────────
# PROFILE MANAGER
# ─────────────────────────────────────────────

class ProfileManager:
    def __init__(self):
        self._current: Optional[dict] = None
        self._load_current()

    # ── Current profile ───────────────────────────────────────────────────────

    def _load_current(self):
        """Load the last active profile on startup."""
        if CURRENT_PTR.exists():
            try:
                ptr = json.loads(CURRENT_PTR.read_text())
                profile_path = PROFILES_DIR / f"{ptr['id']}.json"
                if profile_path.exists():
                    self._current = json.loads(profile_path.read_text())
                    return
            except Exception:
                pass
        # Create a default profile if none exists
        self._current = _empty_profile("Default")
        self._save_current()

    def _save_current(self):
        if self._current is None:
            return
        self._current["updated_at"] = time.time()
        path = PROFILES_DIR / f"{self._current['id']}.json"
        path.write_text(json.dumps(self._current, indent=2, default=_json_default))
        CURRENT_PTR.write_text(json.dumps({"id": self._current["id"]}))

    @property
    def current(self) -> dict:
        return self._current or _empty_profile()

    # ── Profile CRUD ──────────────────────────────────────────────────────────

    def list_profiles(self) -> list[dict]:
        profiles = []
        for p in PROFILES_DIR.glob("*.json"):
            if p.name.startswith("_"):
                continue
            try:
                data = json.loads(p.read_text())
                profiles.append({
                    "id":         data["id"],
                    "name":       data["name"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "calibrated": data["limb_lengths"]["thigh_l"] is not None,
                    "height_cm":  data["body"]["height_cm"],
                })
            except Exception:
                continue
        return sorted(profiles, key=lambda x: x["updated_at"], reverse=True)

    def create_profile(self, name: str) -> dict:
        profile = _empty_profile(name)
        self._current = profile
        self._save_current()
        return profile

    def switch_profile(self, profile_id: str) -> Optional[dict]:
        path = PROFILES_DIR / f"{profile_id}.json"
        if not path.exists():
            return None
        self._current = json.loads(path.read_text())
        CURRENT_PTR.write_text(json.dumps({"id": profile_id}))
        return self._current

    def delete_profile(self, profile_id: str) -> bool:
        path = PROFILES_DIR / f"{profile_id}.json"
        if not path.exists():
            return False
        path.unlink()
        # If deleting the active profile, switch to default
        if self._current and self._current["id"] == profile_id:
            self._current = _empty_profile("Default")
            self._save_current()
        return True

    # ── Calibration data ──────────────────────────────────────────────────────

    def save_calibration(
        self,
        limb_lengths: dict,
        imu_offsets: Optional[dict] = None,
    ) -> dict:
        """
        Called after successful GVHMR calibration + IMU zero-reference capture.
        Stores limb lengths, computes derived metrics, logs to history.
        """
        if self._current is None:
            self._current = _empty_profile()

        # Store limb lengths
        for k, v in limb_lengths.items():
            if k in self._current["limb_lengths"]:
                self._current["limb_lengths"][k] = float(v) if v is not None else None

        # Compute derived measurements
        thigh_l = limb_lengths.get("thigh_l")
        shin_l  = limb_lengths.get("shin_l")
        thigh_r = limb_lengths.get("thigh_r")
        shin_r  = limb_lengths.get("shin_r")

        if thigh_l and shin_l:
            self._current["derived"]["leg_length_l_cm"] = round((thigh_l + shin_l) * 100, 1)
        if thigh_r and shin_r:
            self._current["derived"]["leg_length_r_cm"] = round((thigh_r + shin_r) * 100, 1)

        ll = self._current["derived"]["leg_length_l_cm"]
        lr = self._current["derived"]["leg_length_r_cm"]
        if ll and lr:
            mean_leg = (ll + lr) / 2
            self._current["derived"]["leg_symmetry_pct"] = round(
                100 * (1 - abs(ll - lr) / mean_leg), 1
            ) if mean_leg > 0 else None

        # Store IMU offsets if provided
        if imu_offsets:
            self._current["imu_offsets"] = imu_offsets

        # Log to history
        self._current["calibration_history"].append({
            "timestamp":    time.time(),
            "limb_lengths": limb_lengths,
            "has_imu":      imu_offsets is not None,
        })

        self._save_current()
        return self._current

    def save_imu_offsets(self, offsets: dict):
        """Save IMU zero-reference quaternions independently."""
        if self._current is None:
            return
        self._current["imu_offsets"] = offsets
        self._save_current()

    # ── Body info ─────────────────────────────────────────────────────────────

    def update_body_info(self, **kwargs) -> dict:
        """Update user-entered body measurements."""
        if self._current is None:
            return {}
        allowed = {"height_cm", "weight_kg", "age", "gender", "sport"}
        for k, v in kwargs.items():
            if k in allowed:
                self._current["body"][k] = v
        self._save_current()
        return self._current

    # ── Test history ──────────────────────────────────────────────────────────

    def save_test_result(self, session_id: str, test_id: str, summary: dict):
        """Append a test session summary to the profile's test history."""
        if self._current is None:
            return
        self._current["test_history"].append({
            "session_id": session_id,
            "test_id":    test_id,
            "timestamp":  time.time(),
            "summary":    summary,
        })
        # Keep last 50 sessions
        self._current["test_history"] = self._current["test_history"][-50:]
        self._save_current()

    def get_test_history(self, test_id: Optional[str] = None) -> list:
        if self._current is None:
            return []
        history = self._current.get("test_history", [])
        if test_id:
            history = [h for h in history if h["test_id"] == test_id]
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)

    # ── Limb lengths for skeleton engine ─────────────────────────────────────

    def get_limb_lengths(self) -> Optional[dict]:
        if self._current is None:
            return None
        ll = self._current.get("limb_lengths", {})
        if ll.get("thigh_l") is None:
            return None
        return {k: v for k, v in ll.items() if v is not None}

    def get_imu_offsets(self) -> dict:
        if self._current is None:
            return {}
        return self._current.get("imu_offsets", {})


def _json_default(obj):
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    raise TypeError(type(obj))