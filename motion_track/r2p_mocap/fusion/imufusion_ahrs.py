"""
xioTechnologies Fusion AHRS wrapper  (pip install imufusion)

Handles API differences across imufusion versions — Settings/Convention
are applied when available; falls back to Ahrs defaults otherwise.
Interface matches MadgwickAHRS: update(gyro_rad_s, accel_m_s2, mag=None) -> [w,x,y,z]
"""
from __future__ import annotations
import numpy as np

_AVAIL: bool | None = None


def _check() -> bool:
    global _AVAIL
    if _AVAIL is None:
        try:
            import imufusion  # noqa: F401
            _AVAIL = True
        except ImportError:
            _AVAIL = False
    return _AVAIL


def _apply_settings(lib, ahrs, gain: float, gyro_range_dps: float, sample_rate: float):
    """Try to apply Settings — silently skip if the API doesn't exist in this version."""
    try:
        # Find Convention.NWU (different versions use different casing)
        conv_cls = getattr(lib, "Convention", None)
        if conv_cls is None:
            return
        nwu = getattr(conv_cls, "NWU",
              getattr(conv_cls, "Nwu",
              getattr(conv_cls, "nwu", None)))
        if nwu is None:
            return
        Settings = getattr(lib, "Settings", None)
        if Settings is None:
            return
        ahrs.settings = Settings(
            nwu,
            gain,
            gyro_range_dps,
            10,                      # acceleration rejection (deg)
            10,                      # magnetic rejection (deg)
            int(5 * sample_rate),    # recovery trigger period (samples)
        )
    except Exception:
        pass  # use Ahrs defaults — still works, just less tuned


class ImufusionAHRS:
    """
    Drop-in replacement for MadgwickAHRS using the imufusion C library.
    Gyro: rad/s  →  converted to deg/s internally.
    Accel: m/s²  →  converted to g internally.
    Returns [w, x, y, z] quaternion.
    """

    def __init__(self, sample_rate: float = 50.0,
                 gain: float = 0.5, gyro_range_dps: float = 2000.0):
        if not _check():
            raise ImportError("imufusion not installed — run: pip install imufusion")
        import imufusion as _lib
        self._lib    = _lib
        self._dt     = 1.0 / sample_rate
        self._ahrs   = _lib.Ahrs()
        _apply_settings(_lib, self._ahrs, gain, gyro_range_dps, sample_rate)

        # Offset corrects static gyro bias; constructor may not exist in all versions
        try:
            self._offset = _lib.Offset(int(sample_rate))
        except Exception:
            self._offset = None

    def update(self, gyro: np.ndarray, accel: np.ndarray,
               mag: np.ndarray | None = None) -> np.ndarray:
        g = np.degrees(np.asarray(gyro,  dtype=np.float64))   # rad/s → deg/s
        a = np.asarray(accel, dtype=np.float64) / 9.81         # m/s²  → g

        if self._offset is not None:
            try:
                g = self._offset.update(g)
            except Exception:
                pass

        try:
            if mag is not None:
                self._ahrs.update(g, a, np.asarray(mag, dtype=np.float64), self._dt)
            else:
                self._ahrs.update_no_magnetometer(g, a, self._dt)
        except TypeError:
            # Some versions have positional-only or different arg order
            self._ahrs.update(g, a, self._dt)

        return np.array(self._ahrs.quaternion.wxyz, dtype=np.float64)

    def reset(self):
        try:
            self._ahrs.reset()
        except Exception:
            pass

    @staticmethod
    def available() -> bool:
        return _check()
