"""
R2P — Inertial Motion Capture Suit Framework
Professional-grade real-time skeletal reconstruction from distributed IMU sensors.
"""

from .engine            import R2PEngine
from .core.config       import R2PConfig, SuitMode
from .simulation        import IMUSimulator, MotionType
from .visualization     import SkeletonRenderer
from .output            import SkeletonOutput, MotionCapFrame, BVHExporter
from .calibration       import Calibrator, CalibrationData

__version__ = "1.0.0"
__all__ = [
    "R2PEngine", "R2PConfig", "SuitMode",
    "IMUSimulator", "MotionType",
    "SkeletonRenderer",
    "SkeletonOutput", "MotionCapFrame", "BVHExporter",
    "Calibrator", "CalibrationData",
]
