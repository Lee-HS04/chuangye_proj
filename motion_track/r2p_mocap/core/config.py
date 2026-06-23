from enum import Enum
from dataclasses import dataclass
from typing import List


class SuitMode(Enum):
    CONSUMER     = "consumer"      # 6 IMUs — lower body
    PROFESSIONAL = "professional"  # 15 IMUs — full body


class SensorID:
    # Consumer (6)
    PELVIS     = "pelvis"
    CHEST      = "chest"
    L_THIGH    = "l_thigh"
    R_THIGH    = "r_thigh"
    L_SHIN     = "l_shin"
    R_SHIN     = "r_shin"
    # Professional extensions (9)
    HEAD       = "head"
    L_SHOULDER = "l_shoulder"
    R_SHOULDER = "r_shoulder"
    L_UPPER_ARM = "l_upper_arm"
    R_UPPER_ARM = "r_upper_arm"
    L_FOREARM  = "l_forearm"
    R_FOREARM  = "r_forearm"
    L_FOOT     = "l_foot"
    R_FOOT     = "r_foot"


CONSUMER_SENSORS: List[str] = [
    SensorID.PELVIS, SensorID.CHEST,
    SensorID.L_THIGH, SensorID.R_THIGH,
    SensorID.L_SHIN,  SensorID.R_SHIN,
]

PROFESSIONAL_SENSORS: List[str] = CONSUMER_SENSORS + [
    SensorID.HEAD,
    SensorID.L_SHOULDER, SensorID.R_SHOULDER,
    SensorID.L_UPPER_ARM, SensorID.R_UPPER_ARM,
    SensorID.L_FOREARM, SensorID.R_FOREARM,
    SensorID.L_FOOT, SensorID.R_FOOT,
]


@dataclass
class R2PConfig:
    mode:                  SuitMode = SuitMode.CONSUMER
    sample_rate:           float    = 100.0   # Hz
    madgwick_beta:         float    = 0.033   # filter gain — lower = smoother/more lag
    smoothing_alpha:       float    = 0.75    # quaternion low-pass (1=raw, 0=frozen)
    body_height:           float    = 1.75    # metres
    enable_ik:             bool     = True
    enable_foot_contact:   bool     = True
    enable_drift_correction: bool   = True
    ik_iterations:         int      = 20
    drift_correction_rate: float    = 0.003   # yaw bleed-back rate per frame

    @property
    def active_sensors(self) -> List[str]:
        return PROFESSIONAL_SENSORS if self.mode == SuitMode.PROFESSIONAL else CONSUMER_SENSORS

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate
