from __future__ import annotations

##############################
# Fixed controlling parameters
MAX_SPEED = 150.0
MIN_SPEED = 50.0
MAX_GAS = 1.0
MAX_BRAKE = 0.25

##############################
# Cross Track Error parameters
CTE_START_OFFSET = 0
CTE_END_OFFSET = 2

# Heading Error parameters
HE_START_OFFSET = 0
HE_END_OFFSET = 2

# Curvature parameters
MAX_CURVATURE = 0.08
CURVATURE_START_OFFSET = 1
CURVATURE_END_OFFSET = 10  # Curvature parameter
##############################
# Penalty parameters
DO_SPEED_PENALTY = False
DO_GAS_PENALTY = False

##############################
# Lateral cross-track error PID parameters
LATERAL_PID_CTE_KP = 0.0075
LATERAL_PID_CTE_KI = 0
LATERAL_PID_CTE_KD = 0

# Lateral heading error PID parameters
LATERAL_PID_HE_KP = 0.5
LATERAL_PID_HE_KI = 0
LATERAL_PID_HE_KD = 0

# Longitudinal PID parameters
LONGITUDINAL_PID_GAS_KP = 0.8
LONGITUDINAL_PID_GAS_KI = 0
LONGITUDINAL_PID_GAS_KD = 0

LONGITUDINAL_PID_BRAKE_KP = 1.0
LONGITUDINAL_PID_BRAKE_KI = 0
LONGITUDINAL_PID_BRAKE_KD = 0

PID_SUB_CONFIGS = [
    [0.01, "cconf/controller_conf/pid_extras/pid_normal_driver.py"],
    [0.02, "conf/controller_conf/pid_extras/pid_corner_1.py"],
    [0.03, "conf/controller_conf/pid_extras/pid_corner_2.py"],
    [1000, "conf/controller_conf/pid_extras/pid_corner_3.py"],
]

DO_SPEED_PENALTY = True
DO_GAS_PENALTY = True
