from __future__ import annotations

##############################
# Fixed controlling parameters
MAX_SPEED = 90.0
MIN_SPEED = 50.0
MAX_GAS = 0.9
MAX_BRAKE = 0.3

##############################
# Cross Track Error parameters
CTE_START_OFFSET = -3
CTE_END_OFFSET = 10

# Heading Error parameters
HE_START_OFFSET = -3
HE_END_OFFSET = 11

# Curvature parameters
MAX_CURVATURE = 0.08
CURVATURE_START_OFFSET = 1
CURVATURE_END_OFFSET = 11  # Curvature parameter
##############################
# Lateral cross-track error PID parameters
LATERAL_PID_CTE_KP = 0.005
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
DO_SPEED_PENALTY = True
DO_GAS_PENALTY = True
