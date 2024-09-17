from __future__ import annotations

##############################
# Fixed controlling parameters
MAX_SPEED = 80.0
MIN_SPEED = 40.0
MAX_GAS = 0.8
MAX_BRAKE = 0.4

##############################
# Cross Track Error parameters
CTE_START_OFFSET = -2
CTE_END_OFFSET = 10

# Heading Error parameters
HE_START_OFFSET = -2
HE_END_OFFSET = 11

# Curvature parameters
MAX_CURVATURE = 0.08
CURVATURE_START_OFFSET = 1
CURVATURE_END_OFFSET = 11  # Curvature parameter
##############################
# Lateral cross-track error PID parameters
LATERAL_PID_CTE_KP = 0.01
# Lateral heading error PID parameters
LATERAL_PID_HE_KP = 0.5
# Longitudinal PID parameters
LONGITUDINAL_PID_GAS_KP = 0.8
LONGITUDINAL_PID_BRAKE_KP = 1.0
##############################
# Penalty parameters
DO_SPEED_PENALTY = True
DO_GAS_PENALTY = True
