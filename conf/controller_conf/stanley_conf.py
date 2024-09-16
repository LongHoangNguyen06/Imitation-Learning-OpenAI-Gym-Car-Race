from __future__ import annotations

##############################
# Cross Track Error parameters
CTE_START_OFFSET = 1
CTE_END_OFFSET = 2

# Heading Error parameters
HE_START_OFFSET = 1
HE_END_OFFSET = 2

# Curvature parameters
MAX_CURVATURE = 0.08
CURVATURE_START_OFFSET = 1
CURVATURE_END_OFFSET = 10  # Curvature parameter

PID_SUB_CONFIGS = [
    [0.01, "conf/controller_conf/stanley_extras/pid_normal_driver.py"],
    [0.02, "conf/controller_conf/stanley_extras/pid_corner_1.py"],
    [0.03, "conf/controller_conf/stanley_extras/pid_corner_2.py"],
    [1000, "conf/controller_conf/stanley_extras/pid_corner_3.py"],
]

##############################
# Stanley Controller parameters
CTE_COEFFICIENT = 0.01
HE_COEFFICIENT = 30.0
DAMPING_FACTOR = 0.000001
