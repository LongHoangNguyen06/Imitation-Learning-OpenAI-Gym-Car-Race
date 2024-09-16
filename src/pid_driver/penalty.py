from __future__ import annotations

import numpy as np


def gas(wheels_omegas_std):
    """
    Compute the gas penalty based on the standard deviation of the wheel omegas.

    Args:
        wheels_omegas (np.ndarray): The wheel omegas.

    Returns:
        float: The gas penalty.
    """
    if wheels_omegas_std > 15:
        return 1.0
    elif wheels_omegas_std > 12.5:
        return 0.55
    elif wheels_omegas_std > 10.0:
        return 0.5
    elif wheels_omegas_std > 9.0:
        return 0.45
    elif wheels_omegas_std > 8.0:
        return 0.425
    elif wheels_omegas_std > 7.0:
        return 0.4
    elif wheels_omegas_std > 6.0:
        return 0.35
    elif wheels_omegas_std > 5.0:
        return 0.2
    elif wheels_omegas_std > 4.0:
        return 0.1
    return 0.0


def clamp_speed(desired_speed, wheels_omegas_std):
    if wheels_omegas_std > 11.0:
        return 20
    elif wheels_omegas_std > 10.0:
        return 30
    elif wheels_omegas_std > 9:
        return 35
    elif wheels_omegas_std > 8:
        return 45
    return desired_speed


def clamp_steering(control, gas):
    if gas > 0.75:
        control = np.clip(control, -0.75, 0.75)
    return control


def desired_speed(wheels_omegas_std, curvature):
    """
    Compute the desired speed penalty based on the standard deviation of the wheel omegas.

    Args:
        wheels_omegas (np.ndarray): The wheel omegas.

    Returns:
        float: The desired speed penalty.
    """
    if wheels_omegas_std > 10.0:
        return 100.0
    elif wheels_omegas_std > 8.5:
        return 90.0
    elif wheels_omegas_std > 7.5:
        return 80.0
    elif wheels_omegas_std > 7.0:
        return 55.0
    elif wheels_omegas_std > 6.5:
        return 45.0
    elif wheels_omegas_std > 6.0:
        return 40.0
    elif wheels_omegas_std > 5.5:
        return 35.0
    elif wheels_omegas_std > 5.0:
        return 30.0
    elif wheels_omegas_std > 4.5:
        return 20.0
    elif wheels_omegas_std > 4.0:
        return 15.0
    elif wheels_omegas_std > 3.0:
        return 10.0
    return 0.0
