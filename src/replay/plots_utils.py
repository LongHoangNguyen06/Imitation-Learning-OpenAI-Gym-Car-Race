from __future__ import annotations

import numpy as np


def get_action_arrow(action: np.ndarray, pose: np.ndarray) -> tuple:
    """
    Convert an action tuple (steering, gas, brake) into an arrow (dx, dy).

    Args:
        action (tuple): The action tuple (steering, gas, brake).
        pose (tuple): The car's pose (x, y, angle).

    Returns:
        (dx, dy): Arrow components representing the effect of the action.
    """
    steering, gas, brake = action

    # Compute the net acceleration considering gas and brake
    net_acceleration = gas - brake

    # Adjust the car's angle based on the steering input
    # A full left (-1) turns the car by -90 degrees, and full right (+1) by +90 degrees
    adjusted_angle = -steering * np.pi / 2 + pose[2]
    if net_acceleration < 0:
        if steering < 0:
            adjusted_angle -= np.pi / 2
        elif steering > 0:
            adjusted_angle += np.pi / 2
    # The arrow's length is proportional to the net acceleration
    arrow_length = 25 * net_acceleration  # Adjust length to visualize easily
    if arrow_length == 0 and steering != 0:
        arrow_length = 12.5
    # Compute the dx and dy components of the arrow based on the angle
    dx = arrow_length * np.cos(adjusted_angle)
    dy = arrow_length * np.sin(adjusted_angle)

    return dx, dy


def get_car_orientation_arrow(pose: np.ndarray, length: float=25.0) -> tuple:
    """
    Compute the car's orientation angle in radians.

    Args:
        pose (tuple): The car's pose (x, y, angle).

    Returns:
        float: The car's orientation angle in radians.
    """
    angle = pose[2]
    dx = length * np.cos(angle)
    dy = length * np.sin(angle)
    return dx, dy


def action_to_string(action: np.ndarray) -> str:
    """
    Return a string representation of the action tuple.
    """
    steering, gas, brake = action
    if steering == 0 and gas == 0 and brake == 0:
        return "NO ACTION"
    action_str = ""
    if steering != 0:
        action_str += "RIGHT " if steering > 0 else "LEFT "
    if gas != 0:
        action_str += "GAS "
    if brake != 0:
        action_str += "BRAKE "
    return action_str.strip()


def get_driver_name(driver_id):
    if driver_id == 0:
        return "Normal driver"
    elif driver_id == 1:
        return "Corner 1 driver"
    elif driver_id == 2:
        return "Corner 2 driver"
    elif driver_id == 3:
        return "Corner 3 driver"
    raise ValueError(f"Unknown driver id: {driver_id}")
