from __future__ import annotations

import random

import numpy as np
import torch


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def concatenate_debug_states(src_debug_states, tgt_debug_states):
    """
    Update the debug states with the latest values from the chosen driver's debug states.
    """
    for key, value in src_debug_states.items():
        if value:  # Make sure there is at least one entry in the list
            tgt_debug_states[key].append(value[-1])  # Append the last entry of each property

def normalize_angle(rad):
    """
    Normalize an angle in radians to the range [-pi, pi].

    https://stackoverflow.com/a/2321125
    """
    return np.arctan2(np.sin(rad), np.cos(rad))


def normalize_vector(vec, axis=0):
    """
    Normalize a vector.
    """
    return vec / np.linalg.norm(vec, axis=axis)


def compute_curvature(p1, p2, p3):
    """
    Compute curvature (magnitude) between three points.

    Parameters:
        p1, p2, p3 (np.ndarray): Three consecutive points on the track.

    Returns:
        float: The curvature (magnitude).
    """
    # Vectors between the points
    v1 = p2 - p1
    v2 = p3 - p2
    cross_product = np.cross(v1, v2)  # Cross product to find the area of the triangle (magnitude only)
    norm_v1 = np.linalg.norm(v1)  # Lengths of the vectors
    norm_v2 = np.linalg.norm(v2)
    return (
        2 * abs(cross_product) / (norm_v1 * norm_v2 * np.linalg.norm(p3 - p1))
    )  # Curvature = inverse of radius of the circle formed by p1, p2, p3


def compute_signed_cte(vehicle_position, closest_point, start_point, end_point):
    """
    Compute the signed cross-track error (CTE) of a vehicle.
    Parameters:
    - vehicle_position (numpy.ndarray): The position of the vehicle.
    - closest_point (numpy.ndarray): The closest point on the path to the vehicle.
    - start_point (numpy.ndarray): The starting point of the path.
    - end_point (numpy.ndarray): The ending point of the path.
    Returns:
    - float: The signed cross-track error (CTE) of the vehicle.
    """

    cte = np.linalg.norm(vehicle_position - closest_point)  # The distance from the start point to the vehicle
    path_vector = normalize_vector(end_point - start_point)  # The path tangent
    vehicle_vector = vehicle_position - closest_point  # Vector from the closest point to the vehicle
    normal_vector = np.array([-path_vector[1], path_vector[0]])  # Normal vector to path.
    return cte * np.sign(np.dot(vehicle_vector, normal_vector))


def compute_he(pose, start_point, end_point):
    """
    Compute the heading error between the pose and the path.
    Parameters:
    - pose (list): The pose of the car, represented as [x, y, yaw].
    - start_point (list): The starting point of the path, represented as [x, y].
    - end_point (list): The ending point of the path, represented as [x, y].
    Returns:
    - float: The heading error between the pose and the path, normalized to the range [-pi, pi].
    """
    path_heading = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])  # Path heading
    return normalize_angle(pose[2] - path_heading)  # Diff yaw and the path heading. Normalize to [-pi, pi]


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array.
    Parameters:
    - tensor (torch.Tensor): The input PyTorch tensor.
    Returns:
    - numpy.ndarray: The converted NumPy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    raise ValueError("Input must be a PyTorch tensor.")
