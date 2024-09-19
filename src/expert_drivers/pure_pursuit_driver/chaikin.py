from __future__ import annotations

import numpy as np
from numba import njit


@njit
def chaikin_corner_cutting_numba(points, iterations):
    """
    Applies Chaikin's corner cutting algorithm to a list of points.
    Args:
        points (list): List of points to be processed.
        iterations (int): Number of iterations to perform.
    Returns:
        list: List of points after applying the algorithm.
    """
    for _ in range(iterations):
        new_points = []
        n = len(points)
        for i in range(n - 1):
            p_current = points[i]
            q_next = points[i + 1]

            # First new point (q)
            q = 0.75 * p_current + 0.25 * q_next

            # Second new point (r)
            r = 0.25 * p_current + 0.75 * q_next

            new_points.append(q)
            new_points.append(r)
        points = new_points
    return points


def chaikin_corner_cutting(track, sub_sampling_ratio=9, iterations=5):
    """
    Apply Chaikin's corner cutting algorithm to smooth a track.
    Parameters:
    - track (numpy.ndarray): The track points to be smoothed.
    - sub_sampling_ratio (int): The ratio of subsampling to be applied to the track. Default is 9.
    - iterations (int): The number of iterations to perform the corner cutting. Default is 5.
    Returns:
    - numpy.ndarray: The smoothed track points.
    """
    subsampled_track = track[::sub_sampling_ratio]
    return np.array(chaikin_corner_cutting_numba([p for p in subsampled_track], iterations=iterations))
