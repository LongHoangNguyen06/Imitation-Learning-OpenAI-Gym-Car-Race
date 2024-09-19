from __future__ import annotations

import numpy as np

from src.utils import utils


class PathMetricsComputer:
    def __init__(self, track: np.ndarray, conf):
        self.vehicle_position = None
        self.conf = conf
        self.track = track

        # Distances from pose to track
        self.distances = None
        self.closest_idx = 0
        self.cloest_point = None

        # Cross track error
        self.cte_start_idx = 0
        self.cte_end_idx = 0
        self.cte_start_point = None
        self.cte_end_point = None
        self.cte = 0
        self.cte_signed = 0

        # Heading error
        self.he_start_idx = 0
        self.he_end_idx = 0
        self.he_start_point = None
        self.he_end_point = None
        self.he = 0

        # Curvature
        self.curvature_start_idx = 0
        self.curvature_end_idx = 0
        self.curvature = 0

    def _compute_projection_and_waypoints(self, pose):
        """
        Computes the next waypoint on the track based on the vehicle's pose and curvature.
        Args:
            pose (tuple): The current pose of the vehicle, represented as a tuple of (x, y) coordinates.
        Returns:
            None
        """
        # Compute the distances from the vehicle to each point on the track
        self.vehicle_position = np.array([[pose[0], pose[1]]])
        self.distances = np.linalg.norm(self.track - self.vehicle_position, axis=1)
        self.closest_idx = np.argmin(self.distances)
        self.closest_point = self.track[self.closest_idx, :]

        # Cross track error way points
        self.cte_start_idx = (self.closest_idx + self.conf.CTE_START_OFFSET) % len(self.track)
        self.cte_end_idx = (self.closest_idx + self.conf.CTE_END_OFFSET) % len(self.track)
        self.cte_start_point = self.track[self.cte_start_idx, :]
        self.cte_end_point = self.track[self.cte_end_idx, :]

        # Heading error way points
        self.he_start_idx = (self.closest_idx + self.conf.HE_START_OFFSET) % len(self.track)
        self.he_end_idx = (self.he_start_idx + self.conf.HE_END_OFFSET) % len(self.track)
        self.he_start_point = self.track[self.he_start_idx, :]
        self.he_end_point = self.track[self.he_end_idx, :]

        # Curvature way points
        self.curvature_start_idx = (self.closest_idx + self.conf.CURVATURE_START_OFFSET) % len(self.track)
        self.curvature_end_idx = (self.curvature_start_idx + self.conf.CURVATURE_END_OFFSET) % len(self.track)
        self.curvature_n_points = (self.curvature_end_idx - self.curvature_start_idx) % len(self.track) + 1

    def _compute_curvatures(self):
        """
        Compute signed curvature and total curvature over the waypoints.
        """
        self.curvature = 0
        for i in range(self.curvature_start_idx + 1, self.curvature_end_idx - 1):
            p1 = self.track[i - 1]
            p2 = self.track[i]
            p3 = self.track[i + 1]
            self.curvature += utils.compute_curvature(p1, p2, p3)
        self.curvature /= self.curvature_n_points  # Normalize the curvature by the number of waypoints

    def compute_metrics(self, pose):
        """
        Compute the error between the current pose and the desired pose.

        Parameters:
            pose (Pose): The desired pose.

        Returns:
            tuple: A tuple containing the signed cross-track error (signed_cte) and the heading error (he).
        """
        self._compute_projection_and_waypoints(pose)
        self.signed_cte = utils.compute_signed_cte(
            self.vehicle_position, self.closest_point, self.cte_start_point, self.cte_end_point
        )
        self.he = utils.compute_he(pose, self.he_start_point, self.he_end_point)
        self._compute_curvatures()
        return float(self.signed_cte), float(self.he), float(self.curvature)
