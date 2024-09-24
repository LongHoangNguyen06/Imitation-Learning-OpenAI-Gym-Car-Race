from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.expert_drivers.abstract_classes.abstract_controller import AbstractController
from src.expert_drivers.pid_driver.pid_driver_controller import PidDriverController
from src.expert_drivers.pure_pursuit_driver.chaikin import chaikin_corner_cutting
from src.utils import utils


class PurePursuitController(AbstractController):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.longtitudinal_controller = PidDriverController(conf)
        self.debug_states = defaultdict(list)

    def reset(self):
        """
        Resets the pure pursuit controller.
        This method resets the pure pursuit controller by calling the reset method of the superclass,
        resetting the longitudinal controller, and clearing the debug states.
        Parameters:
            None
        Returns:
            None
        """

        super().reset()
        self.longtitudinal_controller.reset()
        self.debug_states = defaultdict(list)

    def get_action(self, observation, info, *args, **kwargs):
        """
        Get the continuous action to be executed by the car.

        Args:
            observation (np.ndarray): The observation received from the environment.
            info (dict): Additional information about the environment.

        Returns:
            np.ndarray: The continuous action to be executed by the car.
        """
        self.speed = kwargs["speed"]
        self.smooth_track = chaikin_corner_cutting(kwargs["track"])
        kwargs["track"] = self.smooth_track
        _, self.gas, self.brake = self.longtitudinal_controller.get_action(observation, info, *args, **kwargs)
        self.next_step_speed = self.speed + (self.gas - self.brake) * self.conf.DT
        self.wheel_pose = np.array(kwargs["wheel_pose"])  # front left, front right, rear left, rear right
        self.rear_wheel_pose = (self.wheel_pose[2] + self.wheel_pose[3]) / 2
        self.front_wheel_pose = (self.wheel_pose[0] + self.wheel_pose[1]) / 2
        self.wheel_base = np.linalg.norm(self.front_wheel_pose[:2] - self.rear_wheel_pose[:2])
        self.steer = self.pure_pursuit_lateral_control()
        return (self.steer, self.gas, self.brake)

    def _compute_look_ahead_point(self, look_ahead_offset):
        """
        Computes the look ahead point on the track based on the given look ahead offset.
        Parameters:
        - look_ahead_offset (int): The offset value to determine the look ahead point.
        Returns:
        - look_ahead_point (numpy.ndarray): The coordinates of the computed look ahead point on the track.
        """
        track_distances = np.linalg.norm(self.smooth_track - self.rear_wheel_pose[:2], axis=1)
        pose_point_idx = np.argmin(track_distances)
        look_ahead_point = self.smooth_track[(pose_point_idx + look_ahead_offset) % len(self.smooth_track)]
        self.look_ahead_distance = np.linalg.norm(look_ahead_point - self.rear_wheel_pose[:2])
        return look_ahead_point

    def _compute_look_ahead_offset(self):
        """
        Computes the look ahead offset based on the curvature of the longitudinal controller.
        Returns:
            int: The computed look ahead offset.
        """
        default = 10
        if self.longtitudinal_controller.curvature > 0.06:
            default -= 7
        elif self.longtitudinal_controller.curvature > 0.05:
            default -= 6
        elif self.longtitudinal_controller.curvature > 0.04:
            default -= 5
        elif self.longtitudinal_controller.curvature > 0.03:
            default -= 4
        elif self.longtitudinal_controller.curvature > 0.02:
            default -= 3
        return default

    def pure_pursuit_lateral_control(self):
        """
        Computes the steering angle for the pure pursuit lateral control algorithm.
        Returns:
            float: The computed steering angle.
        """
        look_ahead_offset = self._compute_look_ahead_offset()
        look_ahead_point = self._compute_look_ahead_point(look_ahead_offset)
        rear_wheel_heading_error = utils.compute_he(self.rear_wheel_pose, self.rear_wheel_pose[:2], look_ahead_point)
        steering = np.arctan2(2 * self.wheel_base * np.sin(rear_wheel_heading_error), self.look_ahead_distance)
        self.debug_states["look_ahead_offset"].append(look_ahead_offset)
        self.debug_states["look_ahead_point"].append(look_ahead_point)
        for key in self.longtitudinal_controller.debug_states:
            self.debug_states[key] = self.longtitudinal_controller.debug_states[key]
        return steering
