from __future__ import annotations

from collections import defaultdict

import numpy as np
from dynaconf import Dynaconf
from numba import njit

from src.expert_drivers.abstract_classes.abstract_controller import AbstractController
from src.expert_drivers.pid_driver.path_metrics_computer import PathMetricsComputer
from src.expert_drivers.pid_driver.pid_driver_controller import PidDriverController
from src.expert_drivers.pure_pursuit_driver.pure_pursuit_controller import chaikin_corner_cutting
from src.utils import utils


class StanleyController(AbstractController):
    def __init__(self, conf: Dynaconf):
        super().__init__()
        self.conf = conf
        self.longtitudinal_controller = PidDriverController(conf)
        self.debug_states = defaultdict(list)
        self.path_metric_computer = None

    def reset(self):
        """
        Resets the controller.
        """

        super().reset()
        self.longtitudinal_controller.reset()
        self.debug_states = defaultdict(list)
        self.path_metric_computer = None

    def get_continuous_action(self, observation, info, *args, **kwargs):
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
        self.pose = kwargs["pose"]
        if self.path_metric_computer is None:
            self.path_metric_computer = PathMetricsComputer(self.smooth_track, self.conf)
        self.cte, self.he, self.curvature = self.path_metric_computer.compute_metrics(self.pose)
        kwargs["track"] = self.smooth_track
        _, self.gas, self.brake = self.longtitudinal_controller.get_action(observation, info, *args, **kwargs)
        self.wheel_pose = np.array(kwargs["wheel_pose"])  # front left, front right, rear left, rear right
        self.rear_wheel_pose = (self.wheel_pose[2] + self.wheel_pose[3]) / 2
        self.front_wheel_pose = (self.wheel_pose[0] + self.wheel_pose[1]) / 2
        self.wheel_base = np.linalg.norm(self.front_wheel_pose[:2] - self.rear_wheel_pose[:2])
        self.steer = self.stanley_lateral_control()
        return (self.steer, self.gas, self.brake)

    def stanley_lateral_control(self):
        """
        Computes the steering angle .
        Returns:
            float: The computed steering angle.
        """
        speed = self.speed
        scaled_cte = self.conf.CTE_COEFFICIENT * self.cte
        calibrated_he = np.arctan2(self.conf.HE_COEFFICIENT * self.he, speed + self.conf.DAMPING_FACTOR)
        steering = scaled_cte + calibrated_he
        return np.clip(
            steering,
            -1.0,
            1.0,
        )
