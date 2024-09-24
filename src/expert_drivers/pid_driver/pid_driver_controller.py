from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.expert_drivers.abstract_classes.abstract_controller import AbstractController
from src.expert_drivers.pid_driver import penalty
from src.expert_drivers.pid_driver.path_metrics_computer import PathMetricsComputer
from src.expert_drivers.pid_driver.pid import PIDController
from src.utils.conf_utils import extend_conf


class ComponentPidDriverController(AbstractController):
    def __init__(self, conf):
        """
        Initializes a PIDDriverController object.
        Args:
            conf: The configuration object containing the PID controller parameters.
        """

        super().__init__()
        self.conf = conf
        self.debug_states = defaultdict(list)
        self.lateral_cte_pid = PIDController(conf.LATERAL_PID_CTE_KP, conf.LATERAL_PID_CTE_KI, conf.LATERAL_PID_CTE_KD)
        self.lateral_he_pid = PIDController(conf.LATERAL_PID_HE_KP, conf.LATERAL_PID_HE_KI, conf.LATERAL_PID_HE_KD)
        self.longitudinal_gas_pid = PIDController(
            conf.LONGITUDINAL_PID_GAS_KP, conf.LONGITUDINAL_PID_GAS_KI, conf.LONGITUDINAL_PID_GAS_KD
        )
        self.longitudinal_brake_pid = PIDController(
            conf.LONGITUDINAL_PID_BRAKE_KP, conf.LONGITUDINAL_PID_BRAKE_KI, conf.LONGITUDINAL_PID_BRAKE_KD
        )
        self.path_metric_computer = None
        self.track = None
        self.pose = None
        self.speed = 0
        self.wheels_omegas_std = 0
        self.curvature = 0

    def reset(self):
        """
        Resets the PID driver controller.
        This method resets the PID driver controller by calling the reset method of the superclass,
        resetting all the PID controllers for lateral control, longitudinal gas control, and longitudinal brake control,
        and resetting the errors computer, curvatures computer, and debug states.
        Parameters:
            None
        Returns:
            None
        """
        super().reset()  # Correct way to call reset of superclass
        self.lateral_cte_pid.reset()
        self.lateral_he_pid.reset()
        self.longitudinal_gas_pid.reset()
        self.longitudinal_brake_pid.reset()
        self.path_metric_computer = None
        self.debug_states = defaultdict(list)
        self.curvature = 0

    def get_action(self, observation, info, *args, **kwargs):
        """
        Get the continuous action to be executed by the car.

        Args:
            observation (np.ndarray): The observation received from the environment.
            info (dict): Additional information about the environment.

        Returns:
            np.ndarray: The continuous action to be executed by the car.
        """
        self.track = kwargs["track"]
        self.pose = kwargs["pose"]
        self.speed = kwargs["speed"]
        self.wheels_omegas_std = np.std(kwargs["wheels_omegas"])

        if self.path_metric_computer is None:
            self.path_metric_computer = PathMetricsComputer(self.track, self.conf)
        self.cte, self.he, self.curvature = self.path_metric_computer.compute_metrics(self.pose)
        self.debug_states["curvature"].append(self.curvature)
        self.debug_states["cte"].append(self.cte)
        self.debug_states["he"].append(self.he)

        self.gas, self.brake = self.longitudinal_control()
        self.steer = self.lateral_control()
        return (self.steer, self.gas, self.brake)

    def longitudinal_control(self):
        """
        Compute the longitudinal control action for the car.

        Args:
            track (np.ndarray): The track to follow.
            pose (np.ndarray): The current pose of the car.
            speed (float): The current speed of the car.
        Returns:
            (float, float): The longitudinal control action.
        """
        self.desired_speed = self.compute_desired_speed()
        self.debug_states["desired_speed"].append(self.desired_speed)
        self.speed_error = self.desired_speed - self.speed
        self.debug_states["speed_error"].append(self.speed_error)

        self.gas, self.brake = 0.0, 0.0
        if self.speed_error >= 0:
            self.gas = self.compute_gas()
        else:
            self.brake = np.clip(self.longitudinal_brake_pid.update(-self.speed_error), 0, self.conf.MAX_BRAKE)
        return self.gas, self.brake

    def lateral_control(self):
        """
        Compute the lateral control action for the car.

        Args:
            track (np.ndarray): The track to follow.
            pose (np.ndarray): The current pose of the car.

        Returns:
            float: The lateral control action.
        """
        # Update PID controllers and return control
        self.cte_control = np.clip(self.lateral_cte_pid.update(self.cte), -1.0, 1.0)
        self.he_control = np.clip(self.lateral_he_pid.update(self.he), -1.0, 1.0)
        self.debug_states["cte_control"].append(self.cte_control)
        self.debug_states["he_control"].append(self.he_control)
        self.control = np.clip(self.cte_control + self.he_control, -1.0, 1.0)
        self.control = penalty.clamp_steering(self.control, self.gas)
        return self.control

    def compute_desired_speed(self):
        """
        Compute the desired speed based on the curvature of the track.

        Args:
            curvature (float): The curvature of the track.

        Returns:
            float: The desired speed.
        """
        # desired_speed = linear interpolation between min and max speed based on curvature
        self.desired_speed = (
            self.conf.MAX_SPEED - self.curvature * (self.conf.MAX_SPEED - self.conf.MIN_SPEED) / self.conf.MAX_CURVATURE
        )
        self.desired_speed = np.clip(self.desired_speed, self.conf.MIN_SPEED, self.conf.MAX_SPEED)
        self.desired_speed -= penalty.desired_speed(self.wheels_omegas_std, self.curvature)
        self.desired_speed = np.clip(self.desired_speed, self.conf.MIN_SPEED, self.conf.MAX_SPEED)
        self.desired_speed = penalty.clamp_speed(self.desired_speed, self.wheels_omegas_std)
        return self.desired_speed

    def compute_gas(self):
        """
        Compute the gas action based on the speed error.

        Args:
            speed_error (float): The speed error.

        Returns:
            float: The gas action.
        """
        # Compute gas based on speed error
        self.gas = np.clip(self.longitudinal_gas_pid.update(self.speed_error), 0, self.conf.MAX_GAS)
        self.gas = self.gas - penalty.gas(self.wheels_omegas_std)
        self.gas = np.clip(self.gas, 0, self.conf.MAX_GAS)
        return self.gas


def _new_pid_driver(conf, new_conf_path):
    return ComponentPidDriverController(conf=extend_conf(conf, [new_conf_path]))


class PidDriverController(AbstractController):
    def __init__(self, conf=None):  # type: ignore
        """
        Initializes a PIDDriverController object.
        Args:
            conf: The configuration object containing the PID controller parameters.
        """
        super().__init__()
        if conf is None:
            from src.utils import conf_utils

            conf = conf_utils.get_conf(controller="pid", print_out=False)
        self.conf = conf
        self.debug_states = defaultdict(list)
        self.path_metric_computer = None
        self.normal_driver = _new_pid_driver(conf, conf.PID_SUB_CONFIGS[0][1])
        self.corner_1_driver = _new_pid_driver(conf, conf.PID_SUB_CONFIGS[1][1])
        self.corner_2_driver = _new_pid_driver(conf, conf.PID_SUB_CONFIGS[2][1])
        self.corner_3_driver = _new_pid_driver(conf, conf.PID_SUB_CONFIGS[3][1])

    def reset(self):
        """
        Resets the PID driver controller.
        This method resets the PID driver controller by calling the reset method of the superclass,
        resetting all the PID controllers for lateral control, longitudinal gas control, and longitudinal brake control,
        and resetting the errors computer, curvatures computer, and debug states.
        Parameters:
            None
        Returns:
            None
        """
        super().reset()  # Correct way to call reset of superclass
        self.normal_driver.reset()
        self.corner_1_driver.reset()
        self.corner_2_driver.reset()
        self.corner_3_driver.reset()
        self.path_metric_computer = None
        self.track = None
        self.pose = None
        self.debug_states = defaultdict(list)

    def up_to_date_debug_states(self, chosen_driver_debug_states):
        """
        Update the debug states with the latest values from the chosen driver's debug states.
        """
        for key, value in chosen_driver_debug_states.items():
            if value:  # Make sure there is at least one entry in the list
                self.debug_states[key].append(value[-1])  # Append the last entry of each property

    def get_action(self, observation, info, *args, **kwargs):
        """
        Get the continuous action to be executed by the car.

        Args:
            observation (np.ndarray): The observation received from the environment.
            info (dict): Additional information about the environment.

        Returns:
            np.ndarray: The continuous action to be executed by the car.
        """
        self.track = kwargs["track"]
        self.pose = kwargs["pose"]

        if self.path_metric_computer is None:
            self.path_metric_computer = PathMetricsComputer(self.track, self.conf)
        _, _, self.curvature = self.path_metric_computer.compute_metrics(self.pose)

        if self.curvature < self.conf.PID_SUB_CONFIGS[0][0]:
            action = self.normal_driver.get_action(observation, info, *args, **kwargs)
            self.up_to_date_debug_states(self.normal_driver.debug_states)
            self.debug_states["driver"].append(0)
        elif self.curvature < self.conf.PID_SUB_CONFIGS[1][0]:
            action = self.corner_1_driver.get_action(observation, info, *args, **kwargs)
            self.up_to_date_debug_states(self.corner_1_driver.debug_states)
            self.debug_states["driver"].append(1)
        elif self.curvature < self.conf.PID_SUB_CONFIGS[2][0]:
            action = self.corner_2_driver.get_action(observation, info, *args, **kwargs)
            self.up_to_date_debug_states(self.corner_2_driver.debug_states)
            self.debug_states["driver"].append(2)
        else:
            action = self.corner_3_driver.get_action(observation, info, *args, **kwargs)
            self.up_to_date_debug_states(self.corner_3_driver.debug_states)
            self.debug_states["driver"].append(3)
        self.debug_states["decision_curvature"].append(self.curvature)
        return action
