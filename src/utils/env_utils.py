from __future__ import annotations

import gymnasium as gym
import numpy as np


def extract_track(env: gym.ENV) -> np.ndarray:  # type: ignore
    """
    Extracts the track from the environment.
    Args:
        env (gym.Env): The environment from which to extract the track.
    Returns:
        np.ndarray: The track extracted from the environment.
    """
    track = []
    for seg in env.unwrapped.track:
        x = seg[2]
        y = seg[3]
        track.append([x, y])
    return np.array(track)


def create_env(conf) -> gym.Env:
    """
    Creates a CarRacing environment based on the given configuration.
    Parameters:
    - conf: The configuration settings.
    Returns:
    - A CarRacing environment with the specified settings.
    """
    return gym.make(
        "CarRacing-v2",
        domain_randomize=False,
        render_mode=conf.RENDER_MODE,
        continuous=True,
    )


def get_pose(env: gym.Env) -> np.ndarray:
    """
    Returns the position of the racecar in the environment.
    Parameters:
    - env: The environment in which the racecar is located.
    Returns:
    - The position of the racecar as a numpy array.
    """
    return np.array(
        [
            env.unwrapped.car.hull.position[0],  # type: ignore
            env.unwrapped.car.hull.position[1],  # type: ignore
            env.unwrapped.car.hull.angle + np.pi / 2,  # type: ignore
        ]
    )


def get_speed(env: gym.Env) -> float:
    """
    Returns the speed of the racecar in the environment.
    Parameters:
    - env: The environment in which the racecar is located.
    Returns:
    - The speed of the racecar.
    """
    return np.linalg.norm(env.unwrapped.car.hull.linearVelocity)  # type: ignore


def get_wheel_velocities(env: gym.Env) -> np.ndarray:
    """
    Returns the angular velocities of the wheels of the racecar in the environment.
    Parameters:
    - env (gym.Env): The environment containing the racecar.
    Returns:
    - np.ndarray: An array containing the angular velocities of the racecar's wheels.
    """

    return np.array(
        [
            env.unwrapped.car.wheels[0].omega,  # type: ignore
            env.unwrapped.car.wheels[1].omega,  # type: ignore
            env.unwrapped.car.wheels[2].omega,  # type: ignore
            env.unwrapped.car.wheels[3].omega,  # type: ignore
        ]
    )


def is_car_off_track(env: gym.Env) -> bool:
    """
    Check if the car is off the track in the given environment.
    Parameters:
    - env (gym.Env): The environment object.
    Returns:
    - bool: True if the car is off the track, False otherwise.
    """

    return all(len(w.tiles) == 0 for w in env.unwrapped.car.wheels)  # type: ignore


def get_wheel_poses(env):
    """
    Returns the positions and angles (poses) of the car's wheels.

    :param env: The CarRacing environment.
    :return: A list of tuples containing the position (x, y) and angle of each wheel.
    """
    wheel_poses = []

    # Assuming car has wheels stored in a list called 'wheels'
    for wheel in env.unwrapped.car.wheels:
        position = wheel.position
        wheel_poses.append((position.x, position.y, wheel.angle + np.pi / 2))
    return np.array(wheel_poses)

def get_angular_velocity(env: gym.Env) -> float:
    """
    Returns the angular velocity of the racecar in the environment.
    Parameters:
    - env: The environment in which the racecar is located.
    Returns:
    - The angular velocity of the racecar.
    """
    return env.unwrapped.car.hull.angularVelocity  # type: ignore

def get_steering_joint_angle(env: gym.Env) -> float:
    """
    Returns the steering joint angle of the racecar in the environment.
    Parameters:
    - env: The environment in which the racecar is located.
    Returns:
    - The steering
    """
    return env.unwrapped.car.wheels[0].joint.angle  # type: ignore
