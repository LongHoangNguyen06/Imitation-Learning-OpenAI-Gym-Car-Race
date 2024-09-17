from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import pygame
from dynaconf import Dynaconf

from src.expert_drivers.abstract_classes.abstract_controller import AbstractController
from src.expert_drivers.human_driver.human_driver_controller import HumanDriverController
from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from src.expert_drivers.pid_driver.pid_driver_controller import PidDriverController
from src.expert_drivers.pure_pursuit_driver.pure_pursuit_controller import PurePursuitController
from src.expert_drivers.stanley_driver.stanley_controller import StanleyController


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


def get_conf(args: argparse.Namespace, print_out=True) -> Dynaconf:
    """
    Retrieves the configuration settings based on the given arguments.
    Args:
        args (Namespace): The command line arguments.
    Returns:
        Dynaconf: The configuration object.
    Raises:
        ValueError: If the mode or controller is invalid.
    """
    settings_files = ["src/conf/default_conf.py"]
    if args.mode == "demo":
        settings_files.append("src/conf/mode_conf/demo_conf.py")
    elif args.mode == "test":
        settings_files.append("src/conf/mode_conf/test_conf.py")
    elif args.mode == "benchmark":
        settings_files.append("src/conf/mode_conf/benchmark_conf.py")
    elif args.mode == "debug":
        settings_files.append("src/conf/mode_conf/debug_conf.py")
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    if args.controller == "human":
        settings_files.append("src/conf/controller_conf/human_conf.py")
    elif args.controller == "constant":
        settings_files.append("src/conf/controller_conf/constant_conf.py")
    elif args.controller == "pid":
        settings_files.append("src/conf/controller_conf/pid_conf.py")
    elif args.controller == "pure_pursuit":
        settings_files.append("src/conf/controller_conf/pure_pursuit_conf.py")
    elif args.controller == "stanley":
        settings_files.append("src/conf/controller_conf/stanley_conf.py")
    elif args.controller == "imitation":
        settings_files.append("src/conf/controller_conf/imitation_conf.py")
    else:
        raise ValueError(f"Invalid controller: {args.controller}")

    conf = Dynaconf(envvar_prefix="DYNACONF", settings_files=settings_files, lowercase_read=False)
    if print_out:
        print("#" * 100)
        print("# Configuration")
        print("#" * 100)
        for key, value in conf.to_dict().items():
            print(f"{key}: {value}")
    return conf


def get_controller(args: argparse.Namespace, conf: Dynaconf) -> AbstractController:
    """
    Returns a controller based on the given arguments and configuration.
    Parameters:
    - args: The command line arguments.
    - conf: The configuration settings.
    Returns:
    - A controller object based on the specified controller type.
    Raises:
    - ValueError: If an invalid controller type is specified.
    """
    if args.controller == "human":
        return HumanDriverController(conf=conf)
    elif args.controller == "pid":
        return PidDriverController(conf=conf)
    elif args.controller == "pure_pursuit":
        return PurePursuitController(conf=conf)
    elif args.controller == "stanley":
        return StanleyController(conf=conf)
    elif args.controller == "imitation":
        return ImitationDriverController(conf=conf)
    raise ValueError(f"Invalid controller: {args.controller}")


def create_env(conf: Dynaconf) -> gym.Env:
    """
    Creates a CarRacing environment based on the given configuration.
    Parameters:
    - conf: The configuration settings.
    Returns:
    - A CarRacing environment with the specified settings.
    """
    return gym.make(
        "CarRacing-v2",
        domain_randomize=conf.DOMAIN_RANDOMIZE,
        render_mode=conf.RENDER_MODE,
        lap_complete_percent=conf.LAP_COMPLETE_PERCENT,
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


def did_user_quit_or_skip():
    """
    Checks if the user has quit or skipped the race.
    Returns:
        user_quit (bool): True if the user has quit the race by pressing the ESC key, False otherwise.
        user_skipped (bool): True if the user has skipped the race by pressing the SPACE key, False otherwise.
    """

    user_quit = False
    user_skipped = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("Exiting the demo...")
                user_quit = True
            elif event.key == pygame.K_SPACE:
                print("Skipping the seed...")
                user_skipped = True
    return user_quit, user_skipped


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parses the command line arguments.
    Parameters:
    - argv: The list of command line arguments.
    Returns:
    - A Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        type=str,
        default="human",
        choices=["human", "constant", "pid", "pure_pursuit", "stanley", "imitation"],
        help="Controller of agent.",
    )
    parser.add_argument(
        "--mode", type=str, default="demo", choices=["demo", "test", "benchmark", "debug"], help="Mode of the program."
    )
    return parser.parse_args(argv)
