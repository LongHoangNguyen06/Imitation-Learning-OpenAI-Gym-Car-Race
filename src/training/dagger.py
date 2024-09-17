from __future__ import annotations

import warnings

from src.expert_drivers.pid_driver.pid_driver_controller import PidDriverController

# Ignore all warnings
warnings.filterwarnings("ignore")
# isort:maintain_block
import os
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
from dynaconf import Dynaconf
from tqdm.contrib.concurrent import process_map

from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from src.utils import env_utils

conf = Dynaconf(settings_files=["src/conf/default_conf.py"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def teacher_action_probability(epoch):
    """
    Returns the probability of the teacher action being chosen.
    The probability is calculated as p^i, where p is the value of DAGGER_P and i is the epoch number.
    Parameters:
    - epoch (int): The current epoch number.
    Returns:
    - float: The probability of the teacher action being chosen.
    """
    p = conf.DAGGER_P_DECAY**epoch
    if p < conf.DAGGER_TEACHER_P_CUT_OFF:
        p = 0.0
    return p


def choose_action(student_action, teacher_action, epoch):
    """if random(0, 1) > beta = p^i then choose teacher action, else choose student action"""
    if np.random.random() < teacher_action_probability(epoch):
        return teacher_action
    return student_action


def keep_record_probability(epoch):
    """
    Calculates the probability of keeping a record for a given epoch.
    Parameters:
    epoch (int): The current epoch.
    Returns:
    float: The probability of keeping a record for the given epoch.
    """
    return max(conf.DAGGER_KEEP_RECORD_P_DECAY**epoch, conf.DAGGER_KEEP_RECORD_MIN_P)


def keep_record(epoch):
    """
    Determines whether to keep a record for the given epoch.
    Parameters:
    epoch (int): The current epoch.
    Returns:
    bool: True if a record should be kept, False otherwise.
    """
    return np.random.random() < keep_record_probability(epoch)


def dagger_iteration(dagger_input):
    """
    Executes the dagger loop for training a student model using imitation learning.
    Args:
        seed (int): The seed for the environment.
        student_model: The student model to be trained.
        epoch (int): The current epoch of training.
        record_path (str): The path to save the training records.
    Returns:
        None
    """
    seed, student_model, epoch, record_path = dagger_input
    student_model.eval()  # Set model to evaluation mode
    student_driver = ImitationDriverController(conf=conf, model=student_model)  # type: ignore
    teacher_driver = PidDriverController()
    env = env_utils.create_env(conf=conf)

    # Set up history
    history = defaultdict(list)

    # Initialize new scenario
    env = env_utils.create_env(conf=conf)
    terminated = truncated = done = False
    observation, info = env.reset(seed=seed)
    seed_reward = step = 0
    track = env_utils.extract_track(env)

    # Start simulation
    while not (done or terminated or truncated):
        # Record history
        pose = env_utils.get_pose(env)
        speed = env_utils.get_speed(env)
        wheels_omegas = env_utils.get_wheel_velocities(env)
        steering_joint_angle = env.unwrapped.car.wheels[0].joint.angle  # type: ignore
        angular_velocity = env.unwrapped.car.hull.angularVelocity  # type: ignore
        history["speed_history"].append(speed)
        history["pose_history"].append(pose)
        history["wheels_omegas_history"].append(wheels_omegas)
        history["steering_joint_angle_history"].append(steering_joint_angle)
        history["angular_velocity_history"].append(angular_velocity)
        history["observation_history"].append(observation)
        track = env_utils.extract_track(env)

        # Simulation
        student_action = student_driver.get_action(
            observation,
            info,
            speed=speed,
            wheels_omegas=wheels_omegas,
            angular_velocity=angular_velocity,  # type: ignore
            steering_joint_angle=steering_joint_angle,  # type: ignore
        )
        teacher_action = teacher_driver.get_action(
            observation,
            info,
            speed=speed,
            wheels_omegas=wheels_omegas,
            angular_velocity=angular_velocity,  # type: ignore
            steering_joint_angle=steering_joint_angle,  # type: ignore
            track=track,
            pose=pose,
        )
        observation, reward, terminated, truncated, info = env.step(
            choose_action(student_action, teacher_action, epoch=epoch)
        )

        # Record history
        history["action_history"].append(teacher_action)

        # Go to next step
        seed_reward += reward  # type: ignore

        # Increment step
        step += 1
        if step >= conf.MAX_TIME_STEPS:
            terminated = True
            break
    seed_reward = int(seed_reward)

    # Save record
    if keep_record(epoch):
        np.savez(
            os.path.join(record_path, f"{seed}_{epoch}_{seed_reward}.npz"),
            seed=seed,
            seed_reward=seed_reward,
            terminated=terminated,
            truncated=truncated,
            done=done,
            track=track,
            **history,
            **teacher_driver.debug_states,
        )
    env.close()
    return seed_reward


class DaggerLoop:

    def __init__(self, dagger_output_dir, student_model):
        self.counter = conf.DAGGER_START_SEED
        self.dagger_loop_output_dir = dagger_output_dir
        self.student_model = student_model
        os.makedirs(dagger_output_dir, exist_ok=True)

    def dagger_loop(self, epoch):
        """
        Perform the Dagger loop for imitation learning.
        Args:
            student_model (object): The student model used for imitation learning.
            epoch (int): The number of epochs to train the student model.
            record_path (str): The path to record the training data.
            start_seed (int): The starting seed for generating random numbers.
            num_iterations (int, optional): The number of iterations to perform in the Dagger loop. Defaults to 10.
        Returns:
            list: A list of results from each iteration of the Dagger loop.
        """
        self.counter += conf.DAGGER_ITERATION_PER_LOOP
        return process_map(
            dagger_iteration,
            [
                (seed, self.student_model, epoch, self.dagger_loop_output_dir)
                for seed in range(self.counter, self.counter + conf.DAGGER_ITERATION_PER_LOOP)
            ],
            max_workers=conf.GYM_MAX_WORKERS,
            desc="Dagger loop",
        )
