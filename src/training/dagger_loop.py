from __future__ import annotations

import warnings

from src.expert_drivers.pid_driver.pid_driver_controller import PidDriverController
from src.utils import utils

# Ignore all warnings
warnings.filterwarnings("ignore")
# isort:maintain_block
import os
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from tqdm.contrib.concurrent import process_map

import wandb
from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from utils import conf_utils, env_utils

conf = conf_utils.get_default_conf()


def teacher_action_probability(epoch):
    """
    Returns the probability of the teacher action being chosen.
    The probability is calculated as p^i, where p is the value of IMITATION_P and i is the epoch number.
    Parameters:
    - epoch (int): The current epoch number.
    Returns:
    - float: The probability of the teacher action being chosen.
    """
    p = conf.IMITATION_P_DECAY**epoch
    if p < conf.IMITATION_TEACHER_P_CUT_OFF:
        p = 0.0
    return p


def choose_action(student_action, teacher_action, epoch):
    """if random(0, 1) > beta = p^i then choose teacher action, else choose student action"""
    if np.random.random() < teacher_action_probability(epoch):
        return teacher_action
    return student_action


class DaggerLoop:

    def __init__(self, output_dir, student_model):
        self.counter = conf.IMITATION_START_SEED
        self.output_dir = output_dir
        self.student_model = student_model
        os.makedirs(output_dir, exist_ok=True)

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
        self.epoch = epoch

        # At the beginning, the model is bad so we don't even want to waste resource on evaluation.
        # Also the teacher was also driving so we don't want the bias.
        n_seeds_per_loop = conf.DAGGER_BEGINNING_ITERATION_PER_LOOP
        if (
            teacher_action_probability(epoch) == 0
        ):  # Later when the teacher doesn't drive any more, we want to evaluate the model
            n_seeds_per_loop = conf.DAGGER_END_ITERATION_PER_LOOP
        self.counter += n_seeds_per_loop
        self.rewards = process_map(
            self.dagger_iteration,
            list(range(self.counter, self.counter + n_seeds_per_loop)),
            max_workers=conf.GYM_MAX_WORKERS,
            desc="Dagger loop",
        )
        self.reward = np.mean(self.rewards)
        self._log_wandb(epoch)
        return self.reward

    def _log_wandb(self, epoch):
        if conf.WANDB_LOG:
            wandb.log({"dagger/reward": self.reward}, step=epoch)
            wandb.log({"dagger/teacher_action_proboability": teacher_action_probability(epoch)}, step=epoch)
            wandb.log({"dagger/#records": len(os.listdir(self.output_dir))}, step=epoch)

    def dagger_iteration(self, seed):
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
        self.student_model.eval()  # Set model to evaluation mode
        student_driver = ImitationDriverController(model=self.student_model)  # type: ignore
        teacher_driver = PidDriverController()
        env = env_utils.create_env(conf=conf)

        # Set up history
        history = defaultdict(list)

        # Initialize new scenario
        terminated = truncated = done = False
        observation, info = env.reset(seed=seed)
        seed_reward = step = 0

        # Start simulation
        while not (done or terminated or truncated):
            # Student gets only noisy observation
            student_action = student_driver.get_action(observation, info)
            student_driver.reset()
            # Teacher get all the information
            teacher_action = teacher_driver.get_action(
                observation,
                info,
                speed=env_utils.get_speed(env),
                wheels_omegas=env_utils.get_wheel_velocities(env),
                angular_velocity=env_utils.get_angular_velocity(env),  # type: ignore
                steering_joint_angle=env_utils.get_steering_joint_angle(env),  # type: ignore
                track=env_utils.extract_track(env),
                pose=env_utils.get_pose(env),
            )
            history["observation_history"].append(observation)
            observation, reward, terminated, truncated, info = env.step(
                choose_action(student_action, teacher_action, epoch=self.epoch)
            )

            # Record history
            history["action_history"].append(teacher_action)

            # Go to next step
            seed_reward += reward  # type: ignore

            # Increment step
            step += 1
            if step >= conf.TRAINING_MAX_TIME_STEPS:
                terminated = True
                break
        seed_reward = int(seed_reward)

        # Save record
        do_store_record = False
        if self.epoch < conf.IMITATION_STORE_ALL_RECORDS_EPOCH:
            do_store_record = True
        elif seed_reward < conf.IMITATION_STORE_REWARD_THRESHOLD:
            do_store_record = True

        if do_store_record:
            np.savez(
                os.path.join(self.output_dir, f"{seed}_{self.epoch}_{seed_reward}.npz"),
                seed=seed,
                seed_reward=seed_reward,
                terminated=terminated,
                truncated=truncated,
                done=done,
                **history,
                **teacher_driver.debug_states,
            )
        env.close()
        return seed_reward
