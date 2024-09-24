from __future__ import annotations

import os
from collections import defaultdict, namedtuple

import numpy as np
import torch
from tqdm.contrib.concurrent import process_map

from src.expert_drivers.abstract_classes.abstract_controller import AbstractController
from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from src.utils import conf_utils, utils
from src.utils.env_utils import (
    create_env,
    extract_track,
    get_angular_velocity,
    get_pose,
    get_speed,
    get_steering_joint_angle,
    get_wheel_poses,
    get_wheel_velocities,
    is_car_off_track,
)

conf = conf_utils.get_default_conf()


def teacher_action_probability(epoch: int):
    """
    Returns the probability of the teacher self.action being chosen.
    The probability is calculated as p^i, where p is the value of IMITATION_P and i is the epoch number.
    Parameters:
    - epoch (int): The current epoch number.
    Returns:
    - float: The probability of the teacher self.action being chosen.
    """
    p = conf.IMITATION_P_DECAY**epoch
    if p < conf.IMITATION_TEACHER_P_CUT_OFF:
        p = 0.0
    return p


def choose_action(student_action, teacher_action, epoch):
    """
    Choose an action based on the probability determined by the current epoch.
    Parameters:
    student_action (any): The action proposed by the student.
    teacher_action (any): The action proposed by the teacher.
    epoch (int): The current epoch which influences the probability of choosing the teacher's action.
    Returns:
    any: The chosen action, either from the student or the teacher based on the probability.
    """
    if np.random.random() < teacher_action_probability(epoch=epoch):
        return teacher_action
    return student_action


def randomly_discard_low_curvature(record: dict):
    """
    Randomly discards the last element of each key in the record if the curvature
    of the last element is below a specified minimum threshold and a random probability
    condition is met.
    Args:
        record (dict): A dictionary containing various keys, each associated with a list of values.
                       The key "curvature" should be present in the dictionary and its associated
                       list should contain numerical values representing curvature.
    Modifies:
        The input dictionary `record` by potentially removing the last element from each list
        if the conditions are met.
    """
    if (
        record["curvature"][-1] < conf.IMITATION_MIN_CURVATURE_DISCARD_THRESHOLD
        and np.random.random() < conf.IMITATION_MIN_CURVATURE_DISCARD_PROB
    ):
        for key in record:
            record[key] = record[key][:-1]
        return 1
    return 0

SimulationResult = namedtuple("SimulationResult", ["seed", "reward", "done", "terminated", "truncated", "off_track"])

class Simulator:
    def __init__(
        self,
        output_dir: str,
        max_steps: int,
        student_controller: AbstractController,
        teacher_controller: AbstractController | None = None,
        dagger_mode: bool = False,
        start_seed_counter: int = conf.IMITATION_TRAINING_START_SEED,
        do_early_break: bool = True
    ) -> None:
        """
        Initializes the simulator with the given parameters.
        Args:
            output_dir (str): The directory where output files will be saved.
            max_steps (int): The maximum number of steps for the simulation.
            student_controller (AbstractController): The controller used by the student.
            teacher_controller (AbstractController | None, optional): The controller used by the teacher.
                Defaults to None. If None and dagger_mode is True, a ValueError is raised.
            dagger_mode (bool, optional): If True, the simulator operates in DAgger mode. Defaults to False.
            start_seed_counter (int, optional): The starting value for the seed counter. Defaults to conf.IMITATION_TRAINING_START_SEED.
        Raises:
            ValueError: If dagger_mode is True and teacher_controller is None.
        """
        if dagger_mode and teacher_controller is None:
            raise ValueError("Teacher controller must be provided in DAgger mode.")
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.student_controller = student_controller
        self.teacher_controller = teacher_controller if teacher_controller is not None else student_controller
        self.dagger_mode = dagger_mode
        self.epoch = -1
        self.counter = start_seed_counter
        self.do_early_breaking = do_early_break

    def _get_seeds(self):
        """
        Generate a list of seed values for the simulation.
        If the simulator is in DAgger mode, the number of seeds generated depends on the current epoch.
        At the beginning of training, a larger number of seeds is used to avoid wasting resources on evaluation
        when the model is still poor. As training progresses and the teacher's influence decreases, the number
        of seeds is reduced.
        Returns:
            list: A list of seed values for the simulation.
        """
        # At the beginning, the model is bad so we don't even want to waste resource on evaluation.
        # Also the teacher was also driving so we don't want the bias.
        if self.dagger_mode:
            n_seeds = conf.DAGGER_BEGINNING_ITERATION_PER_LOOP
            if teacher_action_probability(self.epoch) == 0:
                n_seeds = conf.DAGGER_END_ITERATION_PER_LOOP
            self.counter += n_seeds
            return list(range(self.counter, self.counter + n_seeds))
        return conf.RECORD_SEEDS

    def simulate(self, epoch: int):
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

        # Simulate and return the average reward
        results = process_map(
            self._simulate_one_seed, self._get_seeds(), max_workers=conf.GYM_MAX_WORKERS, desc=f"Simulation Loop"
        )
        self.rewards = [result.reward for result in results]
        self.off_track = [result.off_track for result in results]
        self.reward = np.mean(self.rewards)
        return results, self.reward

    def _get_action(self, controller, info):
        """
        Get the self.action from the controller.
        Args:
            controller (AbstractController): The controller to get the self.action from.
            observation (np.ndarray): The observation from the environment.
            info (dict): The information from the environment.
        Returns:
            np.ndarray: The self.action to take in the environment.
        """
        if isinstance(controller, ImitationDriverController):
            self.student_controller.model.eval()  # type: ignore
            with torch.no_grad():
                return self.student_controller.get_action(self.record["observation"][-1], info)
        return controller.get_action(
            self.record["observation"][-1],
            info,
            track=self.track,
            pose=self.record["pose"][-1],
            speed=self.record["speed"][-1],
            wheels_omegas=self.record["wheels_omegas"][-1],
            steering_joint_angle=self.record["steering_joint_angle"][-1],
            angular_velocity=self.record["angular_velocity"][-1],
            wheel_pose=self.record["wheel_poses"][-1],
        )

    def _record_pre_simulation(self):
        """
        Records the pre-simulation state of the car into the record dictionary.

        This method appends the current state of various car parameters to the
        corresponding lists in the record dictionary.
        Returns:
            None
        """
        self.record["speed"].append(get_speed(self.env))
        self.record["pose"].append(get_pose(self.env))
        self.record["wheels_omegas"].append(get_wheel_velocities(self.env))
        self.record["steering_joint_angle"].append(get_steering_joint_angle(self.env))
        self.record["angular_velocity"].append(get_angular_velocity(self.env))
        self.record["off_track"].append(is_car_off_track(self.env))
        self.record["wheel_poses"].append(get_wheel_poses(self.env))
        self.record["observation"].append(self.observation)

    def _record_post_simulation(self):
        """
        Records the post-simulation data into the record dictionary.
        Returns:
            None
        """
        self.record["reward"].append(self.reward)
        self.record["realised_action"].append(self.realised_action)
        self.record["student_action"].append(self.student_action)
        self.record["teacher_action"].append(self.teacher_action)
        if self.student_controller is not self.teacher_controller:  # Avoid double counting
            utils.concatenate_debug_states(self.student_controller.debug_states, self.record)
        utils.concatenate_debug_states(self.teacher_controller.debug_states, self.record)
        self.student_controller.reset()
        self.teacher_controller.reset()

    def _store_record(self, seed):
        """
        Stores the simulation record to a file if certain conditions are met.
        Parameters:
        seed (int): The seed value used for the simulation.
        The function checks if the record should be stored based on the `dagger_mode` flag and
        certain conditions related to the epoch and seed reward. If the conditions are met,
        it asserts that the length of each record matches the number of steps taken in the
        simulation. Finally, it saves the record and other relevant information to a .npz file
        in the specified output directory.
        Raises:
        AssertionError: If the length of any record does not match the number of steps taken.
        """
        do_store_record = not self.dagger_mode
        if self.dagger_mode:
            if self.epoch < conf.IMITATION_STORE_ALL_RECORDS_EPOCH:
                do_store_record = True
            elif self.seed_reward < conf.IMITATION_STORE_REWARD_THRESHOLD:
                do_store_record = True

        if do_store_record:
            for key in self.record:
                assert len(self.record[key]) == (
                    self.step - self.discarded
                ), f"Key {key} has length {len(self.record[key])} but simulation ran {self.step} steps and discarded {self.discarded}."
            np.savez(
                os.path.join(self.output_dir, f"{seed}_{self.seed_reward}.npz"),
                seed=seed,
                seed_reward=self.seed_reward,
                terminated=self.terminated,
                truncated=self.truncated,
                done=self.done,
                track=self.track,
                **self.record,
            )

    def _choose_action(self):
        """
        Chooses the action to be taken based on the current mode.
        If `dagger_mode` is enabled, it selects an action using the `choose_action` function,
        which considers both the student and teacher actions along with the current epoch.
        Otherwise, it defaults to the student action.
        Returns:
            The chosen action.
        """
        if self.dagger_mode:
            self.action = choose_action(student_action=self.student_action, teacher_action=self.teacher_action, epoch=self.epoch)  # type: ignore
        else:
            self.action = self.student_action  # type: ignore
        return self.action

    def _do_early_breaking(self) -> bool:
        """
        Determines whether to perform an early termination of the simulation.
        Returns:
            bool: True if any of the early termination conditions are met, otherwise False.
        """
        if self.step >= self.max_steps:
            return True

        if not self._do_early_breaking:
            return False

        if self.steps_without_rewards >= conf.IMITATION_EARLY_BREAK_NO_REWARD_STEPS:
            return True

        if (
            np.abs(self.record["cte"][-1]) >= conf.IMITATION_EARLY_BREAK_MAX_CTE
            and np.abs(self.record["he"][-1]) >= conf.IMITATION_EARLY_BREAK_MAX_HE
        ):
            return True

        return False

    def _simulate_one_seed(self, seed: int):
        """
        Simulates a seed in a racecar environment.
        Args:
            seed (int): The seed value for the simulation.
        """
        utils.set_deterministic(seed=seed)

        self.env = create_env(conf=conf)

        # Set up record
        self.record = defaultdict(list)

        # Initialize new scenario
        self.terminated = self.truncated = self.done = False
        self.observation, info = self.env.reset(seed=seed)
        self.seed_reward = self.step = 0
        self.discarded = 0
        self.track = extract_track(self.env)

        # Early breaking variables
        self.steps_without_rewards = 0

        # Start simulation
        while not (self.done or self.terminated or self.truncated):
            # Query controllers
            self.env.render()
            self._record_pre_simulation()
            self.student_action = self._get_action(self.student_controller, info)
            self.teacher_action = self._get_action(self.teacher_controller, info)
            self.realised_action = self._choose_action()

            # Simulate
            next_observation, self.reward, self.terminated, self.truncated, info = self.env.step(self.action)
            self._record_post_simulation()
            self.observation = next_observation
            self.seed_reward += self.reward  # type: ignore
            if self.reward <= 0.0:  # type: ignore
                self.steps_without_rewards += 1
            else:
                self.steps_without_rewards = 0

            # Increment self.step
            self.step += 1
            if self._do_early_breaking():
                self.terminated = True
                break

            # Discard state randomly
            if self.dagger_mode:
                self.discarded += randomly_discard_low_curvature(self.record)

        # Store record
        self.seed_reward = int(self.seed_reward)
        self._store_record(seed=seed)

        # Finish
        return SimulationResult(seed=seed, reward=self.seed_reward, done=self.done, terminated=self.terminated, truncated=self.truncated, off_track=np.mean(self.record["off_track"]))
