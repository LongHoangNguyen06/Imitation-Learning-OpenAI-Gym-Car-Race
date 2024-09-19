from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from src.utils import conf_utils
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


class Simulator:
    def __init__(self, output_dir: str, max_steps: int, do_record: bool, controller) -> None:
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.do_record = do_record
        self.controller = controller

    def simulate(self, seed):
        """
        Simulates a seed in a racecar environment.
        Args:
            seed (int): The seed value for the simulation.
        """
        env = create_env(conf=conf)

        # Set up history
        history = defaultdict(list)

        # Initialize new scenario
        terminated = truncated = done = False
        observation, info = env.reset(seed=seed)
        seed_reward = step = 0
        track = extract_track(env)

        # Start simulation
        while not (done or terminated or truncated):
            # Record history
            pose = get_pose(env)
            speed = get_speed(env)
            wheels_omegas = get_wheel_velocities(env)
            steering_joint_angle = get_steering_joint_angle(env)
            angular_velocity = get_angular_velocity(env)
            off_track = is_car_off_track(env)
            wheel_poses = get_wheel_poses(env)
            history["speed_history"].append(speed)
            history["pose_history"].append(pose)
            history["wheels_omegas_history"].append(wheels_omegas)
            history["steering_joint_angle_history"].append(steering_joint_angle)
            history["angular_velocity_history"].append(angular_velocity)
            history["off_track_history"].append(off_track)
            history["wheel_poses_history"].append(wheel_poses)
            history["observation_history"].append(observation)

            # Simulation
            env.render()
            if isinstance(self.controller, ImitationDriverController):
                action = self.controller.get_action(observation, info)
                self.controller.reset()
            else:
                action = self.controller.get_action(
                    observation,
                    info,
                    track=track,
                    pose=pose,
                    speed=speed,
                    wheels_omegas=wheels_omegas,
                    steering_joint_angle=steering_joint_angle,
                    angular_velocity=angular_velocity,
                    wheel_pose=wheel_poses,
                )
            observation, reward, terminated, truncated, info = env.step(action)

            # Record history
            history["action_history"].append(action)
            history["reward_history"].append(reward)

            # Go to next step
            seed_reward += reward  # type: ignore

            # Increment step
            step += 1
            if step >= self.max_steps:
                terminated = True
                break
        seed_reward = int(seed_reward)
        # Save record
        if self.do_record:
            np.savez(
                os.path.join(self.output_dir, f"{seed}_{seed_reward}.npz"),
                seed=seed,
                seed_reward=seed_reward,
                terminated=terminated,
                truncated=truncated,
                done=done,
                track=track,
                **history,
                **self.controller.debug_states,
            )
        return seed, seed_reward, done, terminated, truncated, np.mean(history["off_track_history"])
