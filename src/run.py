from __future__ import annotations

# isort:maintain_block
import os

from src.utils import env_utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort:skip
# isort:end_maintain_block

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from tqdm.contrib.concurrent import process_map

from src.utils.env_utils import (
    create_env,
    did_user_quit_or_skip,
    extract_track,
    get_conf,
    get_controller,
    get_pose,
    get_speed,
    get_wheel_poses,
    get_wheel_velocities,
    is_car_off_track,
)
from src.utils.io_utils import get_next_dataset_name, get_next_record_name, get_next_trial_name

REWARD_DIM = 1


def simulate(simulation_input):
    """
    Simulates a seed in a racecar environment.
    Args:
        seed (int): The seed value for the simulation.
        args: The arguments for the simulation.
        conf: The configuration for the simulation.
        dataset_path (str): The path to the dataset.
    Returns:
        tuple: A tuple containing the seed reward, done flag, terminated flag, and truncated flag.
    """
    seed, args, dataset_path, trial_id, id, print_out = simulation_input
    conf = get_conf(args, print_out=False)
    controller = get_controller(args, conf)
    env = create_env(conf=conf)

    # Set up history
    history = defaultdict(list)

    # Initialize new scenario
    terminated = truncated = done = user_skipped = user_quit = False
    observation, info = env.reset(seed=seed)
    seed_reward = step = 0
    track = extract_track(env)

    # Start simulation
    while not (done or terminated or truncated or user_quit or user_skipped):
        # Record history
        pose = get_pose(env)
        speed = get_speed(env)
        wheels_omegas = get_wheel_velocities(env)
        steering_joint_angle = env.unwrapped.car.wheels[0].joint.angle  # type: ignore
        angular_velocity = env.unwrapped.car.hull.angularVelocity  # type: ignore
        history["speed_history"].append(speed)
        history["pose_history"].append(pose)
        history["wheels_omegas_history"].append(wheels_omegas)
        history["steering_joint_angle_history"].append(steering_joint_angle)
        history["angular_velocity_history"].append(angular_velocity)
        history["off_track_history"].append(is_car_off_track(env))
        history["wheel_poses_history"].append(get_wheel_poses(env))
        history["observation_history"].append(observation)

        # Simulation
        env.render()
        action = controller.get_action(
            observation,
            info,
            track=track,
            pose=pose,
            speed=speed,
            wheels_omegas=wheels_omegas,
            steering_joint_angle=steering_joint_angle,
            angular_velocity=angular_velocity,
            wheel_pose=history["wheel_poses_history"][-1],
        )
        observation, reward, terminated, truncated, info = env.step(action)

        # Record history
        history["action_history"].append(action)
        history["reward_history"].append(reward)

        # Go to next step
        seed_reward += reward  # type: ignore

        # Check for keyboard events like Esc or Q to stop the loop
        if conf.RENDER_MODE == "human":
            user_quit, user_skipped = did_user_quit_or_skip()
        # Increment step
        step += 1
        if step >= conf.MAX_TIME_STEPS:
            terminated = True
            break
    seed_reward = int(seed_reward)
    # Save record
    if conf.DO_RECORD and not user_quit:
        if id is None:
            id = get_next_record_name(dataset_path)
        else:
            id = str(id).zfill(3)
        record_path = os.path.join(dataset_path, f"{trial_id}_{id}_{seed}_{seed_reward}.npz")
        np.savez(
            record_path,
            seed=seed,
            seed_reward=seed_reward,
            terminated=terminated,
            truncated=truncated,
            done=done,
            user_skipped=user_skipped,
            track=track,
            **history,
            **controller.debug_states,
        )
        if print_out:
            print(f"ID {id} saved under {record_path}.")

    off_track_percentage = np.mean(history["off_track_history"])
    col_names = ["seed", "seed_reward", "done", "terminated", "truncated", "off_track_percentage"]
    output = seed, seed_reward, done, terminated, truncated, off_track_percentage
    return output, col_names


def run(argv):
    """
    Run the racecar with the given arguments.
    Args:
        argv: The list of command line arguments.
    Returns:
        List[float]: A list of rewards obtained during the simulation.
    """
    # Creating configurations
    args = env_utils.parse_args(argv)
    conf = get_conf(args)
    dataset_path = os.path.join(
        conf.RECORD_OUTPUT_DIR,
        conf.RECORD_DATASET_NAME + "_" + get_next_dataset_name(conf.RECORD_OUTPUT_DIR, conf.RECORD_DATASET_NAME),
    )
    # Create dataset directory
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    # Get config id
    trial_id = get_next_trial_name(dataset_path)

    # Simulation
    if args.mode in ["test", "debug"]:
        rewards = []
        for seed in conf.DEMO_SEEDS:
            output, _ = simulate((seed, args, dataset_path, trial_id, seed, True))
            rewards.append(output[REWARD_DIM])
    elif args.mode == "benchmark":
        rets = process_map(
            simulate,
            [(s, args, dataset_path, trial_id, s, False) for s in conf.DEMO_SEEDS],
            max_workers=conf.GYM_MAX_WORKERS,
        )
        rets_data = [ret[0] for ret in rets]
        rets_cols = rets[0][1]
        # Save results to pandas dataframe
        df = pd.DataFrame(rets_data, columns=rets_cols).sort_values(by="seed_reward")
        rewards = df["seed_reward"]
        df.to_csv(
            os.path.join(dataset_path, f"results_{trial_id}_{int(np.mean(rewards))}_{int(np.std(rewards))}_.csv"),
            index=False,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # Save configuration
    # pylint: disable=R0914
    with open(
        os.path.join(dataset_path, f"config_{trial_id}_{int(np.mean(rewards))}_{int(np.std(rewards))}.yaml"),
        "w",
    ) as f:
        f.write(conf.to_yaml())

    print(f"Trial ended. Mean reward: {np.mean(rewards)}. Std reward: {np.std(rewards)}")
    return rewards


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run(sys.argv[1:])
