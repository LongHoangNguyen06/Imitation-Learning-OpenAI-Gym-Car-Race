from __future__ import annotations

# isort:maintain_block
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # isort:skip
# isort:end_maintain_block

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch.multiprocessing as mp

from src.expert_drivers.abstract_classes.abstract_controller import AbstractController
from src.expert_drivers.pid_driver.pid_driver_controller import PidDriverController
from src.expert_drivers.pure_pursuit_driver.pure_pursuit_controller import PurePursuitController
from src.expert_drivers.stanley_driver.stanley_controller import StanleyController
from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from src.utils import conf_utils
from src.utils.io_utils import get_next_record_number
from src.utils.simulator import Simulator


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
        "--student_controller",
        type=str,
        default="pid",
        choices=["pid", "pure_pursuit", "stanley", "imitation"],
        help="Controller of agent.",
    )
    parser.add_argument(
        "--student_model_path",
        type=str,
        default=None,
        help="Optional parameter path when recording an imitator.",
    )
    parser.add_argument(
        "--teacher_controller",
        type=str,
        default=None,
        choices=["pid", "pure_pursuit", "stanley", "imitation"],
        help="Controller of agent.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=600,
        help="Optional parameter on how many iterations the simulator should run.",
    )
    parser.add_argument(
        "--record_name",
        type=str,
        default="benchmark",
        help="Optional parameter for i which output dir the records should be stored at.",
    )
    return parser.parse_args(argv)


def get_controller(controller: str, conf, *_, **kwargs) -> AbstractController:
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
    if controller == "pid":
        return PidDriverController(conf=conf)
    if controller == "pure_pursuit":
        return PurePursuitController(conf=conf)
    if controller == "stanley":
        return StanleyController(conf=conf)
    if controller == "imitation":
        assert "model_path" in kwargs, "Model path must be provided for imitation controller."
        return ImitationDriverController(conf=conf, weights=kwargs["model_path"], store_debug_states=True)
    raise ValueError(f"Invalid controller: {controller}")


def run(argv):
    """
    Run the racecar with the given arguments.
    Args:
        argv: The list of command line arguments.
    Returns:
        List[float]: A list of rewards obtained during the simulation.
    """
    # Creating configurations
    args = parse_args(argv)
    conf = conf_utils.get_conf(args.student_controller)

    # Create output directory
    dataset_path = os.path.join(
        conf.RECORD_OUTPUT_DIR,
        args.record_name + "_" + get_next_record_number(conf.RECORD_OUTPUT_DIR, args.record_name),
    )
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    print(f"Recording to {dataset_path}")

    # Simulation
    data, reward = Simulator(
        conf=conf,
        output_dir=dataset_path,
        max_steps=args.max_iterations,
        student_controller=get_controller(args.student_controller, conf=conf, model_path=args.student_model_path),
        teacher_controller=(
            get_controller(args.teacher_controller, conf_utils.get_conf(args.teacher_controller))
            if args.teacher_controller is not None
            else None
        ),
        dagger_mode=False,
        do_early_break=False,
    ).simulate(0)

    # Save results to pandas dataframe
    df = pd.DataFrame(
        data, columns=["seed", "seed_reward", "done", "terminated", "truncated", "off_track_percentage"]
    ).sort_values(by="seed_reward")
    df.to_csv(
        os.path.join(dataset_path, f"results_{args.record_name}_{reward}.csv"),
        index=False,
    )

    # Print statistics
    rewards = df["seed_reward"]
    print(f"Mean={reward}. Std={np.std(rewards)}. Min={np.min(rewards)}. Max={np.max(rewards)}")
    return rewards


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run(sys.argv[1:])
