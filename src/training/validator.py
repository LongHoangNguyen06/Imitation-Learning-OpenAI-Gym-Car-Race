from __future__ import annotations

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
# isort:maintain_block
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import torch
from dynaconf import Dynaconf
from tqdm.contrib.concurrent import process_map

from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from src.utils import env_utils
from src.utils.training_debug_plot import *

# isort:end_maintain_block
conf = Dynaconf(settings_files=["src/conf/default_conf.py"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Closed loop simulation
def validate(evaluation_input):
    """
    Validates the performance of a given model on a racecar simulation.
    Args:
        evaluation_input (tuple): A tuple containing the model and seed.
    Returns:
        int: The total reward obtained by the model during the simulation.
    """
    model, seed = evaluation_input
    model.eval()  # Set model to evaluation mode
    controller = ImitationDriverController(conf=conf, model=model)
    env = env_utils.create_env(conf=conf)

    # Initialize new scenario
    terminated = truncated = done = False
    observation, info = env.reset(seed=seed)
    seed_reward = step = 0

    # Start simulation
    while not (done or terminated or truncated):
        action = controller.get_action(
            observation,
            info,
            speed=env_utils.get_speed(env),
            wheels_omegas=env_utils.get_wheel_velocities(env),
            angular_velocity=env.unwrapped.car.hull.angularVelocity,  # type: ignore
            steering_joint_angle=env.unwrapped.car.wheels[0].joint.angle,  # type: ignore
        ).squeeze()
        observation, reward, terminated, truncated, info = env.step(action)
        seed_reward += reward  # type: ignore
        step += 1
        if step >= conf.MAX_TIME_STEPS:
            break
    return int(seed_reward)

class Validator:
    def __init__(self, model):
        self.model = model

    def validate(self):
        return process_map(
            validate,
            [(self.model, seed) for seed in conf.IMITATION_EVALUATION_SEEDS],
            max_workers=conf.GYM_MAX_WORKERS,
            desc="Validating Closed Loop",
        )
