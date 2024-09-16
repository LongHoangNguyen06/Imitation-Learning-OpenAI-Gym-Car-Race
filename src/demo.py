from __future__ import annotations

import sys

import gymnasium as gym
import torch

from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from src.imitation_driver.network import SingleTaskCNN
from src.utils.env_utils import get_speed, get_wheel_velocities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SingleTaskCNN(print_shapes=True).to(device).double()
model.seq = None
model.load_state_dict(torch.load(sys.argv[1]))
controller = ImitationDriverController(conf=None, model=model)

env = gym.make("CarRacing-v2", render_mode="human", continuous=True)
for seed in range(10):
    observation, info = env.reset(seed=0)
    terminated = truncated = done = False
    while not (done or terminated or truncated):
        env.render()
        speed = get_speed(env)
        wheels_omegas = get_wheel_velocities(env)
        steering_joint_angle = env.unwrapped.car.wheels[0].joint.angle  # type: ignore
        angular_velocity = env.unwrapped.car.hull.angularVelocity  # type: ignore
        action = controller.get_action(
            observation,
            info,
            speed=speed,
            wheels_omegas=wheels_omegas,
            steering_joint_angle=steering_joint_angle,
            angular_velocity=angular_velocity,
        )
        observation, reward, terminated, truncated, info = env.step(action)

# Close the environment
env.close()
