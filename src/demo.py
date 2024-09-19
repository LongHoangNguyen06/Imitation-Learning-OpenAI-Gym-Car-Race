from __future__ import annotations

import sys

import gymnasium as gym
import torch

from src.imitation_driver.imitation_driver_controller import ImitationDriverController

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
controller = ImitationDriverController(weights=sys.argv[1])  # type: ignore


env = gym.make("CarRacing-v2", render_mode="human", continuous=True)
for seed in range(10):
    observation, info = env.reset(seed=0)
    terminated = truncated = done = False
    while not (done or terminated or truncated):
        env.render()
        action = controller.get_action(observation,info) # Model only see noisy observations
        observation, reward, terminated, truncated, info = env.step(action)

# Close the environment
env.close()
