from __future__ import annotations

from collections import defaultdict
from turtle import forward

import torch
from dynaconf import Dynaconf

from src.abstract_classes.abstract_controller import AbstractController
from src.imitation_driver import network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImitationDriverController(AbstractController):
    def __init__(self, conf: Dynaconf, model=network.AbstractNet):
        super().__init__()
        if model is None:
            model = network.MultiTaskCNN().double().to(device)
            if conf.IMITATION_MODEL_PATH is not None:
                model.load_state_dict(torch.load(conf.IMITATION_MODEL_PATH))
            model.share_memory()
        self.conf = conf
        self.model = model

    def reset(self):
        """
        Resets the pure pursuit controller.
        This method resets the pure pursuit controller by calling the reset method of the superclass,
        resetting the longitudinal controller, and clearing the debug states.
        Parameters:
            None
        Returns:
            None
        """
        super().reset()
        self.debug_states = defaultdict(list)

    def get_action(self, observation, info, *args, **kwargs):
        """
        Get the continuous action to be executed by the car.

        Args:
            observation (np.ndarray): The observation received from the environment.
            info (dict): Additional information about the environment.

        Returns:
            np.ndarray: The continuous action to be executed by the car.
        """
        from src.utils.dataset import convert_action_models_to_gym, preprocess_input_testing

        observation, state = preprocess_input_testing(
            obs=observation,
            speed=kwargs["speed"],
            wheels_omegas=kwargs["wheels_omegas"],
            angular_velocity=kwargs["angular_velocity"],
            steering_joint_angle=kwargs["steering_joint_angle"],
        )
        observation = torch.tensor(observation).double().to(device)
        state = torch.tensor(state).double().to(device)
        _, __, ___, steering, acceleration = self.model(observation, state)  # type: ignore

        action = torch.cat([steering, acceleration], dim=1).cpu().detach().numpy()

        return convert_action_models_to_gym(action).squeeze()
