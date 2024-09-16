from __future__ import annotations

import pygame
from dynaconf import Dynaconf

from src.abstract_classes.abstract_controller import AbstractController


class HumanDriverController(AbstractController):
    def __init__(self, conf: Dynaconf):
        super().__init__()
        pygame.init()
        self.conf = conf

    def get_action(self, observation, info, *args, **kwargs):
        """
        Get the continuous action based on the keyboard input.
        Args:
            observation: The observation of the environment.
            info: Additional information about the environment.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            action: A list representing the action to be taken. The first element is the steering value,
                    the second element is the gas value, and the third element is the brake value.
        """
        pygame.event.pump()  # Process event queue

        keys = pygame.key.get_pressed()  # Get the state of all keyboard keys

        # Define action variables for steering, gas, and brake
        action = [0, 0, 0]  # [steering, gas, brake]

        if keys[pygame.K_w] or keys[pygame.K_UP]:  # Forward (gas)
            action[1] = self.conf.GAS_VALUE  # gas

        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  # Backward (Brake)
            action[2] = self.conf.BRAKE_VALUE  # Brake

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  # Left (Steering)
            action[0] = self.conf.STEERLEFT_VALUE  # Steer left

        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:  # Right (Steering)
            action[0] = self.conf.STEERRIGHT_VALUE  # Steer right

        return action
