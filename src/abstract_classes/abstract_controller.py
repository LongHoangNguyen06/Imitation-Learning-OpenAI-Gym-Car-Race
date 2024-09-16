from __future__ import annotations


class AbstractController:
    def __init__(self):
        """Constructor"""
        self.conf = None
        self.debug_states = dict()

    def reset(self):
        """
        Resets the controller.
        """
        self.debug_states = dict()

    def get_action(self, observation, info, *args, **kwargs):
        """
        Returns the action based on the given observation and info.
        Parameters:
        - observation: The observation of the environment.
        - info: Additional information about the environment.
        - *args: Variable length argument list.
        - **kwargs: Arbitrary keyword arguments.
        Returns:
        - The action based on the observation and info.
        """
        raise NotImplementedError("Method 'get_action' must be implemented in a subclass.")
