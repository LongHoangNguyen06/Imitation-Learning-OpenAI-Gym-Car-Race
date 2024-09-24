from __future__ import annotations

from collections import defaultdict


class AbstractController:
    def __init__(self):
        self.debug_states: defaultdict = defaultdict(list)
        self.conf = None

    def reset(self):
        """
        Resets the controller.
        """
        self.debug_states = defaultdict(list)

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
