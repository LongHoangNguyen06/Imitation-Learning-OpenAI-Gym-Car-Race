from __future__ import annotations

from collections import defaultdict

import torch

from src.expert_drivers.abstract_classes.abstract_controller import AbstractController
from src.imitation_driver import network
from src.utils import conf_utils
from src.utils.utils import concatenate_debug_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = conf_utils.get_default_conf()


class ImitationDriverController(AbstractController):
    def __init__(self, model: network.AbstractNet | None = None, weights: str | None = None, store_debug_states=False):
        super().__init__()
        if weights is None:
            weights = conf.MODEL_PATH
        if model is None:
            assert weights is not None
            model_weights = torch.load(weights, weights_only=True)
            try:
                model = network.MultiTaskCNN(store_debug_states=store_debug_states).double().to(device)
                model.seq = None  # type: ignore
                model.load_state_dict(model_weights)
            except:
                model = network.SingleTaskCNN(store_debug_states=store_debug_states).double().to(device)
                model.seq = None  # type: ignore
                model.load_state_dict(model_weights)
            model.share_memory()
        self.model = model
        self.debug_states = defaultdict(list)
        self.store_debug_states = store_debug_states

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
        self.model.reset()

    def get_action(self, observation, info, *args, **kwargs):
        """
        Get the continuous action to be executed by the car.

        Args:
            observation (np.ndarray): The observation received from the environment.
            info (dict): Additional information about the environment.

        Returns:
            np.ndarray: The continuous action to be executed by the car.
        """
        from src.training.utils.preprocess import convert_action_models_to_gym, preprocess_input_testing

        observation, state = preprocess_input_testing(obs=observation)
        if self.store_debug_states:
            self.debug_states["noisy_state_history"].append(state)
        observation = torch.tensor(observation).double().to(device)
        state = torch.tensor(state).double().to(device)
        _, __, ___, steering, acceleration = self.model(observation, state)  # type: ignore
        if self.store_debug_states:
            concatenate_debug_states(self.model.debug_states, self.debug_states)
        self.model.reset()
        return convert_action_models_to_gym(steering, acceleration).squeeze()
