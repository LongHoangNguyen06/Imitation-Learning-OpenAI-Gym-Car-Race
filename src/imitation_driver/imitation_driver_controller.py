from __future__ import annotations

from collections import defaultdict

import torch

from src.expert_drivers.abstract_classes.abstract_controller import AbstractController
from src.imitation_driver import network
from src.imitation_driver.multi_task_cnn import MultiTaskCNN
from src.imitation_driver.single_task_cnn import SingleTaskCNN
from src.utils.utils import concatenate_debug_states


class ImitationDriverController(AbstractController):
    def __init__(
        self, conf, model: network.AbstractNet | None = None, weights: str | None = None, store_debug_states=False
    ):
        super().__init__()
        self.conf = conf
        if model is None:
            assert weights is not None
            model_weights = torch.load(weights, weights_only=True)
            try:
                model = SingleTaskCNN(conf=conf, store_debug_states=store_debug_states).double().to(conf.DEVICE)
                model.seq = None  # type: ignore
                model.load_state_dict(model_weights)
            except:
                model = MultiTaskCNN(conf=conf, store_debug_states=store_debug_states).double().to(conf.DEVICE)
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
        from src.imitation_driver.training.preprocess import convert_action_models_to_gym, preprocess_input_testing

        observation, state = preprocess_input_testing(observation=observation, conf=self.conf)
        if self.store_debug_states:
            self.debug_states["noisy_state"].append(state)
        observation = torch.tensor(observation).double().to(self.conf.DEVICE)
        state = torch.tensor(state).double().to(self.conf.DEVICE)
        prediction = self.model(observation=observation, state=state)  # type: ignore
        if self.store_debug_states:
            concatenate_debug_states(self.model.debug_states, self.debug_states)
        self.model.reset()
        return convert_action_models_to_gym(
            steering=prediction.steering, acceleration=prediction.acceleration
        ).squeeze()
