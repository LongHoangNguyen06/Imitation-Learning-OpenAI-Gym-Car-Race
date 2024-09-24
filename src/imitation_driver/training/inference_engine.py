from __future__ import annotations


import torch

from src.imitation_driver import network
from src.imitation_driver.training.loss import Loss
from src.imitation_driver.training.preprocess import GroundTruth
from src.utils import conf_utils

conf = conf_utils.get_default_conf()


class InferenceEngine:
    def __init__(self, model: network.AbstractNet, optimizer: torch.optim.Adam = None):  # type: ignore
        """
        Initializes the engine object.
        Args:
            model: The model to be trained.
            optimizers: The optimizer(s) to be used for training.
        """
        # Initialize the engine
        self.model = model
        self.optimizer = optimizer
        self.loss_function = Loss()

        # Grad flow
        self.gradients = dict()
        self.activations = dict()
        self.weights = dict()

    # Function to train model
    def forward(self, data_loader):
        """
        Trains the model using the given data loader and optimizer for a specified number of epochs.
        Args:
            data_loader (DataLoader): The data loader object that provides the training data.
        """
        # Set model to training mode if needed
        if self.optimizer:
            self.model.train()
        self.model.hook()
        # Initialize losses history
        for batch in data_loader:
            # Move batch to device
            self.loss_function.gt = GroundTruth(*[t.to(conf.DEVICE, non_blocking=True) for t in batch])

            # Reset gradients
            if self.optimizer:
                self.optimizer.zero_grad()

            # Forward pass and compute loss
            self.loss_function.pred = self.model(**self.loss_function.gt._asdict()) # type: ignore
            self.loss = self.loss_function.get_loss()

            # Backpropagate the gradients
            if self.optimizer:
                self.loss.backward()
                self.optimizer.step()

                # Collect gradient statistics
                self._collect_gradient_stats()
        self.model.unhook()

    def _collect_gradient_stats(self):
        """
        Collects gradient statistics for the model's parameters.
        This method iterates over the named parameters of the model and collects the average gradient values
        for the parameters that require gradients and are not biases. The average gradient values are stored
        in the `gradients` dictionary.
        Returns:
            None
        """
        self.gradients = self.model.gradients.copy()
        self.activations = self.model.activations.copy()
        self.weights = self.model.weights.copy()
