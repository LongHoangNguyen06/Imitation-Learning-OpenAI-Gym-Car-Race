from __future__ import annotations

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
# isort:maintain_block
import os
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits

import wandb
from src.training.utils.debug_plot import *
from src.utils import conf_utils

# isort:end_maintain_block
conf = conf_utils.get_default_conf()


class Trainer:
    def __init__(self, model, data_loader, optimizers, tensorboard_writer, epoch):
        """
        Initializes the trainer object.
        Args:
            model: The model to be trained.
            data_loader: The data loader object for loading training data.
            optimizers: The optimizer(s) to be used for training.
            tensorboard_writer: The writer object for logging training progress to TensorBoard.
            epoch: The current epoch number.
            do_train: A boolean indicating whether to perform training or not. Default is True.
        """
        # Initialize the trainer
        self.model = model
        self.data_loader = data_loader
        self.optimizers = optimizers
        self.tensorboard_writer = tensorboard_writer
        self.epoch = epoch

        # Loss history
        self.losses = []
        self.steering_losses = []
        self.acceleration_losses = []
        self.road_segmentation_losses = []
        self.chevron_segmentation_losses = []
        self.curvature_losses = []

        # Grad flow
        self.avg_grad = defaultdict(list)

    # Function to train model
    def one_step(self):
        """
        Trains the model using the given data loader and optimizer for a specified number of epochs.
        Args:
            model (nn.Module): The model to be trained.
            data_loader (DataLoader): The data loader object that provides the training data.
            optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
            tensorboard_writer (SummaryWriter): The tensorboard writer object for logging.
            epoch (int): The current epoch number.
        Returns:
            Tuple[np.ndarray, float]: A tuple containing the array of training losses and the mean training loss.
        """
        # Set model to training mode if needed
        self.model.train()
        self.model.hook()

        # Initialize losses history
        for batch in self.data_loader:
            # Move batch to device
            batch = tuple(t.to(conf.DEVICE, non_blocking=True) for t in batch)
            (
                self.gt_observation,
                self.gt_state,
                self.gt_curvature,
                self.gt_chevron_mask,
                self.gt_chevron_mask_weights,
                self.gt_road_mask,
                self.gt_road_mask_weights,
                self.gt_steering,
                self.gt_acceleration,
            ) = batch

            # Reset gradients
            for opt in self.optimizers:
                opt.zero_grad()

            # Forward pass and compute loss
            self.loss = self._compute_loss()

            # Backpropagate the gradients
            self.loss.backward()

            # Do optimization
            for opt in self.optimizers:
                opt.step()

                # Collect gradient statistics
                self.collect_gradient_stats()

            # Logging
            self._log()
        self.model.unhook()

    def collect_gradient_stats(self):
        """
        Collects gradient statistics for the model's parameters.
        This method iterates over the named parameters of the model and collects the average gradient values
        for the parameters that require gradients and are not biases. The average gradient values are stored
        in the `avg_grad` dictionary.
        Returns:
            None
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad and "bias" not in n:
                self.avg_grad[n].append(p.grad.abs().mean().item())

    def _log(self):
        """
        Logs training information to various logging platforms.
        Parameters:
        i (int): The current iteration or step in the training process.
        force_plot (bool): If True, forces logging to TensorBoard regardless of the iteration step.
        Returns:
        None
        """
        self._log_wandb()
        if not conf.TENSOR_BOARD_LOG:
            return
        self._log_tensorboard_network_debug()

    def _compute_loss(self):
        """
        Compute the loss for the racecar model.
        Returns:
            float: The computed loss value.
        """
        # Forward pass
        (
            self.predicted_road_mask,
            self.predicted_chevrons_mask,
            self.predicted_curvature,
            self.predicted_steering,
            self.predicted_acceleration,
        ) = self.model(observation=self.gt_observation, state=self.gt_state)

        # Steering loss
        self.steering_loss = (
            (self.gt_steering.flatten() - self.predicted_steering.flatten())
            .abs()
            .mean()
        )
        self.steering_loss *= torch.tensor(conf.IMITATION_STEERING_LOSS).to(conf.DEVICE)

        # Acceleration loss
        self.acceleration_loss = (
            (
                self.gt_acceleration.flatten() - self.predicted_acceleration.flatten()
            )
            .abs()
            .mean()
        )
        self.acceleration_loss *= torch.tensor(conf.IMITATION_ACCELERATION_LOSS).to(conf.DEVICE)

        # Chevrons segmentation loss
        self.chevron_segmentation_loss = bce_logits(
            weight=self.gt_chevron_mask_weights.flatten(),
            target=self.gt_chevron_mask.flatten(),
            input=self.predicted_chevrons_mask.flatten(),
        )
        self.chevron_segmentation_loss *= torch.tensor(conf.IMITATION_CHEVRON_SEGMENTATION_LOSS).to(conf.DEVICE)

        # Road segmentation loss
        self.road_segmentation_loss = bce_logits(
            weight=self.gt_road_mask_weights.flatten(),
            target=self.gt_road_mask.flatten(),
            input=self.predicted_road_mask.flatten(),
        )
        self.road_segmentation_loss *= torch.tensor(conf.IMITATION_ROAD_SEGMENTATION_LOSS).to(conf.DEVICE)

        # Curvature loss
        self.curvature_loss = (
            (self.gt_curvature.flatten() - self.predicted_curvature.flatten()) ** 2
        ).mean()
        self.curvature_loss *= torch.tensor(conf.IMITATION_CURVATURE_LOSS).to(conf.DEVICE)

        # Sum all losses
        self.loss = (
            self.steering_loss
            + self.acceleration_loss
            + self.chevron_segmentation_loss
            + self.road_segmentation_loss
            + self.curvature_loss
        )

        # Append losses to history
        self.losses.append(self.loss.item())
        self.steering_losses.append(self.steering_loss.item())
        self.acceleration_losses.append(self.acceleration_loss.item())
        self.road_segmentation_losses.append(self.road_segmentation_loss.item())
        self.chevron_segmentation_losses.append(self.chevron_segmentation_loss.item())
        self.curvature_losses.append(self.curvature_loss.item())

        return self.loss

    def _log_tensorboard_activation(self, model, model_name):
        """
        Logs the gradient, activation, and weight information of a subnetwork to TensorBoard.
        Parameters:
            model (Subnetwork): The subnetwork model.
            model_name (str): The name of the subnetwork model.
        Returns:
            None
        """
        for key in model.activations.keys():
            if key.startswith("BatchNorm2d"):
                continue
            if key.startswith("Conv2d") or key.startswith("ConvTranspose2d"):
                self.tensorboard_writer.add_image(
                    f"{model_name}_activation_image_{key}",
                    plot_tensor_grid(model.activations[key]),
                    self.epoch,
                    dataformats="HWC",
                )

    def _log_tensorboard_network_debug(self):
        """
        Logs various network components to TensorBoard.
        """
        self.tensorboard_writer.add_figure("opt_grad_flow", plot_grad_flow(self.avg_grad), self.epoch)
        self.tensorboard_writer.add_image(
            "statistic_control_steering",
            plot_predicted_action_and_actual_action(
                predicted_action=self.predicted_steering, actual_action=self.gt_steering
            ),
            self.epoch,
            dataformats="HWC",
        )
        self.tensorboard_writer.add_image(
            "statistic_control_acceleration",
            plot_predicted_action_and_actual_action(
                predicted_action=self.predicted_acceleration, actual_action=self.gt_acceleration
            ),
            self.epoch,
            dataformats="HWC",
        )
        self.tensorboard_writer.add_image(
            "statistic_auxiliary_curvature",
            plot_predicted_action_and_actual_action(
                predicted_action=self.predicted_curvature, actual_action=self.gt_curvature
            ),
            self.epoch,
            dataformats="HWC",
        )
        self.tensorboard_writer.add_pr_curve(
            "statistic_auxiliary_chevron_segmentation",
            self.gt_road_mask.flatten(),
            F.sigmoid(self.predicted_chevrons_mask.flatten()),
            self.epoch,
        )
        self.tensorboard_writer.add_pr_curve(
            "statistic_auxiliary_road_segmentation",
            self.gt_road_mask.flatten(),
            F.sigmoid(self.predicted_road_mask.flatten()),
            self.epoch,
        )
        if not conf.TENSOR_BOARD_EXPENSIVE_LOG:
            return

        if hasattr(self.model, "backbone"):
            self._log_tensorboard_activation(self.model.backbone, "backbone")
        if hasattr(self.model, "road_decoder"):
            self._log_tensorboard_activation(self.model.road_decoder, "road_decoder")
        if hasattr(self.model, "chevrons_decoder"):
            self._log_tensorboard_activation(self.model.chevrons_decoder, "chevrons_decoder")
        if hasattr(self.model, "curvature_predictor"):
            self._log_tensorboard_activation(self.model.curvature_predictor, "curvature_predictor")
        if hasattr(self.model, "steering_predictor"):
            self._log_tensorboard_activation(self.model.steering_predictor, "steering_predictor")
        if hasattr(self.model, "acceleration_predictor"):
            self._log_tensorboard_activation(self.model.acceleration_predictor, "acceleration_predictor")

    def _log_wandb(self):
        """
        Logs the training or testing loss and learning rate to wandb.
        Parameters:
        - self: The instance of the class.
        Returns:
        - None
        """
        # Write to wandb
        if conf.WANDB_LOG:
            wandb.log({f"opt/train_loss": wandb.Histogram(self.losses)}, step=self.epoch)
            for i, opt in enumerate(self.optimizers):
                wandb.log({f"opt/lr_{i}": opt.param_groups[0]["lr"]}, step=self.epoch)
            wandb.log({f"opt/train_loss_mean": np.mean(self.losses)}, step=self.epoch)
            wandb.log({f"opt/steering_loss_mean": np.mean(self.steering_losses)}, step=self.epoch)
            wandb.log({f"opt/acceleration_loss_mean": np.mean(self.acceleration_losses)}, step=self.epoch)
            wandb.log({f"opt/road_segmentation_loss_mean": np.mean(self.road_segmentation_losses)}, step=self.epoch)
            wandb.log(
                {f"opt/chevron_segmentation_loss_mean": np.mean(self.chevron_segmentation_losses)}, step=self.epoch
            )
            wandb.log({f"opt/curvature_loss_mean": np.mean(self.curvature_losses)}, step=self.epoch)
