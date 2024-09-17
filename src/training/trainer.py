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
import wandb
from dynaconf import Dynaconf
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from tqdm import tqdm

from src.utils.training_debug_plot import *

# isort:end_maintain_block
conf = Dynaconf(settings_files=["src/conf/default_conf.py"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.chevron_visble_losses = []
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

        # Initialize losses history
        for i, (observation, state, curvature, masks, action, instance_weights) in enumerate(self.data_loader):
            # Reset gradients
            for opt in self.optimizers:
                opt.zero_grad()

            # Load data to GPU
            self.gt_observation = observation.to(device, non_blocking=True)
            self.gt_state = state.to(device, non_blocking=True)
            self.gt_instance_weights = instance_weights.to(device, non_blocking=True)
            self.gt_action = action.to(device, non_blocking=True)
            self.gt_curvature = curvature.to(device, non_blocking=True)
            self.gt_masks = masks.to(device, non_blocking=True)
            self.gt_chevron_mask = masks[:, 0, :, :].to(device, non_blocking=True)
            self.gt_chevron_visible = (
                (self.gt_chevron_mask.sum(dim=(1, 2)) > conf.IMITATION_CHEVRON_VISIBLE_THRESHOLD)
                .to(device, non_blocking=True)
                .flatten()
                .double()
            )
            self.gt_road_mask = masks[:, 1, :, :].to(device, non_blocking=True)
            self.gt_steering = action[:, 0].to(device, non_blocking=True)
            self.gt_acceleration = action[:, 1].to(device, non_blocking=True)

            # Forward pass and compute loss
            self.loss = self._compute_loss()

            # Backpropagate the gradients and do optimization
            self.loss.backward()
            for opt in self.optimizers:
                opt.step()

                self.collect_gradient_stats()
            self._log(i)
        self._log(i, force_plot=True)  # type: ignore

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

    def _log(self, i, force_plot=False):
        if not conf.TENSOR_BOARD_LOG:
            return
        if (i % conf.TENSOR_BOARD_WRITING_FREQUENCY == 0) or (force_plot and conf.TENSOR_BOARD_LOG):
            self._log_tensorboard_network_debug()
        self._log_tensorboard_metrics()
        self._log_wandb()

    def _compute_loss(self):
        """
        Compute the loss for the racecar model.
        Returns:
            float: The computed loss value.
        """
        # Forward pass
        (
            self.predicted_road_mask,
            self.predicted_chevrons_visible,
            self.predicted_curvature,
            self.predicted_steering,
            self.predicted_acceleration,
        ) = self.model(observation=self.gt_observation, state=self.gt_state)

        # Compute and scale individual losses
        self.steering_loss = (
            torch.tensor(conf.IMITATION_STEERING_LOSS).to(device)
            * (self.gt_steering.flatten() - self.predicted_steering.flatten()).abs().mean()
        )
        self.acceleration_loss = (
            torch.tensor(conf.IMITATION_ACCELERATION_LOSS).to(device)
            * (self.gt_acceleration.flatten() - self.predicted_acceleration.flatten()).abs().mean()
        )
        self.chevron_visble_loss = torch.tensor(conf.IMITATION_CHEVRON_VISIBLE_LOSS).to(device) * bce_logits(
            self.predicted_chevrons_visible.flatten(),
            self.gt_chevron_visible,
            weight=self.gt_instance_weights.flatten(),
        )
        self.road_segmentation_loss = torch.tensor(conf.IMITATION_SEGMENTATION_LOSS).to(device) * bce_logits(
            self.predicted_road_mask.flatten(start_dim=1),
            self.gt_road_mask.flatten(start_dim=1),
            weight=self.gt_instance_weights.reshape(-1, 1),
        )
        self.curvature_loss = (
            torch.tensor(conf.IMITATION_CURVATURE_LOSS).to(device)
            * ((self.gt_curvature.flatten() - self.predicted_curvature.flatten()) ** 2).mean()
        )

        # Sum all losses
        self.loss = (
            self.steering_loss
            + self.acceleration_loss
            + self.road_segmentation_loss
            + self.chevron_visble_loss
            + self.curvature_loss
        )

        # Append losses to history
        self.losses.append(self.loss.item())
        self.steering_losses.append(self.steering_loss.item())
        self.acceleration_losses.append(self.acceleration_loss.item())
        self.road_segmentation_losses.append(self.road_segmentation_loss.item())
        self.chevron_visble_losses.append(self.chevron_visble_loss.item())
        self.curvature_losses.append(self.curvature_loss.item())

        return self.loss

    def _log_tensorboard_subnetwork_gradient_and_activation(self, model, model_name):
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
            "input_observation_image",
            plot_observation_grid(
                observation=self.gt_observation, action=self.gt_action, weight=self.gt_instance_weights
            ),
            self.epoch,
            dataformats="HWC",
        )
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
            "statistic_auxiliary_chevron_visibles",
            self.gt_chevron_visible.flatten(),
            F.sigmoid(self.predicted_chevrons_visible.flatten()),
            self.epoch,
        )
        self.tensorboard_writer.add_pr_curve(
            "statistic_auxiliary_road_segmentation",
            self.gt_road_mask.flatten(),
            F.sigmoid(self.predicted_road_mask.flatten()),
            self.epoch,
        )

        if hasattr(self.model, "backbone"):
            self._log_tensorboard_subnetwork_gradient_and_activation(self.model.backbone, "backbone")
        if hasattr(self.model, "road_decoder"):
            self._log_tensorboard_subnetwork_gradient_and_activation(self.model.road_decoder, "road_decoder")
        if hasattr(self.model, "chevrons_predictor"):
            self._log_tensorboard_subnetwork_gradient_and_activation(
                self.model.chevrons_predictor, "chevrons_predictor"
            )
        if hasattr(self.model, "curvature_predictor"):
            self._log_tensorboard_subnetwork_gradient_and_activation(
                self.model.curvature_predictor, "curvature_predictor"
            )
        if hasattr(self.model, "steering_predictor"):
            self._log_tensorboard_subnetwork_gradient_and_activation(
                self.model.steering_predictor, "steering_predictor"
            )
        if hasattr(self.model, "acceleration_predictor"):
            self._log_tensorboard_subnetwork_gradient_and_activation(
                self.model.acceleration_predictor, "acceleration_predictor"
            )

    def _log_tensorboard_metrics(self):
        """
        Logs metrics to TensorBoard.
        These metrics are logged with their corresponding values and the current epoch.
        Parameters:
        - None
        Returns:
        - None
        """
        loss_dict = {
            "loss": np.mean(self.losses),
            "steering_loss": np.mean(self.steering_losses),
            "acceleration_loss": np.mean(self.acceleration_losses),
            "road_segmentation": np.mean(self.road_segmentation_losses),
            "chevron_visible_loss": np.mean(self.chevron_visble_losses),
            "curvature_loss": np.mean(self.curvature_losses),
        }

        # Write all the scalars at once
        self.tensorboard_writer.add_scalars(f"opt_train_losses", loss_dict, self.epoch)

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
            wandb.log({f"opt_train_loss": wandb.Histogram(self.losses)}, step=self.epoch)
            for i, opt in enumerate(self.optimizers):
                wandb.log({f"opt_lr_{i}": opt.param_groups[0]["lr"]}, step=self.epoch)
            wandb.log({f"opt_train_loss_mean": np.mean(self.losses)}, step=self.epoch)
