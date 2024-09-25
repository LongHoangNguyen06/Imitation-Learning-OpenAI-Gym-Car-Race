from __future__ import annotations

import gc
import os
import socket

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import wandb
from src.utils.simulator import teacher_action_probability


def fig_to_img(fig):
    """
    Converts a matplotlib figure to a wandb image.
    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure to convert.
    """
    img = wandb.Image(fig)
    plt.close(fig)
    del fig
    gc.collect()
    return img


def plot_grad_flow(gradients):
    """
    Plots the gradient flow of a neural network during training.
    Args:
        gradients (dict): A dictionary where keys are layer names and values are the average gradients for those layers.
    """
    names = list(gradients.keys())
    values = [np.mean(grad) for grad in gradients.values()]
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(111)
    ax.bar(names, values)
    ax.set_ylim(0, 0.001)
    ax.set_xlabel("Average gradient")
    ax.set_title("Gradient flow")
    return fig_to_img(fig)


def scatter_plot(gt, predicted, name):
    """
    Plots a scatter plot of the ground truth and predicted values.
    Args:
        gt (np.ndarray): The ground truth values.
        predicted (np.ndarray): The predicted values.
    """
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy().flatten()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy().flatten()

    max_abs = max(np.max(np.abs(gt)), np.max(np.abs(predicted)))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(gt, predicted)
    ax.set_xlabel(f"{name} ground truth")
    ax.set_ylabel(f"{name} predicted")
    ax.set_title(f"{name} ground truth vs. predicted")
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    return fig_to_img(fig)


def get_code_artifact_dir(conf):
    """
    Returns the directory path for storing code artifacts based on the current hostname.
    :return: The directory path for code artifacts.
    :rtype: str
    """
    return os.path.join(conf.OUTPUT_DIR, socket.gethostname())


def plot_rgb(rgb, conf):
    """
    Plots an RGB image.
    Args:
        rgb: The RGB image to plot. Shape (64, channel, height, width)
    """
    rgb = rgb.detach().cpu().numpy()
    rgb = (rgb * conf.OBS_STD + conf.OBS_MEAN).astype(np.uint8)
    rgb = np.transpose(rgb, (0, 2, 3, 1))
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    for i in range(min(64, rgb.shape[0])):
        ax = axes[i // 8, i % 8]
        ax.imshow(rgb[i])
        ax.axis("off")
    return fig_to_img(fig)


def plot_masks(masks):
    """
    Plots a mask image.
    Args:
        masks: The mask image to plot. Shape (64, 1, height, width)
    """
    masks = F.sigmoid(masks)
    masks = masks.detach().cpu().numpy().squeeze()
    masks = (masks * 255).astype(np.uint8)
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    for i in range(min(64, masks.shape[0])):
        ax = axes[i // 8, i % 8]
        ax.imshow(masks[i], cmap="gray")
        ax.axis("off")
    return fig_to_img(fig)


class WandbLogger:
    def __init__(self, model, run_id, conf) -> None:
        self.conf = conf
        if conf.IMITATION_WANDB_LOG:
            # Store commit and branch
            wandb_conf = conf.copy()
            if "GIT_COMMIT_HASH" in os.environ:
                wandb_conf["GIT_COMMIT_HASH"] = os.environ["GIT_COMMIT_HASH"]
            if "GIT_BRANCH" in os.environ:
                wandb_conf["GIT_BRANCH"] = os.environ["GIT_BRANCH"]
            if "GIT_COMMIT_MESSAGE" in os.environ:
                wandb_conf["GIT_COMMIT_MESSAGE"] = os.environ["GIT_COMMIT_MESSAGE"]
            wandb.init(project="CarRace", name=f"{run_id}", config=wandb_conf)
            if os.path.exists(get_code_artifact_dir(conf)):
                # Add the code artifact to wandb
                artifact = wandb.Artifact(f"{run_id}_code_artifact", type="code")
                artifact.add_dir(get_code_artifact_dir(conf))
                wandb.log_artifact(artifact)

        self.model = model
        self.best_validate_reward = float("-inf")
        self.wandb_model_artifact = None
        self.model_name = f"{run_id}_{model.__class__.__name__}"
        self.model_save_file_prefix = os.path.join(conf.IMITATION_OUTPUT_MODEL, self.model_name)
        self.model = model

    def log_closed_loop(self, simulator, epoch):
        """
        Logs various metrics to Weights and Biases (wandb) during the closed-loop training process.
        Parameters:
        simulator (Simulator): The simulator object containing the rewards and output directory.
        epoch (int): The current epoch number.
        Metrics logged:
        - "dagger/rewards": Histogram of rewards from the simulator.
        - "dagger/reward": Single reward value from the simulator.
        - "dagger/teacher_action_probability": Probability of the teacher's action at the given epoch.
        - "dagger/#records": Number of records in the simulator's output directory.
        """
        wandb.log({"dagger/rewards": wandb.Histogram(list(simulator.rewards))}, step=epoch)
        wandb.log({"dagger/reward": simulator.reward}, step=epoch)
        wandb.log({"dagger/teacher_action_proboability": teacher_action_probability(epoch, conf=self.conf)}, step=epoch)
        wandb.log({"dagger/#records": len(os.listdir(simulator.output_dir))}, step=epoch)
        wandb.log({"dagger/off_track": wandb.Histogram(simulator.off_track)}, step=epoch)
        wandb.log({"dagger/off_track_mean": np.mean(simulator.off_track)}, step=epoch)

    def _log_losses(self, dataset_name, loss_function, epoch):
        """
        Logs various loss metrics to Weights and Biases (wandb) for a given dataset and epoch.
        Args:
            dataset_name (str): The name of the dataset being used.
            loss_function (object): An object containing various loss metrics.
            epoch (int): The current epoch number.
        Logs:
            - Training loss histogram.
            - Learning rate (if in training mode).
            - Mean training loss.
            - Mean steering loss.
            - Mean acceleration loss.
            - Mean road segmentation loss.
            - Mean chevron segmentation loss.
            - Mean curvature loss.
            - Mean desired speed loss.
            - Mean speed error loss.
            - Mean cross-track error (cte) loss.
            - Mean heading error (he) loss.
        """
        # Log losses
        wandb.log({f"opt_{dataset_name}_mean/loss": np.mean(loss_function.losses)}, step=epoch)
        wandb.log({f"opt_{dataset_name}_mean/steering_loss": np.mean(loss_function.steering_losses)}, step=epoch)
        wandb.log(
            {f"opt_{dataset_name}_mean/acceleration_loss": np.mean(loss_function.acceleration_losses)}, step=epoch
        )
        wandb.log(
            {f"opt_{dataset_name}_mean/road_segmentation_loss": np.mean(loss_function.road_segmentation_losses)},
            step=epoch,
        )
        wandb.log(
            {f"opt_{dataset_name}_mean/chevron_segmentation_loss": np.mean(loss_function.chevron_segmentation_losses)},
            step=epoch,
        )
        wandb.log({f"opt_{dataset_name}_mean/curvature_loss": np.mean(loss_function.curvature_losses)}, step=epoch)
        wandb.log(
            {f"opt_{dataset_name}_mean/desired_speed_loss": np.mean(loss_function.desired_speed_losses)}, step=epoch
        )
        wandb.log({f"opt_{dataset_name}_mean/speed_error_loss": np.mean(loss_function.speed_error_losses)}, step=epoch)
        wandb.log({f"opt_{dataset_name}_mean/cte_loss": np.mean(loss_function.cte_losses)}, step=epoch)
        wandb.log({f"opt_{dataset_name}_mean/he_loss": np.mean(loss_function.he_losses)}, step=epoch)

        # Log losses histogram
        wandb.log({f"opt_{dataset_name}/loss": wandb.Histogram(loss_function.losses)}, step=epoch)
        wandb.log({f"opt_{dataset_name}/steering_loss": wandb.Histogram(loss_function.steering_losses)}, step=epoch)
        wandb.log(
            {f"opt_{dataset_name}/acceleration_loss": wandb.Histogram(loss_function.acceleration_losses)}, step=epoch
        )
        wandb.log(
            {f"opt_{dataset_name}/road_segmentation_loss": wandb.Histogram(loss_function.road_segmentation_losses)},
            step=epoch,
        )
        wandb.log(
            {
                f"opt_{dataset_name}/chevron_segmentation_loss": wandb.Histogram(
                    loss_function.chevron_segmentation_losses
                )
            },
            step=epoch,
        )
        wandb.log({f"opt_{dataset_name}/curvature_loss": wandb.Histogram(loss_function.curvature_losses)}, step=epoch)
        wandb.log(
            {f"opt_{dataset_name}/desired_speed_loss": wandb.Histogram(loss_function.desired_speed_losses)}, step=epoch
        )
        wandb.log(
            {f"opt_{dataset_name}/speed_error_loss": wandb.Histogram(loss_function.speed_error_losses)}, step=epoch
        )
        wandb.log({f"opt_{dataset_name}/cte_loss": wandb.Histogram(loss_function.cte_losses)}, step=epoch)
        wandb.log({f"opt_{dataset_name}/he_loss": wandb.Histogram(loss_function.he_losses)}, step=epoch)

        # Log cosine similarities
        wandb.log(
            {f"cosine_similarity_{dataset_name}_mean/steering": np.mean(loss_function.steering_cosine_similarity)},
            step=epoch,
        )
        wandb.log(
            {
                f"cosine_similarity_{dataset_name}_mean/acceleration": np.mean(
                    loss_function.acceleration_cosine_similarity
                )
            },
            step=epoch,
        )
        wandb.log(
            {f"cosine_similarity_{dataset_name}_mean/curvature": np.mean(loss_function.curvature_cosine_similarity)},
            step=epoch,
        )
        wandb.log(
            {
                f"cosine_similarity_{dataset_name}_mean/desired_speed": np.mean(
                    loss_function.desired_speed_cosine_similarity
                )
            },
            step=epoch,
        )
        wandb.log(
            {
                f"cosine_similarity_{dataset_name}_mean/speed_error": np.mean(
                    loss_function.speed_error_cosine_similarity
                )
            },
            step=epoch,
        )
        wandb.log(
            {f"cosine_similarity_{dataset_name}_mean/cte": np.mean(loss_function.cte_cosine_similarity)}, step=epoch
        )
        wandb.log(
            {f"cosine_similarity_{dataset_name}_mean/he": np.mean(loss_function.he_cosine_similarity)}, step=epoch
        )

        # Cosine similarities histograms
        wandb.log(
            {f"cosine_similarity_{dataset_name}/steering": wandb.Histogram(loss_function.steering_cosine_similarity)},
            step=epoch,
        )
        wandb.log(
            {
                f"cosine_similarity_{dataset_name}/acceleration": wandb.Histogram(
                    loss_function.acceleration_cosine_similarity
                )
            },
            step=epoch,
        )
        wandb.log(
            {f"cosine_similarity_{dataset_name}/curvature": wandb.Histogram(loss_function.curvature_cosine_similarity)},
            step=epoch,
        )
        wandb.log(
            {
                f"cosine_similarity_{dataset_name}/desired_speed": wandb.Histogram(
                    loss_function.desired_speed_cosine_similarity
                )
            },
            step=epoch,
        )
        wandb.log(
            {
                f"cosine_similarity_{dataset_name}/speed_error": wandb.Histogram(
                    loss_function.speed_error_cosine_similarity
                )
            },
            step=epoch,
        )
        wandb.log(
            {f"cosine_similarity_{dataset_name}/cte": wandb.Histogram(loss_function.cte_cosine_similarity)},
            step=epoch,
        )
        wandb.log(
            {f"cosine_similarity_{dataset_name}/he": wandb.Histogram(loss_function.he_cosine_similarity)}, step=epoch
        )

    def _log_predictions(self, dataset_name, loss_function, epoch):
        """
        Logs various prediction visualizations to Weights and Biases (wandb).
        Parameters:
        -----------
        dataset_name : str
            The name of the dataset being used.
        loss_function : object
            The loss function object containing ground truth (gt) and predicted (pred) values.
        epoch : int
            The current epoch number.
        training_mode : bool
            Indicates whether the model is in training mode.
        inference_engine : object
            The inference engine being used.
        Logs:
        -----
        - Inputs as RGB images.
        - Predicted road masks.
        - Predicted chevron masks.
        - Scatter plots for steering, acceleration, curvature, desired speed, speed error, cross track error (cte), and heading error (he).
        """
        wandb.log({f"viz_{dataset_name}/inputs": plot_rgb(loss_function.gt.observation, conf=self.conf)}, epoch)
        wandb.log({f"viz_{dataset_name}/predicted_road_masks": plot_masks(loss_function.pred.road_mask)}, epoch)
        wandb.log(
            {f"viz_{dataset_name}/predicted_chevron_masks": plot_masks(loss_function.pred.chevron_mask)},
            epoch,
        )
        wandb.log(
            {
                f"viz_{dataset_name}/steering": scatter_plot(
                    loss_function.gt.steering, loss_function.pred.steering, "Steering"
                )
            },
            epoch,
        )
        wandb.log(
            {
                f"viz_{dataset_name}/acceleration": scatter_plot(
                    loss_function.gt.acceleration, loss_function.pred.acceleration, "Acceleration"
                )
            },
            epoch,
        )
        wandb.log(
            {
                f"viz_{dataset_name}/curvature": scatter_plot(
                    loss_function.gt.curvature, loss_function.pred.curvature, "Curvature"
                )
            },
            epoch,
        )
        wandb.log(
            {
                f"viz_{dataset_name}/desired_speed": scatter_plot(
                    loss_function.gt.desired_speed, loss_function.pred.desired_speed, "Desired Speed"
                )
            },
            epoch,
        )
        wandb.log(
            {
                f"viz_{dataset_name}/speed_error": scatter_plot(
                    loss_function.gt.speed_error, loss_function.pred.speed_error, "Speed Error"
                )
            },
            epoch,
        )
        wandb.log(
            {
                f"viz_{dataset_name}/cte": scatter_plot(
                    loss_function.gt.cte, loss_function.pred.cte, "Cross Track Error"
                )
            },
            epoch,
        )
        wandb.log(
            {f"viz_{dataset_name}/he": scatter_plot(loss_function.gt.he, loss_function.pred.he, "Heading Error")},
            epoch,
        )

    def _log_model(self, inference_engine, epoch):
        """
        Logs various model parameters, gradients, activations, and weights to Weights and Biases (wandb) for a given epoch.
        Args:
            inference_engine (InferenceEngine): The inference engine containing model parameters, gradients, activations, and weights.
            epoch (int): The current epoch number.
        Logs:
            - Learning rate of the optimizer.
            - Gradient flow plot.
            - Mean and histogram of gradients for each layer.
            - Mean and histogram of activations for each layer.
            - Mean and histogram of weights for each layer.
        """
        wandb.log({f"opt_train/lr": inference_engine.optimizer.param_groups[0]["lr"]}, step=epoch)
        wandb.log({f"grad_flow": plot_grad_flow(inference_engine.gradients)}, epoch)
        for name, gradients in inference_engine.gradients.items():
            if name.startswith("BatchNorm"):
                continue
            gradients = gradients.flatten()
            wandb.log({f"grad/{name}": wandb.Histogram(gradients)}, epoch)
            wandb.log({f"grad_norm/{name}": np.linalg.norm(gradients)}, epoch)
        for name, activations in inference_engine.activations.items():
            if name.startswith("BatchNorm"):
                continue
            activations = activations.reshape(activations.shape[0], -1)
            wandb.log({f"activation/{name}": wandb.Histogram(activations)}, epoch)
            wandb.log({f"activation_norm/{name}": wandb.Histogram(np.linalg.norm(activations, axis=1))}, epoch)
        for name, weights in inference_engine.weights.items():
            if name.startswith("BatchNorm"):
                continue
            weights = weights.flatten()
            wandb.log({f"weight/{name}": wandb.Histogram(weights)}, epoch)
            wandb.log({f"weight_norm/{name}": np.linalg.norm(weights)}, epoch)

    def log_open_loop(self, inference_engine, epoch, training_mode: bool):
        """
        Logs the open loop performance of the inference engine.
        This method collects garbage, determines the dataset name based on the training mode,
        and logs losses, predictions, and model information if WANDB logging is enabled.
        Args:
            inference_engine: The inference engine containing the model and loss function.
            epoch (int): The current epoch number.
            training_mode (bool): Flag indicating whether the model is in training mode.
        Returns:
            None
        """
        gc.collect()  # Collect garbage
        loss_function = inference_engine.loss_function
        if training_mode:
            dataset_name = "train"
        else:
            dataset_name = "test"
        if not self.conf.IMITATION_WANDB_LOG:
            return
        self._log_losses(dataset_name, loss_function, epoch)
        self._log_predictions(dataset_name, loss_function, epoch)
        if training_mode:
            self._log_model(inference_engine, epoch)

    def checkpoint(self, epoch, validate_reward):
        """
        Checkpoint the model based on validation reward.
        This method checks if the current validation reward is better than the
        best validation reward seen so far. If it is, it updates the best
        validation reward and uploads the model. If the validation reward is
        above a minimum threshold, it also uploads the model.
        Args:
            epoch (int): The current epoch number.
            validate_reward (float): The reward obtained from validation.
        """
        if validate_reward > self.best_validate_reward:
            self.best_validate_reward = validate_reward
            self._upload_model(epoch, validate_reward)
        elif validate_reward >= self.conf.IMITATION_MIN_THRESHOLD_UPLOAD:
            self._upload_model(epoch, validate_reward)
        wandb.log({"best_validate_reward": self.best_validate_reward}, step=epoch)

    def _upload_model(self, epoch, validate_reward):
        """
        Saves the model's state dictionary to a local file and optionally uploads it to Weights & Biases (wandb).
        Args:
            epoch (int): The current epoch number.
            validate_reward (float): The validation reward achieved by the model.
        Returns:
            None
        """
        model_save_file = self.model_save_file_prefix + f"{epoch}_{int(validate_reward)}.pth"
        torch.save(self.model.state_dict(), model_save_file)  # Save model to local disk
        if self.conf.IMITATION_WANDB_LOG:  # Save model to wandb
            print(f"Best model saved at epoch {epoch + 1} with validate reward: {validate_reward:.2f}")
            artifact_name = f"{self.model_name}_{epoch}_{int(validate_reward)}"
            wandb_model_artifact = wandb.Artifact(
                artifact_name, type="model", metadata={"validate_reward": validate_reward}
            )
            wandb_model_artifact.add_file(model_save_file)
            wandb.log_artifact(wandb_model_artifact)
