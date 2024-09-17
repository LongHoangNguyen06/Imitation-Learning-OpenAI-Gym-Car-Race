from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from dynaconf import Dynaconf

conf = Dynaconf(settings_files=["src/conf/default_conf.py"])


def plot_grad_flow(avg_grads):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    layer_names = list(avg_grads.keys())
    avg_grads_values = list(avg_grads.values())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(25, 25))

    # Create boxplots
    avg_box = ax.boxplot(avg_grads_values, positions=range(0, len(layer_names) * 2, 2), widths=0.6, patch_artist=True)

    # Set colors for the boxplots
    for box in avg_box["boxes"]:
        box.set(color="blue", linewidth=2)

    # Set the x-axis labels
    ax.set_xticks([i for i in range(0, len(layer_names) * 2, 2)])
    ax.set_xticklabels(layer_names, rotation=45, ha="right")

    # Set the labels and title
    ax.set_xlabel("Layers")
    ax.set_ylabel("Gradient Values")
    ax.set_title("Gradient Flow Across Layers")
    ax.set_ylim((0.0, 0.03))
    return fig


def plot_observation_grid(observation, action, weight, rows=5, cols=5):
    """
    Plots a grid of the observation tensor.

    Args:
    - observation (torch.Tensor): Input tensor of shape (B x 1 x H x W).
    - action (torch.Tensor): Input tensor of shape (B x 2).
    - weight (torch.Tensor): Input tensor of shape (B x 1).
    - rows (int): Maximum number of rows in the grid (instances from the batch). Default is 10.
    - cols (int): Maximum number of columns in the grid (channels from each instance). Default is 10.

    Returns:
    - None. Displays the plot.
    """

    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)
    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(weight)

    observation = observation.cpu().detach().numpy()
    action = action.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()

    # Ensure observation tensor is 4D
    assert observation.ndim == 4, "Input observations must be 4D (B x 1 x H x W)"
    observation = (observation * conf.OBS_STD + conf.OBS_MEAN).astype(np.int32)
    observation = np.transpose(observation, (0, 2, 3, 1))

    # Plot the grid
    fig, axs = plt.subplots(rows, cols, figsize=(25, 25))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.5, hspace=0.5)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= observation.shape[0]:
                break
            axs[i, j].imshow(observation[idx])  # Assuming grayscale for each observation
            axs[i, j].axis("off")
            axs[i, j].set_title(
                f"Steer={action[idx, 0].item():.2f}, Accl={action[idx, 1].item():.2f}, Wt={weight[idx, 0].item():.2f}"
            )

    # Convert the figure to a numpy array
    fig.canvas.draw()
    img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
    img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close the figure
    plt.close(fig)

    return img_np


def plot_tensor_grid(tensor, max_rows=10, max_cols=10):
    """
    Plots a grid of the input tensor.

    Args:
    - tensor (torch.Tensor): Input tensor of shape (B x C x H x W).
    - max_rows (int): Maximum number of rows in the grid (instances from the batch). Default is 10.
    - max_cols (int): Maximum number of columns in the grid (channels from each instance). Default is 10.

    Returns:
    - None. Displays the plot.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    tensor = tensor.cpu().detach()
    # Ensure tensor is 4D
    assert tensor.ndim == 4, "Input tensor must be 4D (B x C x H x W)"

    # Normalize the tensor to [0, 1]
    tensor_min = tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    tensor_max = tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    # Convert to 0-255 range and uint8
    tensor = (tensor * 255).int().numpy()

    # Plot the grid
    fig, axs = plt.subplots(max_rows, max_cols, figsize=(20, 20))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.5, hspace=0.5)

    for i in range(min(tensor.shape[0], max_rows)):
        for j in range(min(tensor.shape[1], max_cols)):
            axs[i, j].imshow(tensor[i, j], cmap="gray")  # Assuming grayscale for each channel
            axs[i, j].axis("off")

    # Convert the figure to a numpy array
    fig.canvas.draw()
    img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
    img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close the figure
    plt.close(fig)

    return img_np


def plot_predicted_action_and_actual_action(predicted_action, actual_action):
    """
    Plots all pairwise combinations of predicted and actual steering and acceleration
    in a 4x4 grid. The diagonal will be self-plots, but symmetry could be ignored.

    Args:
    - predicted_action (torch.Tensor): Predicted action tensor of shape (B x 2).
    - actual_action (torch.Tensor): Actual action tensor of shape (B x 2).

    Returns:
    - img_np (np.ndarray): Image as a numpy array for further processing.
    """
    if not isinstance(predicted_action, torch.Tensor):
        predicted_action = torch.tensor(predicted_action)
    if not isinstance(actual_action, torch.Tensor):
        actual_action = torch.tensor(actual_action)

    predicted_action = predicted_action.cpu().detach().numpy().flatten()
    actual_action = actual_action.cpu().detach().numpy().flatten()
    # Set up the 4x4 grid for plotting
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(predicted_action, actual_action, alpha=0.6)
    plt.xlim([-1.1, 1.1])  # Set x-axis limit
    plt.ylim([-1.1, 1.1])  # Set y-axis limit
    plt.xlabel("Prediction")
    plt.ylabel("Actual")

    # Convert the figure to a numpy array
    fig.canvas.draw()
    img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
    img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close the figure
    plt.close(fig)

    return img_np


if __name__ == "__main__":
    plt.imshow(plot_predicted_action_and_actual_action(np.random.rand(10, 2), np.random.rand(10, 2)))
    plt.show()
