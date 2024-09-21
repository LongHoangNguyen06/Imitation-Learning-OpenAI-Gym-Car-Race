from __future__ import annotations

import numpy as np

import wandb


def plot_grad_flow(avg_grads):
    """Plots the gradients flowing through different layers in the net during training using wandb.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backward() as
    "plot_grad_flow_wandb(self.model.named_parameters())" to visualize the gradient flow with wandb"""

    layer_names = list(avg_grads.keys())
    avg_grads_values = np.array(list(avg_grads.values())).flatten()

    # Create a data table for wandb
    data = [[label, val] for (label, val) in zip(layer_names, avg_grads_values)]
    table = wandb.Table(data=data, columns=["Layer", "Gradient Value"])
    return wandb.plot.bar(table, "Layer", "Gradient Value", title="Gradient Flow Across Layers") # type: ignore
