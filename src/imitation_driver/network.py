from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch
from torch import nn


def model_size(model):
    """
    Calculates the total number of trainable parameters in the given model.
    Parameters:
    model (torch.nn.Module): The model for which to calculate the size.
    Returns:
    int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AbstractNet(nn.Module):
    def __init__(self, conf):
        """
        Initializes the net class.
        """
        super().__init__()
        self.conf = conf

        # Hooks
        self.activations = {}
        self.gradients = {}
        self.weights = {}
        self.hook_handles = []

    def _traverse(self, seq: nn.Sequential, callback):
        counter = 0

        def __traverse(sequential):
            nonlocal counter
            for layer in sequential:
                name = f"{layer.__class__.__name__}_{counter}"
                if isinstance(layer, nn.Sequential):
                    __traverse(layer)
                else:
                    callback(layer, name)
                    counter += 1

        __traverse(seq)

    def _print_shape(self, seq: nn.Sequential):
        """
        Initializes the weights of the CNN model.
        Parameters:
        - input_shape (tuple): The shape of the input data.
        Returns:
        None
        """

        def callback(layer: nn.Module, name):
            if hasattr(layer, "weight"):
                print(f"\t{name}.")
                print(
                    f"\t\tParameters: {model_size(layer)}. Output shape: {self.activations[name].shape[1:]}. Output size: {np.prod(self.activations[name].shape)}"
                )

        self._traverse(seq, callback)
        print("\tModel size: ", model_size(self))

    def _hook(self, seq: nn.Sequential):
        """
        Attaches hooks to the convolutional and fully connected layers of the CNN model.
        """

        def callback(layer, name):
            if hasattr(layer, "weight"):
                self.hook_handles.append(layer.register_forward_hook(self._get_activation(name)))
                self.hook_handles.append(layer.register_full_backward_hook(self._get_gradient(name)))
                self.weights[name] = layer.weight.detach().cpu().numpy()

        self._traverse(seq, callback)

    def _get_activation(self, name: str):  # -> Callable[..., None]:
        def hook(model, _, output) -> None:
            self.activations[name] = output.detach().cpu().numpy()
            self.weights[name] = model.weight.detach().cpu().numpy()

        return hook

    def _get_gradient(self, name: str):
        def hook(_, __, grad_output):
            self.gradients[name] = grad_output[0].cpu().numpy()

        return hook

    def print_shape(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def hook(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def unhook(self):
        """
        Removes all hooks and clears the internal state variables.
        This method removes all hooks by calling `remove()` on each handle in `self.hook_handles`.
        It then clears the list of handles.
        """
        # Remove hooks by calling remove() on each handle
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()  # Clear the list of handles
        self.activations.clear()
        self.gradients.clear()
        self.weights.clear()

    def init_weights(self, print_shapes):
        """
        Initializes the weights of the network by performing a forward pass with zero inputs.
        Args:
            print_shapes (bool): If True, prints the shapes of the network layers after initialization.
        This method performs the following steps:
        1. Sets the network to evaluation mode.
        2. Performs a forward pass with zero inputs to initialize the weights.
        3. Attaches hooks to the network layers.
        4. Performs another forward pass with zero inputs without gradient computation.
        5. Optionally prints the shapes of the network layers if `print_shapes` is True.
        6. Removes the hooks from the network layers.
        """
        self.eval()
        self(torch.zeros(1, *self.conf.OBSERVATION_DIM), torch.zeros(1, self.conf.IMITATION_STATE_DIM))
        self.hook()
        with torch.no_grad():
            self(torch.zeros(1, *self.conf.OBSERVATION_DIM), torch.zeros(1, self.conf.IMITATION_STATE_DIM))
        if print_shapes:
            self.print_shape()
        self.unhook()


class Encoder(AbstractNet):
    def __init__(self, conf):
        super().__init__(conf=conf)
        self.conf = conf
        self.seq = nn.Sequential(
            nn.Sequential(
                nn.LazyConv2d(out_channels=self.conf.IMITATION_NUM_FILTERS_ENCODER // 2, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.Dropout(p=self.conf.IMITATION_DROPOUT_PROB),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(out_channels=self.conf.IMITATION_NUM_FILTERS_ENCODER // 2, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.Dropout(p=self.conf.IMITATION_DROPOUT_PROB),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(self.conf.IMITATION_NUM_FILTERS_ENCODER, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.Dropout(p=self.conf.IMITATION_DROPOUT_PROB),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(out_channels=self.conf.IMITATION_NUM_FILTERS_ENCODER, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.Dropout(p=self.conf.IMITATION_DROPOUT_PROB),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
        )

    def forward(self, observation):
        return self.seq(observation)

    def hook(self):
        self._hook(self.seq)

    def print_shape(self):
        self._print_shape(self.seq)


class Decoder(AbstractNet):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf
        self.deconv1 = nn.Sequential(
            nn.LazyConvTranspose2d(
                out_channels=self.conf.IMITATION_NUM_FILTERS_ENCODER // 2,
                kernel_size=3,
                stride=2,
                output_padding=(0, 0),
                padding=(0, 1),
            ),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(),
            nn.Dropout(p=self.conf.IMITATION_DROPOUT_PROB),
        )
        self.deconv2 = nn.Sequential(
            nn.LazyConvTranspose2d(
                out_channels=self.conf.IMITATION_NUM_FILTERS_ENCODER // 2,
                kernel_size=3,
                stride=2,
                output_padding=(0, 0),
                padding=(1, 0),
            ),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(),
            nn.Dropout(p=self.conf.IMITATION_DROPOUT_PROB),
        )
        self.deconv3 = nn.Sequential(
            nn.LazyConvTranspose2d(
                out_channels=self.conf.IMITATION_NUM_FILTERS_ENCODER // 4,
                kernel_size=3,
                stride=2,
                output_padding=(1, 0),
                padding=(1, 0),
            ),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(),
            nn.Dropout(p=self.conf.IMITATION_DROPOUT_PROB),
        )
        self.deconv4 = nn.Sequential(
            nn.LazyConvTranspose2d(
                out_channels=2,
                kernel_size=3,
                stride=2,
                output_padding=(1, 1),
                padding=(1, 0),
            )
        )
        self.seq = nn.Sequential(self.deconv1, self.deconv2, self.deconv3, self.deconv4)

    def forward(self, code):
        code = code.reshape(-1, self.conf.IMITATION_NUM_FILTERS_ENCODER, 5, 6)  # Hard code the shape
        return self.seq(code)

    def hook(self):
        self._hook(self.seq)

    def print_shape(self):
        self._print_shape(self.seq)


class DenseLayer(AbstractNet):
    def __init__(self, dims, conf):
        super().__init__(conf=conf)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.LazyLinear(dims[i]))
            if i < len(dims) - 2:  # No activation or dropout on the final layer
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(p=self.conf.IMITATION_DROPOUT_PROB))
        layers.append(nn.LazyLinear(dims[-1]))  # Add the final layer
        self.seq = nn.Sequential(*layers)

    def forward(self, input):
        return self.seq(input)

    def hook(self):
        self._hook(self.seq)

    def print_shape(self):
        self._print_shape(self.seq)


Prediction = namedtuple(
    "Prediction",
    ["road_mask", "chevron_mask", "curvature", "desired_speed", "steering", "acceleration", "speed_error", "cte", "he"],
)
