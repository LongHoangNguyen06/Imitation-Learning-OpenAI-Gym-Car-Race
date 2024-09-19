from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.conf_utils import get_default_conf

conf = get_default_conf()


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
    def __init__(self):
        """
        Initializes the net class.
        """
        super().__init__()

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
        self.eval()
        self(torch.zeros(1, *conf.OBSERVATION_DIM), torch.zeros(1, conf.STATE_DIM))
        self.hook()
        with torch.no_grad():
            self(torch.zeros(1, *conf.OBSERVATION_DIM), torch.zeros(1, conf.STATE_DIM))
        if print_shapes:
            self.print_shape()
        self.unhook()


class Encoder(AbstractNet):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Sequential(
                nn.LazyConv2d(out_channels=conf.IMITATION_NUM_FILTERS_ENCODER // 2, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.Dropout(p=conf.IMITATION_DROPOUT_PROB),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(out_channels=conf.IMITATION_NUM_FILTERS_ENCODER // 2, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.Dropout(p=conf.IMITATION_DROPOUT_PROB),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(conf.IMITATION_NUM_FILTERS_ENCODER, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.Dropout(p=conf.IMITATION_DROPOUT_PROB),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(out_channels=conf.IMITATION_NUM_FILTERS_ENCODER, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.Dropout(p=conf.IMITATION_DROPOUT_PROB),
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
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.Sequential(
            nn.LazyConvTranspose2d(
                out_channels=conf.IMITATION_NUM_FILTERS_ENCODER,
                kernel_size=3,
                stride=2,
                output_padding=(0, 0),
                padding=(0, 1),
            ),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(),
            nn.Dropout(p=conf.IMITATION_DROPOUT_PROB),
        )
        self.deconv2 = nn.Sequential(
            nn.LazyConvTranspose2d(
                out_channels=conf.IMITATION_NUM_FILTERS_ENCODER,
                kernel_size=3,
                stride=2,
                output_padding=(0, 0),
                padding=(1, 0),
            ),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(),
            nn.Dropout(p=conf.IMITATION_DROPOUT_PROB),
        )
        self.deconv3 = nn.Sequential(
            nn.LazyConvTranspose2d(
                out_channels=conf.IMITATION_NUM_FILTERS_ENCODER // 2,
                kernel_size=3,
                stride=2,
                output_padding=(1, 0),
                padding=(1, 0),
            ),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(),
            nn.Dropout(p=conf.IMITATION_DROPOUT_PROB),
        )
        self.deconv4 = nn.Sequential(
            nn.LazyConvTranspose2d(
                out_channels=1,
                kernel_size=3,
                stride=2,
                output_padding=(1, 1),
                padding=(1, 0),
            )
        )
        self.seq = nn.Sequential(self.deconv1, self.deconv2, self.deconv3, self.deconv4)

    def forward(self, code):
        code = code.reshape(-1, conf.IMITATION_NUM_FILTERS_ENCODER, 5, 6)  # Hard code the shape
        return self.seq(code)

    def hook(self):
        self._hook(self.seq)

    def print_shape(self):
        self._print_shape(self.seq)


class StatelessDensePredictor(AbstractNet):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.LazyLinear(dims[i]))
            if i < len(dims) - 2:  # No activation or dropout on the final layer
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(p=conf.IMITATION_DROPOUT_PROB))
        layers.append(nn.LazyLinear(dims[-1]))  # Add the final layer
        self.seq = nn.Sequential(*layers)

    def forward(self, code):
        return self.seq(code)

    def hook(self):
        self._hook(self.seq)

    def print_shape(self):
        self._print_shape(self.seq)


class StatefulDensePredictor(AbstractNet):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = nn.Sequential(
                nn.LazyLinear(out_features=dims[i]), nn.LeakyReLU(), nn.Dropout(p=conf.IMITATION_DROPOUT_PROB)
            )
            self.layers.append(layer)
        self.final_layer = nn.LazyLinear(out_features=dims[-1])  # Add the final layer
        self.seq = nn.Sequential(*self.layers, self.final_layer)

    def forward(self, code, state):
        for layer in self.layers:
            code = layer(torch.cat((code, state), dim=1))
        return self.final_layer(code)

    def hook(self):
        self._hook(self.seq)

    def print_shape(self):
        self._print_shape(self.seq)


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    else:
        raise NotImplementedError(f"Initialization for {m.__class__.__name__} is not implemented.")


class SingleTaskCNN(AbstractNet):
    def __init__(self, print_shapes=False):
        super().__init__()
        # Network architecture
        super().__init__()
        self.conv = nn.Sequential(
            nn.Sequential(
                nn.LazyConv2d(32, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(32, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
        )
        self.fc = nn.Sequential(
            nn.Sequential(nn.LazyLinear(64), nn.LeakyReLU(), nn.Dropout(p=conf.IMITATION_DROPOUT_PROB)),
            nn.Sequential(nn.LazyLinear(32), nn.LeakyReLU(), nn.Dropout(p=conf.IMITATION_DROPOUT_PROB)),
            nn.Sequential(nn.LazyLinear(2), nn.Softsign()),
        )
        self.seq = nn.Sequential(self.conv, self.fc)
        self.debug_states = defaultdict(list)
        self.init_weights(print_shapes=print_shapes)

    def reset(self):
        self.debug_states = defaultdict(list)

    def init_weights(self, print_shapes):
        super().init_weights(print_shapes)
        for m in self.conv:
            init_weight(m[0])  # type: ignore
        for f in self.fc:
            init_weight(f[0])  # type: ignore

    def forward(self, observation, state, *args, **kwargs):
        """
        Forward pass of the CNN model.
        Args:
            observation (torch.Tensor): The input observation.
            state (torch.Tensor): The input state.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            torch.Tensor: The output of the forward pass.
        """
        # Actual computation
        feature_map = self.conv(observation)
        feature_map = feature_map.view(feature_map.size(0), -1)
        enriched_feature_map = torch.cat((feature_map, state), dim=1)
        outputs = self.fc(enriched_feature_map)
        steering = outputs[:, 0].reshape(-1, 1)
        acceleration = outputs[:, 1].reshape(-1, 1)
        batch_size = observation.size(0)
        return (
            torch.zeros((batch_size, *conf.MASK_DIM), device=observation.device),
            torch.zeros((batch_size, *conf.MASK_DIM), device=observation.device),
            torch.zeros(batch_size, 1, device=observation.device),
            steering,
            acceleration,
        )

    def hook(self):
        self._hook(self.seq)
        self.debug_states = defaultdict(list)

    def print_shape(self):
        print(f"Single task network")
        self._print_shape(self.seq)


class MultiTaskCNN(AbstractNet):
    def __init__(self, print_shapes=False, store_debug_states = False):
        super().__init__()
        # Backbone
        self.backbone = Encoder()

        # Auxiliary tasks
        self.road_decoder = Decoder()
        self.chevrons_decoder = Decoder()
        self.curvature_predictor = StatelessDensePredictor(conf.IMITATION_CURVATURE_DIMS)

        # Control tasks
        self.steering_predictor = StatefulDensePredictor(conf.IMITATION_STEERING_DIMS)
        self.acceleration_predictor = StatefulDensePredictor(conf.IMITATION_ACCELERATION_DIMS)

        # Sequential model (almost)
        self.seq = nn.Sequential(
            self.backbone,
            self.road_decoder,
            self.chevrons_decoder,
            self.curvature_predictor,
            self.steering_predictor,
            self.acceleration_predictor,
        )

        # Activation function
        self.debug_states = defaultdict(list)
        self.store_debug_states = store_debug_states
        self.init_weights(print_shapes)

    def reset(self):
        self.debug_states = defaultdict(list)

    def forward(self, observation, state):
        # Back bone forward pass
        code = self.backbone(observation)
        if self.store_debug_states:
            self.debug_states["code_history"].append(code.detach().cpu().numpy())

        # Auxiliary tasks
        road_mask = self.road_decoder(code)
        if self.store_debug_states:
            self.debug_states["road_mask_prediction_history"].append(road_mask.detach().cpu().numpy())
        chevrons_mask = self.chevrons_decoder(code)
        if self.store_debug_states:
            self.debug_states["chevrons_mask_prediction_history"].append(chevrons_mask.detach().cpu().numpy())
        code = code.flatten(start_dim=1)
        curvature = self.curvature_predictor(code)
        if self.store_debug_states:
            self.debug_states["curvature_prediction_history"].append(curvature.detach().cpu().numpy())

        # Control tasks
        steering = F.softsign(self.steering_predictor(code, state))
        if self.store_debug_states:
            self.debug_states["steering_prediction_history"].append(steering.detach().cpu().numpy())
        acceleration = F.softsign(self.acceleration_predictor(code, state))
        if self.store_debug_states:
            self.debug_states["acceleration_prediction_history"].append(acceleration.detach().cpu().numpy())
        return road_mask, chevrons_mask, curvature, steering, acceleration

    def hook(self):
        for module in self.seq:
            module.hook()

    def unhook(self):
        for module in self.seq:
            module.unhook()

    def print_shape(self):
        sep = "#" * 75
        print(sep)
        for name, module in self.named_children():
            if isinstance(module, AbstractNet):
                print(name)
                module.print_shape()
                print(sep)
        print("Total model size: ", model_size(self))


def main():
    SingleTaskCNN(print_shapes=True)


if __name__ == "__main__":
    main()
