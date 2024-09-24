from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from src.imitation_driver.network import AbstractNet, Encoder, DenseLayer, Decoder, Prediction, model_size
from src.utils.conf_utils import get_default_conf

conf = get_default_conf()


class MultiTaskCNN(AbstractNet):
    def __init__(self, print_shapes=False, store_debug_states=False):
        super().__init__()
        # Backbone
        self.backbone = Encoder()

        # Auxiliary tasks
        self.mask_decoder = Decoder()
        self.curvature_predictor = DenseLayer(conf.IMITATION_CURVATURE_DIMS)
        self.desired_speed_preditor = DenseLayer(conf.IMITATION_DESIRED_SPEED_DIMS)
        self.cte_predictor = DenseLayer(conf.IMITATION_CTE_DIMS)
        self.he_predictor = DenseLayer(conf.IMITATION_HE_DIMS)

        # Control tasks
        self.steering_predictor = DenseLayer(conf.IMITATION_STEERING_DIMS)
        self.acceleration_predictor = DenseLayer(conf.IMITATION_ACCELERATION_DIMS)

        # Sequential model (almost)
        self.seq = nn.Sequential(
            self.backbone,
            self.mask_decoder,
            self.curvature_predictor,
            self.desired_speed_preditor,
            self.cte_predictor,
            self.he_predictor,
            self.steering_predictor,
            self.acceleration_predictor,
        )

        # Activation function
        self.debug_states = defaultdict(list)
        self.store_debug_states = store_debug_states
        with torch.no_grad():
            self.init_weights(print_shapes)

    def reset(self):
        self.debug_states = defaultdict(list)

    def forward(self, observation, state, *args, **kwargs):
        # Back bone forward pass
        code = self.backbone(observation)
        if self.store_debug_states:
            self.debug_states["code"].append(code.detach().cpu().numpy())

        # Masks
        mask = self.mask_decoder(code)
        predicted_chevron_mask = mask[:, 0, ...].reshape(-1, *conf.MASK_DIM)
        predicted_road_mask = mask[:, 1, ...].reshape(-1, *conf.MASK_DIM)
        if self.store_debug_states:
            self.debug_states["road_mask_prediction"].append(predicted_road_mask.detach().cpu().numpy())
            self.debug_states["chevron_mask_prediction"].append(predicted_chevron_mask.detach().cpu().numpy())

        # Curvature
        code = code.flatten(start_dim=1)
        predicted_curvature = self.curvature_predictor(code)
        if self.store_debug_states:
            self.debug_states["curvature_prediction"].append(predicted_curvature.detach().cpu().numpy())

        if self.training:
            curvature = kwargs["curvature"]  # Teacher forcing
        else:
            curvature = predicted_curvature

        # Desired speed
        predicted_desired_speed = self.desired_speed_preditor(torch.cat((curvature, state), dim=1))
        if self.store_debug_states:
            self.debug_states["desired_speed_prediction"].append(predicted_desired_speed.detach().cpu().numpy())

        if self.training:
            desired_speed = kwargs["desired_speed"]
        else:
            desired_speed = predicted_desired_speed

        # Speed error
        unnormalized_desired_speed = (predicted_desired_speed * 300.0).flatten()  # see preprocess.py
        unnormalized_speed = state[:, 0] * 100  # see replay.py
        predicted_speed_error = (unnormalized_desired_speed - unnormalized_speed).reshape(-1, 1) / 300.0

        if self.store_debug_states:
            self.debug_states["speed_error_prediction"].append(predicted_speed_error.detach().cpu().numpy())

        if self.training:
            speed_error = kwargs["speed_error"]
        else:
            speed_error = predicted_speed_error

        # Acceleration
        predicted_acceleration = F.softsign(
            self.acceleration_predictor(torch.cat((code, state, curvature, desired_speed, speed_error), dim=1))
        )
        if self.store_debug_states:
            self.debug_states["acceleration_prediction"].append(predicted_acceleration.detach().cpu().numpy())

        # Cross track error and heading error
        predicted_cte = self.cte_predictor(torch.cat((code, state), dim=1))
        predicted_he = self.he_predictor(torch.cat((code, state), dim=1))
        if self.store_debug_states:
            self.debug_states["cte_prediction"].append(predicted_cte.detach().cpu().numpy())
            self.debug_states["he_prediction"].append(predicted_he.detach().cpu().numpy())

        if self.training:
            cte = kwargs["cte"]
            he = kwargs["he"]
        else:
            cte = predicted_cte
            he = predicted_he

        # Steering
        predicted_steering = F.softsign(self.steering_predictor(torch.cat((code, state, curvature, cte, he), dim=1)))
        if self.store_debug_states:
            self.debug_states["steering_prediction"].append(predicted_steering.detach().cpu().numpy())

        # Return the predictions
        return Prediction(
            road_mask=predicted_road_mask,
            chevron_mask=predicted_chevron_mask,
            curvature=predicted_curvature,
            desired_speed=predicted_desired_speed,
            steering=predicted_steering,
            acceleration=predicted_acceleration,
            speed_error=predicted_speed_error,
            cte=predicted_cte,
            he=predicted_he,
        )

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
    MultiTaskCNN(print_shapes=True)


if __name__ == "__main__":
    main()
