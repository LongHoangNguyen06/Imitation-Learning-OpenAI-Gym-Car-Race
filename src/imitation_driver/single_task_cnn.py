from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from src.imitation_driver.network import AbstractNet, Prediction


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    else:
        raise NotImplementedError(f"Initialization for {m.__class__.__name__} is not implemented.")


class SingleTaskCNN(AbstractNet):
    def __init__(self, conf, print_shapes=False, store_debug_states=False):
        super().__init__(conf=conf)
        self.conf = conf
        # Features extraction
        self.conv = nn.Sequential(
            nn.Sequential(
                nn.LazyConv2d(conf.IMITATION_NUM_FILTERS_ENCODER // 2, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(conf.IMITATION_NUM_FILTERS_ENCODER // 2, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(conf.IMITATION_NUM_FILTERS_ENCODER, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.LazyConv2d(conf.IMITATION_NUM_FILTERS_ENCODER, kernel_size=3, stride=1, padding=1),
                nn.LazyBatchNorm2d(),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
        )

        # Dense head
        fc = []
        for i in range(conf.IMITATION_FC_NUM_LAYERS):
            fc.append(
                nn.Sequential(
                    nn.LazyLinear(
                        out_features=conf.IMITATION_FC_INITIAL_LAYER_SIZE // (i + 1),
                    ),
                    nn.LeakyReLU(),
                    nn.Dropout(p=conf.IMITATION_DROPOUT_PROB),
                )
            )
        fc.append(nn.Sequential(nn.LazyLinear(2), nn.Softsign()))

        # Concatenate everything together
        self.fc = nn.Sequential(*fc)
        self.seq = nn.Sequential(self.conv, self.fc)
        self.debug_states = defaultdict(list)
        self.store_debug_states = store_debug_states
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
        if self.store_debug_states:
            self.debug_states["steering_prediction"].append(steering.detach().cpu().numpy())
        if self.store_debug_states:
            self.debug_states["acceleration_prediction"].append(acceleration.detach().cpu().numpy())

        return Prediction(
            road_mask=torch.zeros((batch_size, *self.conf.MASK_DIM), device=observation.device),
            chevron_mask=torch.zeros((batch_size, *self.conf.MASK_DIM), device=observation.device),
            curvature=torch.zeros(batch_size, 1, device=observation.device),
            steering=steering,
            acceleration=acceleration,
            desired_speed=torch.zeros(batch_size, 1, device=observation.device),
            speed_error=torch.zeros(batch_size, 1, device=observation.device),
            cte=torch.zeros(batch_size, 1, device=observation.device),
            he=torch.zeros(batch_size, 1, device=observation.device),
        )

    def hook(self):
        self._hook(self.seq)
        self.debug_states = defaultdict(list)

    def print_shape(self):
        print(f"Single task network")
        self._print_shape(self.seq)
