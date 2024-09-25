from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.imitation_driver.network import Prediction
from src.imitation_driver.training.preprocess import GroundTruth


def cos_sim(x, y):
    return nn.CosineSimilarity(dim=0)(x.flatten(), y.flatten()).item()


class Loss:
    def __init__(self, conf):
        self.conf = conf
        # Loss history
        self.losses = []
        self.steering_losses = []
        self.acceleration_losses = []
        self.road_segmentation_losses = []
        self.chevron_segmentation_losses = []
        self.curvature_losses = []
        self.desired_speed_losses = []
        self.speed_error_losses = []
        self.cte_losses = []
        self.he_losses = []

        # Cosine similarity between ground truth and prediction
        self.steering_cosine_similarity = []
        self.acceleration_cosine_similarity = []
        self.curvature_cosine_similarity = []
        self.desired_speed_cosine_similarity = []
        self.speed_error_cosine_similarity = []
        self.cte_cosine_similarity = []
        self.he_cosine_similarity = []

        # Actual data
        self.gt: GroundTruth = None  # type: ignore
        self.pred: Prediction = None  # type: ignore

    def _steering_loss(self):
        # Steering loss
        self.steering_loss = (self.gt.steering.flatten() - self.pred.steering.flatten()).abs().mean()
        self.steering_losses.append(self.steering_loss.item())
        self.steering_cosine_similarity.append(cos_sim(self.gt.steering, self.pred.steering))
        return self.steering_loss

    def _acceleration_loss(self):
        # Acceleration loss
        self.acceleration_loss = (self.gt.acceleration.flatten() - self.pred.acceleration.flatten()).abs().mean()
        self.acceleration_losses.append(self.acceleration_loss.item())
        self.acceleration_cosine_similarity.append(cos_sim(self.gt.acceleration, self.pred.acceleration))
        return self.acceleration_loss

    def _chevron_segmentation_loss(self):
        # Chevrons segmentation loss
        self.chevron_segmentation_loss = F.binary_cross_entropy_with_logits(
            weight=self.gt.chevron_mask_weight.flatten(),
            target=self.gt.chevron_mask.flatten(),
            input=self.pred.chevron_mask.flatten(),
        )
        self.chevron_segmentation_losses.append(self.chevron_segmentation_loss.item())
        return self.chevron_segmentation_loss

    def _road_segmentation_loss(self):
        # Road segmentation loss
        self.road_segmentation_loss = F.binary_cross_entropy_with_logits(
            weight=self.gt.road_mask_weight.flatten(),
            target=self.gt.road_mask.flatten(),
            input=self.pred.road_mask.flatten(),
        )
        self.road_segmentation_losses.append(self.road_segmentation_loss.item())
        return self.road_segmentation_loss

    def _curvature_loss(self):
        # Curvature loss
        self.curvature_loss = (self.gt.curvature.flatten() - self.pred.curvature.flatten()).abs().mean()
        self.curvature_losses.append(self.curvature_loss.item())
        self.curvature_cosine_similarity.append(cos_sim(self.gt.curvature, self.pred.curvature))
        return self.curvature_loss

    def _desired_speed_loss(self):
        # Desired speed loss
        self.desired_speed_loss = (self.gt.desired_speed.flatten() - self.pred.desired_speed.flatten()).abs().mean()
        self.desired_speed_losses.append(self.desired_speed_loss.item())
        self.desired_speed_cosine_similarity.append(cos_sim(self.gt.desired_speed, self.pred.desired_speed))
        return self.desired_speed_loss

    def _speed_error_loss(self):
        # Speed error loss
        self.speed_error_loss = (self.gt.speed_error.flatten() - self.pred.speed_error.flatten()).abs().mean()
        self.speed_error_losses.append(self.speed_error_loss.item())
        self.speed_error_cosine_similarity.append(cos_sim(self.gt.speed_error, self.pred.speed_error))
        return self.speed_error_loss

    def _cte_loss(self):
        # CTE loss
        self.cte_loss = (self.gt.cte.flatten() - self.pred.cte.flatten()).abs().mean()
        self.cte_losses.append(self.cte_loss.item())
        self.cte_cosine_similarity.append(cos_sim(self.gt.cte, self.pred.cte))
        return self.cte_loss

    def _he_loss(self):
        # HE loss
        self.he_loss = (self.gt.he.flatten() - self.pred.he.flatten()).abs().mean()
        self.he_losses.append(self.he_loss.item())
        self.he_cosine_similarity.append(cos_sim(self.gt.he, self.pred.he))
        return self.he_loss

    def get_loss(self):
        self.loss = torch.tensor(self.conf.IMITATION_STEERING_LOSS).to(self.conf.DEVICE) * self._steering_loss()
        self.loss += torch.tensor(self.conf.IMITATION_ACCELERATION_LOSS).to(self.conf.DEVICE) * self._acceleration_loss()
        self.loss += torch.tensor(self.conf.IMITATION_CHEVRON_SEGMENTATION_LOSS).to(self.conf.DEVICE) * self._chevron_segmentation_loss()
        self.loss += torch.tensor(self.conf.IMITATION_ROAD_SEGMENTATION_LOSS).to(self.conf.DEVICE) * self._road_segmentation_loss()
        self.loss += torch.tensor(self.conf.IMITATION_CURVATURE_LOSS).to(self.conf.DEVICE) * self._curvature_loss()
        self.loss += torch.tensor(self.conf.IMITATION_DESIRED_SPEED_LOSS).to(self.conf.DEVICE) * self._desired_speed_loss()
        self.loss += torch.tensor(self.conf.IMITATION_SPEED_ERROR_LOSS).to(self.conf.DEVICE) * self._speed_error_loss()
        self.loss += torch.tensor(self.conf.IMITATION_CTE_LOSS).to(self.conf.DEVICE) * self._cte_loss()
        self.loss += torch.tensor(self.conf.IMITATION_HE_LOSS).to(self.conf.DEVICE) * self._he_loss()
        self.losses.append(self.loss.item())

        return self.loss
