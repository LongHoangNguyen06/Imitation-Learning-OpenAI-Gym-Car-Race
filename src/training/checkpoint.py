from __future__ import annotations

import os

import torch

import wandb
from src.utils import conf_utils

conf = conf_utils.get_default_conf()


class Checkpoint:
    def __init__(self, model, run_id) -> None:
        self.best_validate_reward = float("-inf")
        self.wandb_model_artifact = None
        self.model_name = f"{run_id}_{model.__class__.__name__}"
        self.model_save_file_prefix = os.path.join(conf.IMITATION_OUTPUT_MODEL, self.model_name)
        self.model = model

    def checkpoint(self, epoch, validate_reward):
        if validate_reward > self.best_validate_reward:
            self.best_validate_reward = validate_reward
            model_save_file_best = self.model_save_file_prefix + f"_{int(validate_reward)}.pth"
            torch.save(self.model.state_dict(), model_save_file_best)  # Save model to local disk
            print(f"Best model saved at epoch {epoch + 1} with validate reward: {validate_reward:.2f}")
            if conf.WANDB_LOG:  # Save model to wandb
                if self.wandb_model_artifact is not None:
                    self.wandb_model_artifact.delete(delete_aliases=True)
                artifact_name = f"{self.model_name}_{int(validate_reward)}"
                wandb_model_artifact = wandb.Artifact(
                    artifact_name, type="model", metadata={"validate_reward": validate_reward}
                )
                wandb_model_artifact.add_file(model_save_file_best)
                wandb.log_artifact(wandb_model_artifact)
                wandb_model_artifact.wait()
