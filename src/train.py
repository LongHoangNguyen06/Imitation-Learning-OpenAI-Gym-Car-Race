from __future__ import annotations

import warnings

from tqdm import tqdm

# Ignore all warnings
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
# isort:maintain_block
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import shutil
import socket
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from dynaconf import Dynaconf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

import wandb
from src.imitation_driver import network
from src.training.dagger import DaggerLoop, keep_record_probability, teacher_action_probability
from src.training.trainer import Trainer
from src.training.validator import Validator
from src.utils import io_utils
from src.utils.dataset import SequenceDataset, StateDataset, sequence_collate_fn
from src.utils.utils import set_deterministic

# isort:end_maintain_block
conf = Dynaconf(settings_files=["src/conf/default_conf.py"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_deterministic()


def get_model():
    """
    Returns the model, optimizer, and scheduler for training a racecar.
    Returns:
        model (Network): The initialized model for training.
        optimizer (torch.optim.Adam): The optimizer for updating the model parameters.
        scheduler (torch.optim.lr_scheduler.StepLR): The learning rate scheduler for the optimizer.
    """
    if conf.DO_MULTITASK_LEARNING:
        model = network.MultiTaskCNN(print_shapes=True).to(device).double().share_memory()
    else:
        model = network.SingleTaskCNN(print_shapes=True).to(device).double().share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.IMITATION_LR)  # type: ignore
    if conf.DAGGER_TRAINING:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=conf.DAGGER_LR_SCHEDULER_STEP_SIZE, gamma=conf.IMITATION_LR_SCHEDULER_GAMMA
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=conf.IMITATION_LR_SCHEDULER_STEP_SIZE, gamma=conf.IMITATION_LR_SCHEDULER_GAMMA
        )
    return model, [optimizer], [scheduler]


def get_code_artifact_dir():
    """
    Returns the directory path for storing code artifacts based on the current hostname.
    :return: The directory path for code artifacts.
    :rtype: str
    """
    return os.path.join(conf.OUTPUT_DIR, socket.gethostname())


def get_tensorboard_writer(run_id):
    """
    Returns a SummaryWriter object for TensorBoard logging.
    Parameters:
        run_id (str): The unique identifier for the current run.
    Returns:
        SummaryWriter: The SummaryWriter object for logging.
    """
    if not conf.TENSOR_BOARD_LOG:
        return None
    run_dir = os.path.join(conf.TENSOR_BOARD_DIR, run_id)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=run_dir)

    # Save code artifact
    code_artifact_dir = get_code_artifact_dir()
    if os.path.exists(get_code_artifact_dir()):
        src = code_artifact_dir
        dst = os.path.join(run_dir, socket.gethostname())
        shutil.copytree(src, dst)
    return summary_writer


def initialize_wandb(run_id):
    """
    Initializes the wandb (Weights and Biases) logging for the CarRace project.
    Parameters:
    - run_id (str): The unique identifier for the run.
    Returns:
    None
    """
    if conf.WANDB_LOG:
        wandb.init(project="CarRace", name=f"{run_id}")
        if os.path.exists(get_code_artifact_dir()):
            # Add the code artifact to wandb
            artifact = wandb.Artifact(f"{run_id}_code_artifact", type="code")
            artifact.add_dir(get_code_artifact_dir())
            wandb.log_artifact(artifact)


def get_data_loader(data_dir):
    """
    Returns a generator that yields StateDataset objects from a given data directory.
    Parameters:
    - data_dir (str): The directory path where the data is stored.
    Yields:
    - StateDataset: A StateDataset object containing a batch of sequences.
    """
    sequence_dataset = SequenceDataset(data_dir=data_dir)
    sequence_dataloader = DataLoader(
        sequence_dataset,
        batch_size=conf.SEQUENCE_BATCH_SIZE,
        shuffle=True,
        num_workers=conf.SEQUENCE_NUM_WORKERS,
        collate_fn=sequence_collate_fn,
        prefetch_factor=conf.SEQUENCE_PREFETCH_FACTOR,
        persistent_workers=True,
    )
    for sequences in tqdm(sequence_dataloader, desc="Training"):
        yield DataLoader(
            StateDataset(sequence_dataset_batch=sequences),
            batch_size=conf.STATE_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )


def main():
    """
    Main function for training the autoencoder model.
    This function performs the following steps:
    1. Initializes logging and sets up TensorBoard and WandB.
    2. Configures the device and sets the model output path.
    3. Initializes the model, optimizers, and schedulers.
    4. Creates data loaders for training and testing.
    5. Initializes checkpointing and sets the best validation reward.
    6. Trains the model for the specified number of epochs.
    7. Optionally finishes the WandB run.
    Args:
        None
    Returns:
        None
    """
    run_id: str = io_utils.get_current_time_formatted()

    # Initialize logging
    tensorboard_writer = get_tensorboard_writer(run_id)
    initialize_wandb(run_id)

    # Device configuration
    model_output_path = conf.IMITATION_OUTPUT_MODEL

    # Initialize the model, optimizer, and loss function
    model, optimizers, schedulers = get_model()

    # Initialize checkpointing
    best_validate_reward = float("-inf")
    model_name = f"{run_id}_{model.__class__.__name__}"
    model_save_file_prefix = os.path.join(model_output_path, model_name)
    wandb_model_artifact = None

    # Training loop
    dagger_output_dir = os.path.join(conf.DAGGER_OUTPUT_DIR, run_id)
    dagger_loop = DaggerLoop(dagger_output_dir=dagger_output_dir, student_model=model)
    if conf.DAGGER_TRAINING:
        dagger_loop.dagger_loop(epoch=0)  # Initialize the data
    for epoch in range(conf.DAGGER_EPOCHS if conf.DAGGER_TRAINING else conf.IMITATION_EPOCHS):
        # Training
        model.hook()
        for state_dataloader in get_data_loader(dagger_output_dir if conf.DAGGER_TRAINING else conf.IMITATION_DATA_DIR):
            Trainer(model, state_dataloader, optimizers, tensorboard_writer, epoch).one_step()
        model.unhook()

        # Step the learning rate scheduler
        [s.step() for s in schedulers]

        # Validate with closed loop simulation
        if not conf.DAGGER_TRAINING:
            validate_rewards = Validator(model).validate()
        else:
            dagger_rewards = dagger_loop.dagger_loop(epoch=epoch)
            if teacher_action_probability(epoch) == 0.0:  # If the teacher is not driving at all, then we can save time.
                validate_rewards = dagger_rewards
            else:
                validate_rewards = Validator(model).validate()

        validate_reward = int(np.mean(validate_rewards))  # type: ignore
        # Check if this test loss is the best so far
        if validate_reward > best_validate_reward:
            best_validate_reward = validate_reward

            model_save_file_best = model_save_file_prefix + f"_{int(validate_reward)}.pth"
            torch.save(model.state_dict(), model_save_file_best)  # Save model to local disk
            print(f"Best model saved at epoch {epoch + 1} with validate reward: {validate_reward:.2f}")
            if conf.WANDB_LOG:  # Save model to wandb
                if wandb_model_artifact is not None:
                    wandb_model_artifact.delete(delete_aliases=True)
                artifact_name = f"{model_name}_{int(validate_reward)}"
                wandb_model_artifact = wandb.Artifact(
                    artifact_name, type="model", metadata={"validate_reward": validate_reward}
                )
                wandb_model_artifact.add_file(model_save_file_best)
                wandb.log_artifact(wandb_model_artifact)
                wandb_model_artifact.wait()
        if conf.WANDB_LOG:
            wandb.log({f"opt_reward": validate_reward}, step=epoch)
            wandb.log({f"opt_teacher_action_proboability": teacher_action_probability(epoch)}, step=epoch)
            wandb.log({f"opt_keep_record_probability": keep_record_probability(epoch)}, step=epoch)
            wandb.log({f"#_dagger_records": len(os.listdir(dagger_output_dir))}, step=epoch)
        if conf.TENSOR_BOARD_LOG:
            assert tensorboard_writer is not None
            tensorboard_writer.add_scalar(f"opt_reward", validate_reward, epoch)

    if conf.TENSOR_BOARD_LOG:
        assert tensorboard_writer is not None
        tensorboard_writer.close()
    if conf.WANDB_LOG:
        wandb.finish()
        assert wandb_model_artifact is not None
        wandb_model_artifact.save()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
