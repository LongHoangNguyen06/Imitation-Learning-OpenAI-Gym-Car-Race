from __future__ import annotations

import warnings

from tqdm import tqdm

from src.training.checkpoint import Checkpoint
from utils.conf_utils import get_default_conf

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

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

import wandb
from src.imitation_driver import network
from src.training.dagger_loop import DaggerLoop, teacher_action_probability
from src.training.dataset import SequenceDataset, StateDataset, sequence_collate_fn
from src.training.one_epoch_trainer import Trainer
from src.utils import io_utils
from src.utils.utils import set_deterministic

# isort:end_maintain_block
conf = get_default_conf()
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


def get_data_loader(data_dir, epoch):
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
        prefetch_factor=conf.SEQUENCE_PREFETCH_FACTOR,
        persistent_workers=True,
        collate_fn=sequence_collate_fn,
    )
    if conf.WANDB_LOG:
        wandb.log({"training_dataset/#sequences_epoch": len(sequence_dataset.record_files)}, step=epoch)
        wandb.log({"training_dataset/seeds_epoch": wandb.Histogram(sequence_dataset.seeds)}, step=epoch)
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

    # Initialize the model, optimizer, and loss function
    model, optimizers, schedulers = get_model()

    # Initialize checkpointing
    checkpoint = Checkpoint(model, run_id)

    # Training loop
    dagger_output_dir = os.path.join(conf.DAGGER_OUTPUT_DIR, run_id)
    dagger_loop = DaggerLoop(output_dir=dagger_output_dir, student_model=model)
    dagger_loop.dagger_loop(epoch=0)  # Initialize the data

    # Actual training
    for epoch in range(conf.IMITATION_EPOCHS):
        # Training
        for state_dataloader in get_data_loader(dagger_output_dir, epoch=epoch):
            Trainer(model, state_dataloader, optimizers, tensorboard_writer, epoch).one_step()

        # Step the learning rate scheduler
        [s.step() for s in schedulers]

        # Validate with closed loop simulation
        validate_reward = dagger_loop.dagger_loop(epoch=epoch)
        if teacher_action_probability(epoch) == 0.0:
            checkpoint.checkpoint(epoch=epoch, validate_reward=validate_reward)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
