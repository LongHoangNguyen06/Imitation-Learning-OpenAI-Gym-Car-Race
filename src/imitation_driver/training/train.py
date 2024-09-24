from __future__ import annotations

import warnings

from tqdm import tqdm

# Ignore all warnings
warnings.filterwarnings("ignore")
# isort:maintain_block
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import wandb
from src.expert_drivers.pid_driver.pid_driver_controller import PidDriverController
from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from src.imitation_driver.training.dataset import SequenceDataset, StateDataset, sequence_collate_fn
from src.imitation_driver.training.inference_engine import InferenceEngine
from src.imitation_driver.training.wandb_logger import WandbLogger
from src.utils import io_utils
from src.utils.conf_utils import get_default_conf
from src.utils.simulator import Simulator, teacher_action_probability
from src.utils.utils import set_deterministic
from src.imitation_driver.multi_task_cnn import MultiTaskCNN
from src.imitation_driver.single_task_cnn import SingleTaskCNN

# isort:end_maintain_block
conf = get_default_conf()


def get_model():  # type: ignore
    """
    Returns the model, optimizer, and scheduler for training a racecar.
    Returns:
        model (Network): The initialized model for training.
        optimizer (torch.optim.Adam): The optimizer for updating the model parameters.
        scheduler (torch.optim.lr_scheduler.StepLR): The learning rate scheduler for the optimizer.
    """
    if conf.DO_MULTITASK_LEARNING:
        model = MultiTaskCNN(print_shapes=True).to(conf.DEVICE).double().share_memory()
    else:
        model = SingleTaskCNN(print_shapes=True).to(conf.DEVICE).double().share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.IMITATION_LR)  # type: ignore
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=conf.IMITATION_LR_SCHEDULER_STEP_SIZE, gamma=conf.IMITATION_LR_SCHEDULER_GAMMA
    )
    return model, optimizer, scheduler


def get_data_loader(data_dir: str, epoch: int, read_all_sequences: bool, train: bool):
    """
    Returns a generator that yields StateDataset objects from a given data directory.
    Parameters:
    - data_dir (str): The directory path where the data is stored.
    Yields:
    - StateDataset: A StateDataset object containing a batch of sequences.
    """
    if train:
        dataset_name = "training_dataset"
        desc = "Training"
    else:
        dataset_name = "testing_dataset"
        desc = "Testing"
    sequence_dataset = SequenceDataset(data_dir=data_dir, read_all_sequences=read_all_sequences)
    if conf.WANDB_LOG:
        wandb.log({f"{dataset_name}/seeds": wandb.Histogram(sequence_dataset.seeds)}, step=epoch)
        wandb.log({f"{dataset_name}/#sequences": len(sequence_dataset.seeds)}, step=epoch)
    sequence_dataloader = DataLoader(
        sequence_dataset,
        batch_size=conf.SEQUENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=conf.SEQUENCE_NUM_WORKERS,
        prefetch_factor=conf.SEQUENCE_PREFETCH_FACTOR,
        persistent_workers=True,
        collate_fn=sequence_collate_fn,
        pin_memory=True
    )
    num_states = 0
    for i, sequences in tqdm(enumerate(sequence_dataloader), desc=desc):
        num_states += len(sequences.observation)
        if i == len(sequence_dataloader) - 1:
            wandb.log({f"{dataset_name}/#states": num_states}, step=epoch)
        yield DataLoader(
            StateDataset(sequence_dataset_batch=sequences),
            batch_size=conf.STATE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=i == (len(sequence_dataloader) - 1)
        )


def main():
    """
    Main function for training the imitation model.
    """
    run_id: str = io_utils.get_current_time_formatted()
    model, optimizer, scheduler = get_model()
    wandb_logger = WandbLogger(model=model, run_id=run_id)

    dagger_train_output_dir = io_utils.join_dir(conf.DAGGER_OUTPUT_DIR, run_id)

    simulator = Simulator(
        output_dir=dagger_train_output_dir,
        max_steps=conf.TRAINING_MAX_TIME_STEPS,
        student_controller=ImitationDriverController(model=model),
        teacher_controller=PidDriverController(),
        dagger_mode=True,
    )
    simulator.simulate(epoch=0)  # Bootstrap training data

    for epoch in range(conf.IMITATION_EPOCHS):
        # Train
        train_inference = InferenceEngine(model=model, optimizer=optimizer)
        for data_loader in get_data_loader(
            data_dir=dagger_train_output_dir, read_all_sequences=False, epoch=epoch, train=True
        ):
            train_inference.forward(data_loader=data_loader)
        scheduler.step()
        wandb_logger.log_open_loop(inference_engine=train_inference, epoch=epoch, training_mode=True)

        # Test
        test_inference = InferenceEngine(model=model, optimizer=None)  # type: ignore
        for data_loader in get_data_loader(
            data_dir=conf.DAGGER_TEST_DIR, read_all_sequences=True, epoch=epoch, train=False
        ):
            test_inference.forward(data_loader=data_loader)
        wandb_logger.log_open_loop(inference_engine=test_inference, epoch=epoch, training_mode=False)

        # Validation
        _, validate_reward = simulator.simulate(epoch=epoch)
        wandb_logger.log_closed_loop(simulator=simulator, epoch=epoch)

        # Save and log
        if teacher_action_probability(epoch) == 0.0:
            wandb_logger.checkpoint(epoch=epoch, validate_reward=validate_reward)


if __name__ == "__main__":
    set_deterministic()
    mp.set_start_method("spawn", force=True)
    main()
