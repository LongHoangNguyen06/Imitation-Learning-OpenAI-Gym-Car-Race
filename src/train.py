from __future__ import annotations

import random
import warnings

from src.pid_driver.pid_driver_controller import PidDriverController

# Ignore all warnings
warnings.filterwarnings("ignore")

import matplotlib

from src.utils.utils import set_deterministic

matplotlib.use("Agg")
# isort:maintain_block
import os
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import shutil
import socket
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from dynaconf import Dynaconf
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.imitation_driver import network
from src.imitation_driver.imitation_driver_controller import ImitationDriverController
from src.utils import env_utils, io_utils
from src.utils.dataset import DaggerDataset, ImitationDataset
from src.utils.training_debug_plot import *

# isort:end_maintain_block
conf = Dynaconf(settings_files=["conf/default_conf.py"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_deterministic()


class OneEpochTrainer:
    def __init__(self, model, data_loader, optimizers, tensorboard_writer, epoch, do_train=True):
        """
        Initializes the trainer object.
        Args:
            model: The model to be trained.
            data_loader: The data loader object for loading training data.
            optimizers: The optimizer(s) to be used for training.
            tensorboard_writer: The writer object for logging training progress to TensorBoard.
            epoch: The current epoch number.
            do_train: A boolean indicating whether to perform training or not. Default is True.
        """
        # Initialize the trainer
        self.model = model
        self.data_loader = data_loader
        self.optimizers = optimizers
        self.tensorboard_writer = tensorboard_writer
        self.epoch = epoch
        self.do_train = do_train

        # Loss history
        self.losses = []
        self.steering_losses = []
        self.acceleration_losses = []
        self.road_segmentation_losses = []
        self.chevron_visble_losses = []
        self.curvature_losses = []

        # Grad flow
        self.avg_grad = defaultdict(list)

    # Function to train model
    def one_step(self):
        """
        Trains the model using the given data loader and optimizer for a specified number of epochs.
        Args:
            model (nn.Module): The model to be trained.
            data_loader (DataLoader): The data loader object that provides the training data.
            optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
            tensorboard_writer (SummaryWriter): The tensorboard writer object for logging.
            epoch (int): The current epoch number.
        Returns:
            Tuple[np.ndarray, float]: A tuple containing the array of training losses and the mean training loss.
        """
        # Set model to training mode if needed
        if self.do_train:
            self.model.train()
        else:
            self.model.eval()

        # Initialize losses history
        progress_bar = tqdm(self.data_loader, unit="batch", desc="Training" if self.do_train else "Testing", leave=True)
        for i, (observation, state, action, instance_weights, curvature, masks) in enumerate(progress_bar):
            # Reset gradients
            for opt in self.optimizers:
                opt.zero_grad()

            # Load data to GPU
            self.gt_observation = observation.to(device, non_blocking=True)
            self.gt_state = state.to(device, non_blocking=True)
            self.gt_instance_weights = instance_weights.to(device, non_blocking=True)
            self.gt_action = action.to(device, non_blocking=True)
            self.gt_curvature = curvature.to(device, non_blocking=True)
            self.gt_masks = masks.to(device, non_blocking=True)
            self.gt_chevron_mask = masks[:, 0, :, :].to(device, non_blocking=True)
            self.gt_chevron_visible = (
                (self.gt_chevron_mask.sum(dim=(1, 2)) > conf.IMITATION_CHEVRON_VISIBLE_THRESHOLD)
                .to(device, non_blocking=True)
                .flatten()
                .double()
            )
            self.gt_road_mask = masks[:, 1, :, :].to(device, non_blocking=True)
            self.gt_steering = action[:, 0].to(device, non_blocking=True)
            self.gt_acceleration = action[:, 1].to(device, non_blocking=True)

            # Forward pass and compute loss
            self.loss = self._compute_loss()

            # Backpropagate the gradients and do optimization
            if self.do_train:
                self.loss.backward()
                for opt in self.optimizers:
                    opt.step()

                self.collect_gradient_stats()
            progress_bar.set_postfix(loss=self.loss.item())
            self._log(i)
        self._log(i, force_plot=True)  # type: ignore

    def collect_gradient_stats(self):
        """
        Collects gradient statistics for the model's parameters.
        This method iterates over the named parameters of the model and collects the average gradient values
        for the parameters that require gradients and are not biases. The average gradient values are stored
        in the `avg_grad` dictionary.
        Returns:
            None
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad and "bias" not in n:
                self.avg_grad[n].append(p.grad.abs().mean().item())

    def _log(self, i, force_plot=False):
        if ((i % conf.IMITATION_TENSOR_BOARD_WRITING_FREQUENCY == 0) or force_plot) and self.do_train:
            self._log_tensorboard_network_debug()
        self._log_tensorboard_metrics()
        self._log_wandb()

    def _compute_loss(self):
        """
        Compute the loss for the racecar model.
        Returns:
            float: The computed loss value.
        """
        # Forward pass
        (
            self.predicted_road_mask,
            self.predicted_chevrons_visible,
            self.predicted_curvature,
            self.predicted_steering,
            self.predicted_acceleration,
        ) = self.model(observation=self.gt_observation, state=self.gt_state)

        # Compute and scale individual losses
        self.steering_loss = (
            torch.tensor(conf.IMITATION_STEERING_LOSS).to(device)
            * (self.gt_steering.flatten() - self.predicted_steering.flatten()).abs().mean()
        )
        self.acceleration_loss = (
            torch.tensor(conf.IMITATION_ACCELERATION_LOSS).to(device)
            * (self.gt_acceleration.flatten() - self.predicted_acceleration.flatten()).abs().mean()
        )
        self.chevron_visble_loss = torch.tensor(conf.IMITATION_CHEVRON_VISIBLE_LOSS).to(device) * bce_logits(
            self.predicted_chevrons_visible.flatten(),
            self.gt_chevron_visible,
            weight=self.gt_instance_weights.flatten(),
        )
        self.road_segmentation_loss = torch.tensor(conf.IMITATION_SEGMENTATION_LOSS).to(device) * bce_logits(
            self.predicted_road_mask.flatten(start_dim=1),
            self.gt_road_mask.flatten(start_dim=1),
            weight=self.gt_instance_weights.reshape(-1, 1),
        )
        self.curvature_loss = (
            torch.tensor(conf.IMITATION_CURVATURE_LOSS).to(device)
            * ((self.gt_curvature.flatten() - self.predicted_curvature.flatten()) ** 2).mean()
        )

        # Sum all losses
        self.loss = (
            self.steering_loss
            + self.acceleration_loss
            + self.road_segmentation_loss
            + self.chevron_visble_loss
            + self.curvature_loss
        )

        # Append losses to history
        self.losses.append(self.loss.item())
        self.steering_losses.append(self.steering_loss.item())
        self.acceleration_losses.append(self.acceleration_loss.item())
        self.road_segmentation_losses.append(self.road_segmentation_loss.item())
        self.chevron_visble_losses.append(self.chevron_visble_loss.item())
        self.curvature_losses.append(self.curvature_loss.item())

        return self.loss

    def _log_tensorboard_subnetwork_gradient_and_activation(self, model, model_name):
        """
        Logs the gradient, activation, and weight information of a subnetwork to TensorBoard.
        Parameters:
            model (Subnetwork): The subnetwork model.
            model_name (str): The name of the subnetwork model.
        Returns:
            None
        """
        for key in model.activations.keys():
            if key.startswith("BatchNorm2d"):
                continue
            if key.startswith("Conv2d") or key.startswith("ConvTranspose2d"):
                self.tensorboard_writer.add_image(
                    f"{model_name}_activation_image_{key}",
                    plot_tensor_grid(model.activations[key]),
                    self.epoch,
                    dataformats="HWC",
                )

    def _log_tensorboard_network_debug(self):
        """
        Logs various network components to TensorBoard.
        """
        self.tensorboard_writer.add_figure("opt_grad_flow", plot_grad_flow(self.avg_grad), self.epoch)
        self.tensorboard_writer.add_image(
            "input_observation_image",
            plot_observation_grid(
                observation=self.gt_observation, action=self.gt_action, weight=self.gt_instance_weights
            ),
            self.epoch,
            dataformats="HWC",
        )
        self.tensorboard_writer.add_image(
            "statistic_control_steering",
            plot_predicted_action_and_actual_action(
                predicted_action=self.predicted_steering, actual_action=self.gt_steering
            ),
            self.epoch,
            dataformats="HWC",
        )
        self.tensorboard_writer.add_image(
            "statistic_control_acceleration",
            plot_predicted_action_and_actual_action(
                predicted_action=self.predicted_acceleration, actual_action=self.gt_acceleration
            ),
            self.epoch,
            dataformats="HWC",
        )
        self.tensorboard_writer.add_image(
            "statistic_auxiliary_curvature",
            plot_predicted_action_and_actual_action(
                predicted_action=self.predicted_curvature, actual_action=self.gt_curvature
            ),
            self.epoch,
            dataformats="HWC",
        )
        self.tensorboard_writer.add_pr_curve(
            "statistic_auxiliary_chevron_visibles",
            self.gt_chevron_visible.flatten(),
            F.sigmoid(self.predicted_chevrons_visible.flatten()),
            self.epoch,
        )
        self.tensorboard_writer.add_pr_curve(
            "statistic_auxiliary_road_segmentation",
            self.gt_road_mask.flatten(),
            F.sigmoid(self.predicted_road_mask.flatten()),
            self.epoch,
        )

        if hasattr(self.model, "backbone"):
            self._log_tensorboard_subnetwork_gradient_and_activation(self.model.backbone, "backbone")
        if hasattr(self.model, "road_decoder"):
            self._log_tensorboard_subnetwork_gradient_and_activation(self.model.road_decoder, "road_decoder")
        if hasattr(self.model, "chevrons_predictor"):
            self._log_tensorboard_subnetwork_gradient_and_activation(
                self.model.chevrons_predictor, "chevrons_predictor"
            )
        if hasattr(self.model, "curvature_predictor"):
            self._log_tensorboard_subnetwork_gradient_and_activation(
                self.model.curvature_predictor, "curvature_predictor"
            )
        if hasattr(self.model, "steering_predictor"):
            self._log_tensorboard_subnetwork_gradient_and_activation(
                self.model.steering_predictor, "steering_predictor"
            )
        if hasattr(self.model, "acceleration_predictor"):
            self._log_tensorboard_subnetwork_gradient_and_activation(
                self.model.acceleration_predictor, "acceleration_predictor"
            )

    def _log_tensorboard_metrics(self):
        """
        Logs metrics to TensorBoard.
        These metrics are logged with their corresponding values and the current epoch.
        Parameters:
        - None
        Returns:
        - None
        """
        id = "train" if self.do_train else "test"
        loss_dict = {
            "loss": np.mean(self.losses),
            "steering_loss": np.mean(self.steering_losses),
            "acceleration_loss": np.mean(self.acceleration_losses),
            "road_segmentation": np.mean(self.road_segmentation_losses),
            "chevron_visible_loss": np.mean(self.chevron_visble_losses),
            "curvature_loss": np.mean(self.curvature_losses),
        }

        # Write all the scalars at once
        self.tensorboard_writer.add_scalars(f"opt_{id}_losses", loss_dict, self.epoch)

    def _log_wandb(self):
        """
        Logs the training or testing loss and learning rate to wandb.
        Parameters:
        - self: The instance of the class.
        Returns:
        - None
        """
        # Write to wandb
        if conf.WANDB_LOG:
            id = "train" if self.do_train else "test"
            wandb.log({f"opt_{id}_loss": wandb.Histogram(self.losses)}, step=self.epoch)
            for i, opt in enumerate(self.optimizers):
                wandb.log({f"opt_lr_{i}": opt.param_groups[0]["lr"]}, step=self.epoch)
            wandb.log({f"opt_{id}_loss_mean": np.mean(self.losses)}, step=self.epoch)


# Closed loop simulation
def validate(evaluation_input):
    """
    Validates the performance of a given model on a racecar simulation.
    Args:
        evaluation_input (tuple): A tuple containing the model and seed.
    Returns:
        int: The total reward obtained by the model during the simulation.
    """
    model, seed = evaluation_input
    model.eval()  # Set model to evaluation mode
    controller = ImitationDriverController(conf=conf, model=model)
    env = env_utils.create_env(conf=conf)

    # Initialize new scenario
    terminated = truncated = done = False
    observation, info = env.reset(seed=seed)
    seed_reward = step = 0

    # Start simulation
    while not (done or terminated or truncated):
        action = controller.get_action(
            observation,
            info,
            speed=env_utils.get_speed(env),
            wheels_omegas=env_utils.get_wheel_velocities(env),
            angular_velocity=env.unwrapped.car.hull.angularVelocity,  # type: ignore
            steering_joint_angle=env.unwrapped.car.wheels[0].joint.angle,  # type: ignore
        ).squeeze()
        observation, reward, terminated, truncated, info = env.step(action)
        seed_reward += reward  # type: ignore
        step += 1
        if step >= conf.MAX_TIME_STEPS:
            break
    return int(seed_reward)


def get_test_loaders():
    """
    Returns the data loaders for testing the race car model.
    Returns:
        DataLoader: The testing data loader.
    """
    for testing_set in conf.IMITATION_TESTING_SETS:
        yield DataLoader(
            ImitationDataset(testing_set),
            batch_size=conf.IMITATION_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )


def get_train_loaders():
    """
    Returns a generator that yields DataLoader objects for each training set in conf.IMITATION_TRAINING_SETS.
    Each DataLoader object is created with a RaceCarDataset object as the dataset, using the training_set as the input.
    The batch size is set to conf.IMITATION_BATCH_SIZE, and the data is shuffled.
    The number of workers is set to 0, and pin_memory is set to False.
    Returns:
        A generator that yields DataLoader objects.
    """
    for training_set in conf.IMITATION_TRAINING_SETS:
        yield DataLoader(
            ImitationDataset(training_set),
            batch_size=conf.IMITATION_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )


def get_dagger_train_loaders(dagger_output_dir):
    """
    Returns a generator that yields DataLoader objects for each dataset created from
    the files in dagger_output_dir. The number of DataLoader objects is limited to
    DAGGER_TRAINING_DATALOADER_LIMIT, and each DataLoader is created with a
    DaggerDataset object as the dataset, containing a random subset of files.

    Returns:
        A generator that yields DataLoader objects.
    """
    # List all the files in the dagger_output_dir
    all_files = os.listdir(dagger_output_dir)
    random.shuffle(all_files)

    # Split the files into disjoint sets
    for _ in range(conf.DAGGER_TRAINING_NUM_DATALOADER):
        # Get a subset of files for this DataLoader
        subset_size = min(conf.DAGGER_TRAINING_NUM_DATA_PER_DATALOADER, len(all_files))
        subset_files = [all_files.pop() for _ in range(subset_size)]
        subset_files = [os.path.join(dagger_output_dir, file) for file in subset_files]
        if len(subset_files) == 0:
            break

        yield DataLoader(
            DaggerDataset(subset_files),
            batch_size=conf.IMITATION_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )


def get_model():
    """
    Returns the model, optimizer, and scheduler for training a racecar.
    Returns:
        model (Network): The initialized model for training.
        optimizer (torch.optim.Adam): The optimizer for updating the model parameters.
        scheduler (torch.optim.lr_scheduler.StepLR): The learning rate scheduler for the optimizer.
    """
    if conf.DO_MULTITASK_LEARNING:
        model = network.MultiTaskCNN(print_shapes=True).to(device).double()
    else:
        model = network.SingleTaskCNN(print_shapes=True).to(device).double()
    model.share_memory()
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
            artifact = wandb.Artifact("code_artifact", type="code")
            artifact.add_dir(get_code_artifact_dir())
            wandb.log_artifact(artifact)


def print_config():
    """
    Prints the configuration values.
    This function iterates over the `conf` dictionary and prints each key-value pair.
    Parameters:
    None
    Returns:
    None
    """
    for key in conf:
        print(f"{key}: {conf[key]}")


def teacher_action_probability(epoch):
    """
    Returns the probability of the teacher action being chosen.
    The probability is calculated as p^i, where p is the value of DAGGER_P and i is the epoch number.
    Parameters:
    - epoch (int): The current epoch number.
    Returns:
    - float: The probability of the teacher action being chosen.
    """
    return conf.DAGGER_P**epoch


def dagger_loop(seed, student_model, epoch, record_path):
    """
    Executes the dagger loop for training a student model using imitation learning.
    Args:
        seed (int): The seed for the environment.
        student_model: The student model to be trained.
        epoch (int): The current epoch of training.
        record_path (str): The path to save the training records.
    Returns:
        None
    """

    def choose_action(student_action, teacher_action):
        """if random(0, 1) > beta = p^i then choose teacher action, else choose student action"""
        if np.random.random() < teacher_action_probability(epoch):
            return teacher_action
        return student_action

    student_model.eval()  # Set model to evaluation mode
    student_driver = ImitationDriverController(conf=conf, model=student_model)  # type: ignore
    teacher_driver = PidDriverController()  # type: ignore
    env = env_utils.create_env(conf=conf)

    # Set up history
    history = defaultdict(list)

    # Initialize new scenario
    env = env_utils.create_env(conf=conf)
    terminated = truncated = done = False
    observation, info = env.reset(seed=seed)
    seed_reward = step = 0
    track = env_utils.extract_track(env)

    # Start simulation
    while not (done or terminated or truncated):
        # Record history
        pose = env_utils.get_pose(env)
        speed = env_utils.get_speed(env)
        wheels_omegas = env_utils.get_wheel_velocities(env)
        steering_joint_angle = env.unwrapped.car.wheels[0].joint.angle  # type: ignore
        angular_velocity = env.unwrapped.car.hull.angularVelocity  # type: ignore
        history["speed_history"].append(speed)
        history["pose_history"].append(pose)
        history["wheels_omegas_history"].append(wheels_omegas)
        history["steering_joint_angle_history"].append(steering_joint_angle)
        history["angular_velocity_history"].append(angular_velocity)
        history["observation_history"].append(observation)
        track = env_utils.extract_track(env)

        # Simulation
        student_action = student_driver.get_action(
            observation,
            info,
            speed=speed,
            wheels_omegas=wheels_omegas,
            angular_velocity=angular_velocity,  # type: ignore
            steering_joint_angle=steering_joint_angle,  # type: ignore
        )
        teacher_action = teacher_driver.get_action(
            observation,
            info,
            speed=speed,
            wheels_omegas=wheels_omegas,
            angular_velocity=angular_velocity,  # type: ignore
            steering_joint_angle=steering_joint_angle,  # type: ignore
            track=track,
            pose=pose,
        )
        observation, reward, terminated, truncated, info = env.step(choose_action(student_action, teacher_action))

        # Record history
        history["action_history"].append(teacher_action)

        # Go to next step
        seed_reward += reward  # type: ignore

        # Increment step
        step += 1
        if step >= conf.MAX_TIME_STEPS:
            terminated = True
            break
    seed_reward = int(seed_reward)

    # Save record
    np.savez(
        os.path.join(
            record_path, io_utils.get_current_time_formatted() + f"_seed_{seed}_epoch_{epoch}_{seed_reward}.npz"
        ),
        seed=seed,
        seed_reward=seed_reward,
        terminated=terminated,
        truncated=truncated,
        done=done,
        track=track,
        **history,
        **teacher_driver.debug_states,
    )
    env.close()

    print(f"Finished DAgger loop. Reward: {seed_reward}")
    return seed_reward


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
    # Initialize logging
    run_id = io_utils.get_current_time_formatted()
    tensorboard_writer = get_tensorboard_writer(run_id)
    initialize_wandb(run_id)

    # Device configuration
    model_output_path = conf.IMITATION_OUTPUT_MODEL
    os.makedirs(model_output_path, exist_ok=True)

    # Initialize the model, optimizer, and loss function
    model, optimizers, schedulers = get_model()
    print("Model initialized")
    print_config()

    # Initialize checkpointing
    best_validate_reward = float("-inf")
    model_save_file = os.path.join(model_output_path, f"{run_id}_{model.__class__.__name__}")
    print(f"Model save file: {model_save_file}")

    # Training loop
    dagger_output_dir = os.path.join(conf.DAGGER_OUTPUT_DIR, run_id)
    os.makedirs(dagger_output_dir, exist_ok=True)
    dagger_start_demo_seed = conf.DAGGER_START_DEMO_SEED
    if conf.DAGGER_TRAINING:
        epochs = conf.DAGGER_EPOCHS
    else:
        epochs = conf.IMITATION_EPOCHS
    for epoch in range(epochs):
        # Run a DAgger loop to collect data
        if conf.DAGGER_TRAINING:
            dagger_start_demo_seed += 1
            dagger_seed_reward = dagger_loop(
                seed=dagger_start_demo_seed, student_model=model, epoch=epoch, record_path=dagger_output_dir
            )
            if conf.WANDB_LOG:
                wandb.log({f"opt_dagger_seed_reward": dagger_seed_reward}, step=epoch)
                wandb.log({f"opt_teacher_action_proboability": teacher_action_probability(epoch)}, step=epoch)
                print(f"WandB log")

        # Train for one epoch
        model.hook()
        if conf.DAGGER_TRAINING:
            train_loaders = get_dagger_train_loaders(dagger_output_dir)
            print("Dagger data loaders")
        else:
            train_loaders = get_train_loaders()
            print("Imitation data loaders")

        for train_loader in train_loaders:
            OneEpochTrainer(model, train_loader, optimizers, tensorboard_writer, epoch, do_train=True).one_step()
        model.unhook()

        # Step the learning rate scheduler
        for scheduler in schedulers:
            scheduler.step()

        # Test for one epoch
        for test_loader in get_test_loaders():
            OneEpochTrainer(model, test_loader, optimizers, tensorboard_writer, epoch, do_train=False).one_step()

        if conf.DAGGER_TRAINING and epoch < conf.DAGGER_SKIP_VALIDATION_FIRST_N_EPOCHS:
            continue
        # Validate with closed loop simulation
        validate_rewards = process_map(
            validate,
            [(model, seed) for seed in conf.IMITATION_EVALUATION_SEEDS],
            max_workers=conf.MAX_WORKERS,
            desc="Validating Closed Loop",
        )
        validate_reward = np.mean(validate_rewards)

        # Check if this test loss is the best so far
        if validate_reward > best_validate_reward:
            best_validate_reward = validate_reward
            torch.save(model.state_dict(), model_save_file + f"_{int(validate_reward)}.pth")
            print(f"Best model saved at epoch {epoch + 1} with validate reward: {validate_reward:.2f}")
        if conf.WANDB_LOG:
            wandb.log({f"opt_reward": validate_reward}, step=epoch)
            wandb.log({f"opt_teacher_action_proboability": teacher_action_probability(epoch)}, step=epoch)
        tensorboard_writer.add_scalar(f"opt_reward", validate_reward, epoch)
        print(f"Epoch {epoch + 1}/{conf.IMITATION_EPOCHS}, validate reward: {validate_reward:.2f}")

    tensorboard_writer.close()
    if conf.WANDB_LOG:
        wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
