from __future__ import annotations

from collections import namedtuple
import random

import numpy as np
import torch

from src.utils import utils


def convert_action_gym_to_models(action: np.ndarray) -> np.ndarray:
    """
    Convert the action from the gym environment format to the models format.
    Args:
        action (numpy.ndarray): The action array in the gym environment format.
    Returns:
        numpy.ndarray: The converted action array in the models format.
    """
    assert not np.any(
        (action[:, 1] > 0.0) & (action[:, 2] > 0.0)
    ), "No projection possible. Both throttle and brake found in at least one action."
    converted_actions = np.zeros((action.shape[0], 2))
    converted_actions[:, 0] = action[:, 0]  # Steering remains unchanged
    converted_actions[:, 1] = action[:, 1] - action[:, 2]  # acceleration = throttle - brake
    return converted_actions[:, 0], converted_actions[:, 1]  # type: ignore


def convert_action_models_to_gym(steering: torch.Tensor, acceleration: torch.Tensor) -> np.ndarray:
    """
    Converts the action models to the gym format.
    Parameters:
    - outputs: numpy.ndarray
        The output array containing the action models.
    Returns:
    - numpy.ndarray
        The converted controls array in the gym format.
    """
    steering = utils.torch_to_numpy(steering).flatten()  # type: ignore
    acceleration = utils.torch_to_numpy(acceleration).flatten()  # type: ignore
    controls = np.zeros((len(steering), 3))
    controls[:, 0] = steering
    controls[:, 1][acceleration >= 0] = acceleration[acceleration >= 0]
    controls[:, 2][acceleration < 0] = -acceleration[acceleration < 0]
    return controls


def extract_masks(image: np.ndarray, conf) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts observation features from an image.
    Parameters:
    - image: numpy.ndarray
        The input image.
    Returns:
    - numpy.ndarray
        The extracted observation features.
    """
    image = image.copy()
    if len(image.shape) == 3:
        image = image[None, :, :, :]

    # Car position. Fixed.
    x_min, x_max = 45, 52
    y_min, y_max = 65, 80

    image = image[:, : conf.IMAGE_CUTTING_THRESHOLD, :, :]  # Cut image
    image[:, y_min:y_max, x_min:x_max] = [105, 105, 105]  # Cut auto: TODO: this assumes the auto is on the road.
    chevron_masks = image[:, :, :, 0] > 150  # Extract chevron masks
    road_masks = np.abs(image.mean(axis=3) - 105) < 15  # Extract road masks
    chevron_masks = chevron_masks[:, None, :, :]  # Extend channel
    road_masks = road_masks[:, None, :, :]  # Extend channel
    return chevron_masks, road_masks


def preprocess_obs(observation: np.ndarray, conf) -> np.ndarray:
    """
    Preprocesses the observation.
    """
    observation = observation.copy()
    if len(observation.shape) == 3:
        observation = observation[None, :, :, :]
    if observation.dtype != np.float64:
        observation = observation.astype(np.float64)
    observation = observation[:, : conf.IMAGE_CUTTING_THRESHOLD, :, :]
    observation = np.transpose(observation, (0, 3, 1, 2))
    observation = (observation - conf.OBS_MEAN) / conf.OBS_STD
    return observation


def extract_noisy_sensor_values(npy_observation) -> np.ndarray:
    """
    Extracts and processes noisy sensor values from the given observation tensor.
    Parameters:
    observation (np.ndarray): A tensor of shape (batch_size, height, width, channels) representing the observation data.
    Returns:
    tuple: A tuple containing the following processed sensor values:
        - speed (torch.Tensor): A tensor of shape (batch_size, 1) representing the normalized speed values.
        - abs_sensors (torch.Tensor): A tensor of shape (batch_size, 4) representing the normalized ABS sensor values.
        - abs_sennsor_std (torch.Tensor): A tensor of shape (batch_size, 1) representing the normalized ABS sensor standard deviation values.
        - steering (torch.Tensor): A tensor of shape (batch_size, 1) representing the normalized steering values.
        - gyroscope (torch.Tensor): A tensor of shape (batch_size, 1) representing the normalized gyroscope values.
    """
    observation = torch.from_numpy(npy_observation)
    if len(observation.shape) == 3:
        observation = observation[None, :, :, :]

    batch_size = observation.shape[0]

    speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
    speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

    abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
    abs_sensors = (abs_crop.sum(dim=1) / 255 / 5).reshape(batch_size, 4)
    abs_sennsor_std = abs_sensors.std(dim=1, keepdim=True)

    steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
    steer_crop[:, :10] *= -1
    steering = steer_crop.sum(dim=1, keepdim=True)

    gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
    gyro_crop[:, :14] *= -1
    gyroscope = gyro_crop.sum(dim=1, keepdim=True)

    return torch.cat((speed, abs_sensors, abs_sennsor_std, gyroscope, steering), dim=1).numpy()


def preprocess_input_testing(observation, conf):
    """
    Preprocesses the input data for a race car model.
    Args:
        observation (ndarray): The observation data of the race car.
    Returns:
        tuple: A tuple containing the preprocessed observation data and state data.
    """
    state = extract_noisy_sensor_values(observation)

    observation = preprocess_obs(observation, conf)

    return observation, state


def create_balance_weights(labels, unique_classes, class_weights):
    """
    Compute weights based on classes.
    """
    labels = labels.astype(int)
    weight_lookup = np.zeros(np.max(unique_classes) + 1)
    weight_lookup[unique_classes] = class_weights
    weights = weight_lookup[labels]
    return weights.reshape(labels.shape)


def preprocess_sequences_training(sorted_record_files, read_all_sequences, conf):
    """
    Preprocesses the training sequences by selecting a subset of record files.
    If `read_all_sequences` is False, the function selects a combination of the most recent
    record files and a random subset of the remaining record files. This helps in avoiding
    catastrophic forgetting by ensuring that recent data is always included and the rest
    is shuffled and limited per epoch.
    Args:
        sorted_record_files (list): A list of sorted record file paths.
        read_all_sequences (bool): A flag indicating whether to read all sequences or not.
    Returns:
        list: A list of selected record file paths.
    """
    if not read_all_sequences:
        recent_record_files = sorted_record_files[-conf.IMITATION_DATASET_RECENT_MUST_INCLUDE :]
        random_record_files = sorted_record_files[: -conf.IMITATION_DATASET_RECENT_MUST_INCLUDE]
        random.shuffle(random_record_files)  # Shuffle the rest to avoid catastrophic forgetting
        random_record_files = random_record_files[: conf.IMITATION_DATASET_LIMIT_PER_EPOCH]
        return list(set(recent_record_files + random_record_files))
    return sorted_record_files


GroundTruth = namedtuple(
    "GroundTruth",
    [
        "observation",
        "state",
        "curvature",
        "chevron_mask",
        "chevron_mask_weight",
        "road_mask",
        "road_mask_weight",
        "steering",
        "acceleration",
        "desired_speed",
        "speed_error",
        "cte",
        "he"
    ],
)


def preprocess_input_training(record, conf):
    """
    Preprocesses the input data for a race car model.
    Args:
        observation (ndarray): The observation data of the race car.
        action (ndarray): The expert action
    Returns:
        tuple: A tuple containing the preprocessed observation data and state data.
    """
    # Choose only a subset of the data
    random_indices = np.random.choice(
        record["observation"].shape[0],
        record["observation"].shape[0] // conf.IMITATION_DATASET_SAMPLING_RATE,
        replace=False,
    )

    observation=record["observation"][random_indices]
    action=record["teacher_action"][random_indices]
    curvature=record["curvature"][random_indices]
    desired_speed=record["desired_speed"][random_indices]
    speed_error=record["speed_error"][random_indices]
    cte=record["cte"][random_indices]
    he=record["he"][random_indices]

    # Normalize and weighting curvatures
    curvature = np.array((curvature - conf.CURVATURE_MEAN) / conf.CURVATURE_STD).reshape(-1, 1)
    desired_speed = desired_speed / 300.0  # TODO: better normalization
    speed_error = speed_error / 300.0  # TODO: better normalization
    cte /= 10.0

    # Extract masks and pixel wise weights
    chevron_mask, road_mask = extract_masks(observation, conf)
    chevron_mask_weight = create_balance_weights(
        chevron_mask.flatten(), unique_classes=[0, 1], class_weights=conf.CHEVRON_WEIGHT
    ).reshape(chevron_mask.shape)
    road_mask_weight = create_balance_weights(
        road_mask.flatten(), unique_classes=[0, 1], class_weights=conf.ROAD_WEIGHT
    ).reshape(road_mask.shape)

    # Extract steering and acceleration
    steering, acceleration = convert_action_gym_to_models(action)
    # Preprocess inputs
    observation, state = preprocess_input_testing(observation, conf)

    return GroundTruth(
        observation=torch.from_numpy(observation).double(),
        state=torch.from_numpy(state).double(),
        curvature=torch.from_numpy(curvature).double().reshape(-1, 1),
        chevron_mask=torch.from_numpy(chevron_mask).double(),
        chevron_mask_weight=torch.from_numpy(chevron_mask_weight).double(),
        road_mask=torch.from_numpy(road_mask).double(),
        road_mask_weight=torch.from_numpy(road_mask_weight).double(),
        steering=torch.from_numpy(steering).double().reshape(-1, 1),
        acceleration=torch.from_numpy(acceleration).double().reshape(-1, 1),
        desired_speed=torch.from_numpy(desired_speed).double().reshape(-1, 1),
        speed_error=torch.from_numpy(speed_error).double().reshape(-1, 1),
        cte=torch.from_numpy(cte).double().reshape(-1, 1),
        he=torch.from_numpy(he).double().reshape(-1, 1),
    )
