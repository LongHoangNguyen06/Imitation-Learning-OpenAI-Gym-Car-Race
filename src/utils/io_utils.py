from __future__ import annotations

import os
import socket
from datetime import datetime


def get_next_record_name(directory: str, digits=3) -> str:
    """
    Dataset in directory is named as "<trial>_<nr>_<seed>_<reward>.npz", etc.
    where <nr> is the next number in the sequence with digits digits.

    Args:
        directory (str): The directory where the records are stored.
    Returns:
        str: The nr.
    """
    if not os.path.exists(directory):
        return "1".zfill(digits)
    record_files = os.listdir(directory)
    record_files = [file for file in record_files if file.endswith(".npz")]
    record_files_without_ext = [file.split(".")[0] for file in record_files]
    record_numbers = [int(file.split("_")[0]) for file in record_files_without_ext if file.split("_")[0].isdigit()]
    next_record_number = max(record_numbers) + 1 if record_numbers else 1
    return str(next_record_number).zfill(digits)


def get_next_trial_name(directory: str, digits=3) -> str:
    """
    Config in directory is named as "<config>_<trial>_<mean>_<std>.yaml", etc.
    where <nr> is the next number in the sequence with digits digits.

    Args:
        directory (str): The directory where the records are stored.
    Returns:
        str: The nr.
    """
    if not os.path.exists(directory):
        return "1".zfill(digits)
    config_files = os.listdir(directory)
    config_files = [file for file in config_files if file.endswith(".yaml")]
    config_files_without_ext = [file.split(".")[1] for file in config_files]
    config_numbers = [int(file.split("_")[1]) for file in config_files_without_ext if file.split("_")[1].isdigit()]
    next_config_number = max(config_numbers) + 1 if config_numbers else 1
    return str(next_config_number).zfill(digits)


def get_next_dataset_name(parent_directory, dataset_name, digits=3) -> str:
    """
    Dataset in parent_directory is named as "<dataset_name> <nr>", etc.

    Args:
        directory (str): The directory where the records are stored.
    Returns:
        str: The nr.
    """
    if not os.path.exists(parent_directory):
        return "1".zfill(digits)
    record_files = os.listdir(parent_directory)
    record_files = [file for file in record_files if file.startswith(dataset_name)]
    record_numbers = [int(file.split("_")[1]) for file in record_files if file.split("_")[1].isdigit()]
    next_record_number = max(record_numbers) + 1 if record_numbers else 1
    return str(next_record_number).zfill(digits)


def get_current_time_formatted():
    """
    Returns the current time formatted as "HH_MM_SS_dd_mm_YYYY".
    :return: The formatted current time.
    :rtype: str
    """
    return datetime.now().strftime(f"%Y_%m_%d_{socket.gethostname()}_%H_%M_%S")
