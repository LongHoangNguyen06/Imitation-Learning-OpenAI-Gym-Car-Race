from __future__ import annotations

import os
import socket
from datetime import datetime

from pydantic import validate_call


@validate_call
def get_next_record_number(parent_directory: str, dataset_name: str, digits=3) -> str:
    """
    Dataset in parent_directory is named as "<record_name>_<nr>", etc.

    Args:
        directory (str): The directory where the records are stored.
    Returns:
        str: The nr.
    """
    if not os.path.exists(parent_directory):
        return "1".zfill(digits)
    record_files = os.listdir(parent_directory)
    record_files = [file for file in record_files if file.startswith(dataset_name)]
    record_numbers = [int(file.split("_")[-1]) for file in record_files if file.split("_")[-1].isdigit()]
    next_record_number = max(record_numbers) + 1 if record_numbers else 1
    return str(next_record_number).zfill(digits)


def get_current_time_formatted():
    """
    Returns the current time formatted as "HH_MM_SS_dd_mm_YYYY".
    :return: The formatted current time.
    :rtype: str
    """
    return datetime.now().strftime(f"%Y_%m_%d_{socket.gethostname()}_%H_%M_%S")

@validate_call
def join_dir(*args)-> str:
    """
    Joins two directories.
    Args:
        a (str): The first directory.
        b (str): The second directory.
    Returns:
        str: The joined directory.
    """
    c = os.path.join(*args)
    os.makedirs(c, exist_ok=True)
    return c
