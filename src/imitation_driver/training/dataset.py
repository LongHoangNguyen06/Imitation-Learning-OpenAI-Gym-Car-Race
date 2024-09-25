from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.imitation_driver.training.preprocess import GroundTruth, preprocess_input_training, preprocess_sequences_training

class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, conf, read_all_sequences=False) -> None:
        """
        Initializes the dataset object.
        Args:
            data_dir (str): The directory where the data is stored.
            read_all_sequences (bool, optional): Flag to determine whether to read all sequences. Defaults to False.
        Returns:
            None
        """
        super().__init__()
        # Initialize tensors to save memory
        self.data_dir = data_dir
        self.record_files = []
        self.read_all_sequences = read_all_sequences
        self._update_data_dir()
        self.conf = conf

    def _read_record_files_names(self) -> list[str]:
        """
        Reads and returns a list of record file names from the specified data directory.
        This method performs the following steps:
        1. Lists all files in the data directory.
        2. Filters the files to include only those with a ".npz" extension.
        3. Constructs the full path for each filtered file.
        4. Resolves symbolic links to their target paths.
        5. Sorts the list of record files.
        Returns:
            list[str]: A sorted list of record file names with resolved symbolic links.
        """
        rfs = os.listdir(self.data_dir)
        rfs = [file for file in rfs if file.endswith(".npz")]
        rfs = [os.path.join(self.data_dir, file) for file in rfs]

        record_files = []
        for record_file in rfs:
            if os.path.islink(record_file):
                record_file = os.readlink(record_file)
            record_files.append(record_file)
        record_files = sorted(record_files)  # Sort by seed
        return record_files

    def _update_data_dir(self):
        """
        Updates the directory containing the dataset records.
        This method reads the record file names and updates the list of record files to be used for training.
        If `read_all_sequences` is False, it selects a subset of the most recent record files and a random
        subset of the remaining files, ensuring that the total number of files does not exceed a specified limit.
        If `read_all_sequences` is True, it uses all available record files.
        The method also extracts and stores the seeds from the filenames of the selected record files.
        Attributes:
            record_files (list): List of selected record file paths.
            seeds (list): List of seeds extracted from the selected record file names.
        """
        sorted_record_files = self._read_record_files_names()
        self.record_files = preprocess_sequences_training(sorted_record_files, self.read_all_sequences, conf=self.conf)
        self.seeds = [int(os.path.basename(record_file).split("_")[0]) for record_file in self.record_files]

    def __len__(self):
        return len(self.record_files)

    def __getitem__(self, idx):
        record_path = self.record_files[idx]
        if os.path.islink(record_path):
            record_path = os.readlink(record_path)
        return preprocess_input_training(record=np.load(record_path), conf=self.conf)


def sequence_collate_fn(batch: list[GroundTruth]) -> GroundTruth:
    """
    Collates a batch of GroundTruth sequences into a single GroundTruth object.
    Args:
        batch (list[GroundTruth]): A list of GroundTruth objects to be collated.
    Returns:
        GroundTruth: A single GroundTruth object with concatenated data from the batch.
    """
    batch = zip(*batch)  # type: ignore
    return GroundTruth(*[torch.cat(b, dim=0) for b in batch])


class StateDataset(Dataset):
    def __init__(self, sequence_dataset_batch: GroundTruth):
        super().__init__()
        self.data = sequence_dataset_batch
        self.n = len(self.data.observation)

    def __len__(self)-> int:
        return self.n

    def __getitem__(self, idx) -> GroundTruth:
        return GroundTruth(*[d[idx] for d in self.data])
