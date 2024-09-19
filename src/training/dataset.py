from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.utils.preprocess import preprocess_input_training
from src.utils import conf_utils

conf = conf_utils.get_default_conf()


class SequenceDataset(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        # Initialize tensors to save memory
        self.data_dir = data_dir
        self.record_files = []
        self._update_data_dir()

    def _update_data_dir(self):
        rfs = os.listdir(self.data_dir)
        rfs = [file for file in rfs if file.endswith(".npz")]
        rfs = [os.path.join(self.data_dir, file) for file in rfs]
        __record_files__ = []
        for record_file in rfs:
            if os.path.islink(record_file):
                record_file = os.readlink(record_file)
            __record_files__.append(record_file)


        __record_files__ = sorted(__record_files__)  # Sort by seed
        recent_record_files = __record_files__[
            -conf.IMITATION_DATASET_RECENT_MUST_INCLUDE :
        ]  # Make the most recent record files available for learning to learn from most recent errors
        random.shuffle(__record_files__)  # Shuffle the rest to avoid catastrophic forgetting
        __record_files__ = recent_record_files + __record_files__[: conf.IMITATION_DATASET_LIMIT_PER_EPOCH]
        __record_files__ = list(set(__record_files__))  # Remove duplicates
        self.record_files = __record_files__
        self.seeds = [int(os.path.basename(record_file).split("_")[0]) for record_file in self.record_files]

    def __len__(self):
        return len(self.record_files)

    def __getitem__(self, idx):
        record_path = self.record_files[idx]
        if os.path.islink(record_path):
            record_path = os.readlink(record_path)
        record = np.load(record_path)

        return preprocess_input_training(
            obs=record["observation_history"][:: conf.DATASET_SAMPLING_RATE],
            action=record["action_history"][:: conf.DATASET_SAMPLING_RATE],
            curvature=record["curvature_history"][:: conf.DATASET_SAMPLING_RATE],
        )


def sequence_collate_fn(batch):
    batch = zip(*batch)
    return tuple(torch.cat(b, dim=0) for b in batch)


class StateDataset(Dataset):
    def __init__(self, sequence_dataset_batch):
        super().__init__()
        self.data = sequence_dataset_batch
        self.n = len(self.data[0])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return tuple(torch.tensor(d[idx]).double() for d in self.data)
