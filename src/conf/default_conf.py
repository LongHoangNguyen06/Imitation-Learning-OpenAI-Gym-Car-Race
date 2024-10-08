from __future__ import annotations

import multiprocessing
import os

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################################################
RECORD_SEEDS = list(range(1_000_000, 1_000_100)) # Seeds for testing during training.
# RECORD_SEEDS = list(range(3000, 3100))  # Seeds for final closed-loop benchmarking.
RENDER_MODE = "rgb_array"
TRAINING_MAX_TIME_STEPS = 600
DT = 1.0 / 50.0

###########################################################################
# General output directories
OUTPUT_DIR = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs"
os.makedirs(name=OUTPUT_DIR, exist_ok=True)
RECORD_OUTPUT_DIR = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/records"
os.makedirs(name=RECORD_OUTPUT_DIR, exist_ok=True)
GYM_MAX_WORKERS = min(max(multiprocessing.cpu_count() - 1, 4), 12)

###########################################################################
# Cutting images threshold
IMAGE_CUTTING_THRESHOLD = 84
OBSERVATION_DIM = (3, IMAGE_CUTTING_THRESHOLD, 96)
MASK_DIM = (1, IMAGE_CUTTING_THRESHOLD, 96)

from src.conf.class_frequencies_conf import *
from src.conf.imitation_learning_conf import *
from src.conf.normalization_conf import *
