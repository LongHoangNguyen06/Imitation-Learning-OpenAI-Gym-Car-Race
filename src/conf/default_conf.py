from __future__ import annotations

import multiprocessing
import os

DEBUG = False  ##### DEBUG #####: change this!
DO_MULTITASK_LEARNING = True  ##### DEBUG #####: change this!
DAGGER_TRAINING = True  ##### DEBUG #####: change this!

USE_SMALL_DATASET = DEBUG
WANDB_LOG = not DEBUG
TENSOR_BOARD_LOG = False
USE_RGB = False

################################################################################################################################
MAX_TIME_STEPS = 600

################################################################################################################################
# Environment's initialization
DOMAIN_RANDOMIZE = False
RENDER_MODE = "rgb_array"
LAP_COMPLETE_PERCENT = 0.95
RENDER_FPS = 50.0
DT = 1.0 / float(RENDER_FPS)

################################################################################################################################
# Recording parameters when doing Demos
DO_RECORD = True
OUTPUT_DIR = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs"
os.makedirs(name=OUTPUT_DIR, exist_ok=True)
RECORD_OUTPUT_DIR = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/records"
os.makedirs(name=RECORD_OUTPUT_DIR, exist_ok=True)
GYM_MAX_WORKERS = max(multiprocessing.cpu_count() - 1, 4)

################################################################################################################################
# Data normalization constants
OBS_MEAN = 148.5031
CURVATURE_MEAN = 0.0101
SPEED_MEAN = 71.0694
WHEEL_OMEGA_MEAN = 132.1778
WHEEL_OMEGA_STD_MEAN = 4.4831
ANGULAR_VELOCITY_MEAN = 0.4464
STEERING_JOINT_ANGLE_MEAN = 0.0282
OBS_STD = 29.1733
CURVATURE_STD = 0.0144
SPEED_STD = 22.0007
WHEEL_OMEGA_STD = 40.5991
WHEEL_OMEGA_STD_STD = 2.2241
ANGULAR_VELOCITY_STD = 1.5146
STEERING_JOINT_ANGLE_STD = 0.1114
STATE_DIM = 8  # Vehicle's kinematic states
TENSOR_BOARD_DIR = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/tensorboard"
os.makedirs(name=TENSOR_BOARD_DIR, exist_ok=True)
TENSOR_BOARD_WRITING_FREQUENCY = 1000

################################################################################################################################
# Output parameters for imitation learning
IMITATION_OUTPUT_MODEL = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/imitation_models/"
os.makedirs(name=IMITATION_OUTPUT_MODEL, exist_ok=True)

################################################################################################################################
# Imitation training parameters
IMITATION_EPOCHS = 50
IMITATION_LR = 0.01
IMITATION_LR_SCHEDULER_STEP_SIZE = 2
IMITATION_LR_SCHEDULER_GAMMA = 0.5
IMITATION_CHEVRON_VISIBLE_THRESHOLD = (
    10  # How many pixels being red/white to be considered a chevron marking is visible on the road
)
IMITATION_BALANCE_BINS = 2
IMITATION_DROPOUT_PROB = 0.1
IMITATION_DATASET_LIMIT = 500000

################################################################################################################################
# Backbone network parameters
IMITATION_NUM_FILTERS_ENCODER = 64  # Constants for number of filters in the encoder and decoder
IMITATION_LATENT_DIM = 6 * IMITATION_NUM_FILTERS_ENCODER  # Constants for the latent dimension of the final layer

################################################################################################################################
# Prediction network parameters
IMITATION_CHEVRON_DIMS = [32, 16, 1]
IMITATION_CURVATURE_DIMS = [32, 16, 1]
IMITATION_STEERING_DIMS = [64, 32, 1]
IMITATION_ACCELERATION_DIMS = [64, 32, 1]

################################################################################################################################
# Imitation data parameters
IMITATION_DATASET_SAMPLING_RATE = 2  # Ignore (n-1)/n of the data
if USE_SMALL_DATASET:
    IMITATION_DATA_DIR = "/graphics/scratch2/students/nguyenlo/CarRace/CarRaceOutputs/records/benchmark_004"
    IMITATION_EVALUATION_SEEDS = list(range(1000, 1001))
    DAGGER_ITERATION_PER_LOOP = 1
else:
    IMITATION_DATA_DIR = "/graphics/scratch2/students/nguyenlo/CarRace/CarRaceOutputs/merged"
    IMITATION_EVALUATION_SEEDS = list(range(1000, 1100))
    DAGGER_ITERATION_PER_LOOP = 100
if USE_RGB:
    OBSERVATION_DIM = (3, 84, 96)
else:
    OBSERVATION_DIM = (1, 84, 96)

# Multi tasking loss scaling coefficients
if DO_MULTITASK_LEARNING:
    IMITATION_CHEVRON_VISIBLE_LOSS = 1  # Auxilliary loss: predict whether chevron markings are visible.
    IMITATION_SEGMENTATION_LOSS = 1  # Auxilliary loss: Reconstruction loss: predict the road segmentation.
    IMITATION_CURVATURE_LOSS = 1  # Auxilliary loss: Predict the curvature of the road.
else:
    IMITATION_CHEVRON_VISIBLE_LOSS = 0  # Auxilliary loss: predict whether chevron markings are visible.
    IMITATION_SEGMENTATION_LOSS = 0  # Auxilliary loss: Reconstruction loss: predict the road segmentation.
    IMITATION_CURVATURE_LOSS = 0  # Auxilliary loss: Predict the curvature of the road.
IMITATION_ACCELERATION_LOSS = 1  # Control loss: Predict the acceleration of expert.
IMITATION_STEERING_LOSS = 1  # Control loss: Predict the steering angle of expert.

################################################################################################################################
# DAgger parameters
DAGGER_EPOCHS = 5000
DAGGER_OUTPUT_DIR = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/dagger_data/"
os.makedirs(name=DAGGER_OUTPUT_DIR, exist_ok=True)
DAGGER_LR_SCHEDULER_STEP_SIZE = 25
DAGGER_P_DECAY = 0.98  # Decay the probability of using the teacher model.
DAGGER_KEEP_RECORD_P_DECAY = 0.8  # Decay the probability of keeping the record
DAGGER_KEEP_RECORD_MIN_P = 0.01  # Keep record with min_p% probability at least.
DAGGER_TEACHER_P_CUT_OFF = 0.2  # if p_teacher < cut_off, we just use the student model.
DAGGER_START_SEED = 100000  # Start seed for the environment to generate the data with dagger.
DAGGER_DATASET_LIMIT_PER_EPOCH = 64  # Avoid quadratic growth of the dataset since dagger produces a lot of data.
DAGGER_DATASET_RECENT_MUST_INCLUDE = 16  # Must learn current n data records.

################################################################################################################################
# Data loading parameters
SEQUENCE_BATCH_SIZE = 32  # Loading 32 sequences as one batch. Each sequence at most 600 frames.
STATE_BATCH_SIZE = 64  # Train on 256 frames at once. This is equal to BATCH_SIZE.
SEQUENCE_NUM_WORKERS = 2  # How many subprocesses to use for sequence loading.
SEQUENCE_PREFETCH_FACTOR = 2  # Number of batches of sequences loaded in advance by each worker.
