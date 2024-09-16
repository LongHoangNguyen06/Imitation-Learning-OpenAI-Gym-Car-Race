from __future__ import annotations

import multiprocessing

DO_MULTITASK_LEARNING = True  ##### DEBUG #####: change this!
IMITATION_SMALL_DATA = False  ##### DEBUG #####: change this!
DAGGER_TRAINING = True  ##### DEBUG #####: change this!
WANDB_LOG = True  ##### DEBUG #####: change this!

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
RECORD_OUTPUT_DIR = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/records"
MAX_WORKERS = max(multiprocessing.cpu_count() - 1, 4)

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

################################################################################################################################
# Output parameters for imitation learning
IMITATION_OUTPUT_MODEL = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/imitation_models/"
IMITATION_TENSOR_BOARD_WRITING_FREQUENCY = 1000

################################################################################################################################
# Imitation training parameters
IMITATION_EPOCHS = 50
IMITATION_LR = 0.01
IMITATION_LR_SCHEDULER_STEP_SIZE = 5
IMITATION_LR_SCHEDULER_GAMMA = 0.75
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
IMITATION_BATCH_SIZE = 128
IMITATION_MAX_MEMORY = 500_000
USE_RGB = False
if IMITATION_SMALL_DATA:
    IMITATION_TRAINING_SETS = [
        "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/records/benchmark_004/",
    ]
    IMITATION_TESTING_SETS = ["/graphics/scratch2/students/nguyenlo/CarRaceOutputs/records/benchmark_004/"]
    IMITATION_EVALUATION_SEEDS = list(range(1000, 1001))
else:
    IMITATION_TRAINING_SETS = [
        "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/redistributed/benchmark_000/",  # Symlink data
        "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/redistributed/benchmark_001/",  # Symlink data
        "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/redistributed/benchmark_002/",  # Symlink data
        "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/redistributed/benchmark_003/",  # Symlink data
        "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/redistributed/benchmark_004/",  # Symlink data
        "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/redistributed/benchmark_005/",  # Symlink data
        "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/redistributed/benchmark_006/",  # Symlink data
    ]
    IMITATION_TESTING_SETS = [
        # "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/records/benchmark_002/",
    ]
    IMITATION_EVALUATION_SEEDS = list(range(1000, 1100))
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
DAGGER_START_DEMO_SEED = 10000
DAGGER_TRAINING_NUM_DATALOADER = 10
DAGGER_TRAINING_NUM_DATA_PER_DATALOADER = 200
DAGGER_LR_SCHEDULER_STEP_SIZE = 25
DAGGER_P = 0.98
DAGGER_SKIP_VALIDATION_FIRST_N_EPOCHS = 100