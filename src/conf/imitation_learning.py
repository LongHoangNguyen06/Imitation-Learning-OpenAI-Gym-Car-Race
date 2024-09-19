from __future__ import annotations

import os

DO_MULTITASK_LEARNING = True
USE_SMALL_DATASET = False
WANDB_LOG = True
TENSOR_BOARD_LOG = False
TENSOR_BOARD_EXPENSIVE_LOG = False

################################################################################################################################
# Output parameters for imitation learning
IMITATION_OUTPUT_MODEL = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/imitation_models/"
os.makedirs(name=IMITATION_OUTPUT_MODEL, exist_ok=True)

################################################################################################################################
# Imitation training parameters
IMITATION_LR = 0.01
IMITATION_LR_SCHEDULER_GAMMA = 0.5
IMITATION_CHEVRON_VISIBLE_THRESHOLD = (
    10  # How many pixels being red/white to be considered a chevron marking is visible on the road
)
IMITATION_DROPOUT_PROB = 0.1

################################################################################################################################
# Backbone network parameters
IMITATION_NUM_FILTERS_ENCODER = 64  # Constants for number of filters in the encoder and decoder
IMITATION_LATENT_DIM = 6 * IMITATION_NUM_FILTERS_ENCODER  # Constants for the latent dimension of the final layer

################################################################################################################################
# Prediction network parameters
IMITATION_CURVATURE_DIMS = [32, 16, 1]
IMITATION_STEERING_DIMS = [64, 32, 1]
IMITATION_ACCELERATION_DIMS = [64, 32, 1]

################################################################################################################################
# Imitation data parameters
if USE_SMALL_DATASET:
    DAGGER_BEGINNING_ITERATION_PER_LOOP = 1
    DAGGER_END_ITERATION_PER_LOOP = 1
else:
    DAGGER_BEGINNING_ITERATION_PER_LOOP = 20
    DAGGER_END_ITERATION_PER_LOOP = 100

# Multi tasking loss scaling coefficients
if DO_MULTITASK_LEARNING:
    IMITATION_CHEVRON_SEGMENTATION_LOSS = 1  # Auxilliary loss: predict whether chevron markings are visible.
    IMITATION_ROAD_SEGMENTATION_LOSS = 1  # Auxilliary loss: Reconstruction loss: predict the road segmentation.
    IMITATION_CURVATURE_LOSS = 1  # Auxilliary loss: Predict the curvature of the road.
else:
    IMITATION_CHEVRON_SEGMENTATION_LOSS = 0  # Auxilliary loss: predict whether chevron markings are visible.
    IMITATION_ROAD_SEGMENTATION_LOSS = 0  # Auxilliary loss: Reconstruction loss: predict the road segmentation.
    IMITATION_CURVATURE_LOSS = 0  # Auxilliary loss: Predict the curvature of the road.
IMITATION_ACCELERATION_LOSS = 1  # Control loss: Predict the acceleration of expert.
IMITATION_STEERING_LOSS = 1  # Control loss: Predict the steering angle of expert.

################################################################################################################################
# DAgger parameters
IMITATION_EPOCHS = 5000
DAGGER_OUTPUT_DIR = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/dagger_data/"
os.makedirs(name=DAGGER_OUTPUT_DIR, exist_ok=True)
IMITATION_LR_SCHEDULER_STEP_SIZE = 25
IMITATION_P_DECAY = 0.975  # Decay the probability of using the teacher model.
IMITATION_TEACHER_P_CUT_OFF = 0.05  # if p_teacher < cut_off, we just use the student model.
IMITATION_DATASET_LIMIT_PER_EPOCH = 32  # Avoid quadratic growth of the dataset since dagger produces a lot of data.
IMITATION_DATASET_RECENT_MUST_INCLUDE = 8  # Must learn current n data records.
IMITATION_STORE_ALL_RECORDS_EPOCH = 30  # Below this epoch, store all the records. Bootstrapping.
IMITATION_STORE_REWARD_THRESHOLD = 600  # Store the data if the reward is below this threshold.
IMITATION_START_SEED = 100000  # Start seed for the environment to generate the data with dagger.

################################################################################################################################
# Data loading parameters
SEQUENCE_BATCH_SIZE = 8  # Loading 32 sequences as one batch. Each sequence at most 600 frames.
STATE_BATCH_SIZE = 64  # Train on 256 frames at once. This is equal to BATCH_SIZE.
SEQUENCE_NUM_WORKERS = 2  # How many subprocesses to use for sequence loading.
SEQUENCE_PREFETCH_FACTOR = 2  # Number of batches of sequences loaded in advance by each worker.
