from __future__ import annotations

import os

IMITATION_DO_MULTITASK_LEARNING = False
IMITATION_USE_SMALL_DATASET = False
IMITATION_WANDB_LOG = True

################################################################################################################################
# Output parameters for imitation learning
IMITATION_OUTPUT_MODEL = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/imitation_models/"
os.makedirs(name=IMITATION_OUTPUT_MODEL, exist_ok=True)
IMITATION_DAGGER_OUTPUT_DIR = "/graphics/scratch2/students/nguyenlo/CarRaceOutputs/dagger_data/"
os.makedirs(name=IMITATION_DAGGER_OUTPUT_DIR, exist_ok=True)

################################################################################################################################
# Imitation training parameters
IMITATION_OPTIMIZER = "Adam"
IMITATION_EPOCHS = 200
IMITATION_LR = 0.01
IMITATION_LR_SCHEDULER_GAMMA = 0.5
IMITATION_LR_SCHEDULER_STEP_SIZE = 25
IMITATION_DROPOUT_PROB = 0.1
IMITATION_BATCH_SIZE = 64  # Train on 64 frames at once. This is equal to BATCH_SIZE.

################################################################################################################################
# Data loading parameters
IMITATION_SEQUENCE_BATCH_SIZE = 8  # Loading 8 sequences as one batch. Each sequence at most 600 frames.
IMITATION_SEQUENCE_NUM_WORKERS = 2  # How many subprocesses to use for sequence loading.
IMITATION_SEQUENCE_PREFETCH_FACTOR = 2  # Number of batches of sequences loaded in advance by each worker.

################################################################################################################################
# Backbone network parameters
IMITATION_NUM_FILTERS_ENCODER = 64  # Constants for number of filters in the encoder and decoder
IMITATION_LATENT_DIM = 6 * IMITATION_NUM_FILTERS_ENCODER  # Constants for the latent dimension of the final layer
IMITATION_STATE_DIM = 8  # Vehicle's kinematic states
IMITATION_DATASET_SAMPLING_RATE = 2  # Ignore (n-1)/n of the data

################################################################################################################################
# Single Task Network Parameters
IMITATION_FC_NUM_LAYERS = 2
IMITATION_FC_INITIAL_LAYER_SIZE = 64

################################################################################################################################
# Multi Task Network parameters
IMITATION_CURVATURE_DIMS = [64, 32, 1]
IMITATION_DESIRED_SPEED_DIMS = [64, 32, 1]
IMITATION_CTE_DIMS = [64, 32, 1]
IMITATION_HE_DIMS = [64, 32, 1]
IMITATION_STEERING_DIMS = [64, 32, 1]
IMITATION_ACCELERATION_DIMS = [64, 32, 1]

################################################################################################################################
# Training data parameters
if IMITATION_USE_SMALL_DATASET:
    IMITATION_DAGGER_BEGINNING_ITERATION_PER_LOOP = 1
    IMITATION_DAGGER_END_ITERATION_PER_LOOP = 1
else:
    IMITATION_DAGGER_BEGINNING_ITERATION_PER_LOOP = 20
    IMITATION_DAGGER_END_ITERATION_PER_LOOP = 100

################################################################################################################################
# Testing data parameters
IMITATION_TEST_DIR = "/graphics/scratch2/students/nguyenlo/CarRace/CarRaceOutputs/records/dagger_testing_data_008"

################################################################################################################################
# Validation data parameters. Validation and training data generation with DAgger are the same.
IMITATION_DAGGER_VALIDATION_START_SEED = 100000  # Start seed for the environment to generate the data with dagger.

################################################################################################################################
# Multi tasking loss scaling coefficients
if IMITATION_DO_MULTITASK_LEARNING:
    IMITATION_CHEVRON_SEGMENTATION_LOSS = 1  # Auxilliary loss: predict whether chevron markings are visible.
    IMITATION_ROAD_SEGMENTATION_LOSS = 1  # Auxilliary loss: Reconstruction loss: predict the road segmentation.
    IMITATION_CURVATURE_LOSS = 1  # Auxilliary loss: Predict the curvature of the road.
    IMITATION_DESIRED_SPEED_LOSS = 1  # Auxilliary loss: Predict the desired speed of the expert.
    IMITATION_SPEED_ERROR_LOSS = 1  # Control loss: Predict the speed error of the expert.
    IMITATION_CTE_LOSS = 1  # Control loss: Predict the cross track error of the expert.
    IMITATION_HE_LOSS = 1  # Control loss: Predict the heading error of the expert.
else:
    IMITATION_CHEVRON_SEGMENTATION_LOSS = 0  # Auxilliary loss: predict whether chevron markings are visible.
    IMITATION_ROAD_SEGMENTATION_LOSS = 0  # Auxilliary loss: Reconstruction loss: predict the road segmentation.
    IMITATION_CURVATURE_LOSS = 0  # Auxilliary loss: Predict the curvature of the road.
    IMITATION_DESIRED_SPEED_LOSS = 0  # Auxilliary loss: Predict the desired speed of the expert.
    IMITATION_SPEED_ERROR_LOSS = 0  # Control loss: Predict the speed error of the expert.
    IMITATION_CTE_LOSS = 0  # Control loss: Predict the cross track error of the expert.
    IMITATION_HE_LOSS = 0  # Control loss: Predict the heading error of the expert
IMITATION_ACCELERATION_LOSS = 1  # Control loss: Predict the acceleration of expert.
IMITATION_STEERING_LOSS = 1  # Control loss: Predict the steering angle of expert.

################################################################################################################################
# DAgger parameters
IMITATION_P_DECAY = 0.975  # Decay the probability of using the teacher model.
IMITATION_TEACHER_P_CUT_OFF = 0.05  # if p_teacher < cut_off, we just use the student model.
IMITATION_DATASET_LIMIT_PER_EPOCH = 32  # Avoid quadratic growth of the dataset since dagger produces a lot of data.
IMITATION_DATASET_RECENT_MUST_INCLUDE = 8  # Must learn current n data records.
IMITATION_STORE_ALL_RECORDS_EPOCH = 30  # Below this epoch, store all the records. Bootstrapping.
IMITATION_STORE_REWARD_THRESHOLD = 700  # Store the data if the reward is below this threshold.

#################################################################################################################################
# Sequence and state discarding parameter
IMITATION_MIN_CURVATURE_DISCARD_THRESHOLD = 0.01
IMITATION_MIN_CURVATURE_DISCARD_PROB = 0.0
IMITATION_EARLY_BREAK_NO_REWARD_STEPS = 600
IMITATION_EARLY_BREAK_MAX_CTE = 100.0
IMITATION_EARLY_BREAK_MAX_HE = 100.0

#################################################################################################################################
# Model save point parameter
IMITATION_MIN_THRESHOLD_UPLOAD = 810
