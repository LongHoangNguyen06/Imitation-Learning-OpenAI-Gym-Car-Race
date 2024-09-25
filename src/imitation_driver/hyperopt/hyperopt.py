import os
import sys
import traceback

import numpy as np
import torch.multiprocessing as mp

import wandb
from src.imitation_driver.training.train import train
from src.utils import io_utils
from src.utils.conf_utils import get_default_conf
from src.utils.utils import set_deterministic

os.environ["WANDB_TIMEOUT"] = "300"


def wandb_config_to_config(wandb_config, conf):
    for key in wandb_config.keys():
        conf[key] = wandb_config[key]
    return conf


def recompute_conf(conf):
    conf.IMITATION_LATENT_DIM = 6 * conf.IMITATION_NUM_FILTERS_ENCODER


def exception_handling_train():
    set_deterministic()
    run_id: str = io_utils.get_current_time_formatted()
    with wandb.init(
        project="CarRace",
        name=run_id,
    ) as run:
        try:
            conf = wandb_config_to_config(run.config, get_default_conf())
            recompute_conf(conf)
            train(conf=conf, run_id=run_id)
        except Exception as _:  # pylint: disable=broad-except
            print("An error occurred during training.")
            traceback.print_exc()


def hyper_parameter_optimize(sweep_id=None):
    if sweep_id is None:
        sweep_id = wandb.sweep(
            {
                "project": "CarRace",
                "name": "CarRace",
                "method": "bayes",
                "metric": {
                    "goal": "minimize",
                    "name": "best_validate_reward",
                },
                "parameters": {
                    "IMITATION_OPTIMIZER": {"values": ["Adam", "AdamW", "SGD"]},
                    "IMITATION_DROPOUT_PROB": {"values": np.linspace(0.1, 0.9, 9)},
                    "IMITATION_NUM_FILTERS_ENCODER": {"values": [32, 64, 128]},
                    "IMITATION_FC_NUM_LAYERS": {"values": [1, 2, 3, 4]},
                    "IMITATION_FC_INITIAL_LAYER_SIZE": {"values": [16, 32, 64]},
                    "IMITATION_LR": {"values": np.logspace(-5, -1, 5)},
                    "IMITATION_P_DECAY": {"values": np.linspace(0.1, 0.9, 9)},
                    "IMITATION_TEACHER_P_CUT_OFF": {"values": np.linspace(0.1, 0.9, 9)},
                    "IMITATION_DATASET_LIMIT_PER_EPOCH": {"values": [1, 10, 100, 1000]},
                    "IMITATION_DATASET_RECENT_MUST_INCLUDE": {"values": [8, 16, 32]},
                    "IMITATION_STORE_REWARD_THRESHOLD": {"values": [500, 600, 700, 800, 900, 1000]},
                    "IMITATION_BATCH_SIZE": {"values": [32, 64, 128, 256]},
                    "IMITATION_MIN_CURVATURE_DISCARD_PROB": {"values": np.linspace(0.1, 0.9, 9)},
                    "IMITATION_EARLY_BREAK_NO_REWARD_STEPS": {"values": [5, 10, 15, 20]},
                    "IMITATION_EARLY_BREAK_MAX_CTE": {"values": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
                    "IMITATION_EARLY_BREAK_MAX_HE": {"values": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
                },
            },
        )
        with open("tools/agent.sh", "w", encoding="utf-8") as f:
            f.write(f'screen -S hyperopt -d -m bash -c "python3 -m src.imitation_driver.hyperopt.hyperopt {sweep_id}"')

    wandb.agent(
        sweep_id,
        exception_handling_train,
        count=50,
        project="CarRace",
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if len(sys.argv) > 1:
        print(f"Starting new agent for sweep {sys.argv[1]}")
        hyper_parameter_optimize(sys.argv[1])
    else:
        hyper_parameter_optimize()
