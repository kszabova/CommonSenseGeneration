from typing import NamedTuple
from dataclasses import dataclass

import yaml


@dataclass
class Config:
    # logging
    log_interval: int = 100
    log_level: str = "INFO"

    # pretraining
    pretrained_ckpt_path: str = None

    # enhancement
    enh_type: str = None
    enh_path: str = None

    # postprocessing
    postprocess_type: str = None
    postprocess_model_path: str = None

    # data
    csv_path: str = None
    valid_path: str = None

    # training:
    learning_rate: float = 1e-4
    batch_size: int = 3
    min_epochs: int = 1
    max_epochs: int = 1

    # environment
    gpus: int = 1
    base_dir: str = None
    rnd_seed: int = 42

    # output
    model_name: str = None
    ckpt_path: str = None

    iterations: int = 1
    remove_dev_duplicates: bool = False

    @staticmethod
    def load_config(path):
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return Config(**config)
