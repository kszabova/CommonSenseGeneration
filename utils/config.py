from typing import NamedTuple

import yaml


class Config(NamedTuple):
    # logging
    log_interval: int
    log_level: str

    # pretraining
    pretrained_ckpt_path: str

    # enhancement
    enh_type: str
    enh_path: str

    # data
    csv_path: str
    valid_path: str

    # training:
    learning_rate: float
    batch_size: int
    min_epochs: int
    max_epochs: int

    # environment
    gpus: int
    base_dir: str
    rnd_seed: int

    # output
    model_name: str
    ckpt_path: str

    iterations: int
    remove_dev_duplicates: bool

    @staticmethod
    def load_config(path):
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return Config(**config)
