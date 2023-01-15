import argparse
import yaml
import random
import torch
import pytorch_lightning as pl
import logging

from transformers import BartTokenizer, BartForConditionalGeneration

from common_gen_model import CommonGenModel
from common_gen_data_module import CommonGenDataModule

from utils.config import Config

from callbacks import (
    CoverageCallback,
    LossCallback,
    TensorBoardCallback,
    ValidationCallback,
    SavePretrainedModelCallback,
)
from pytorch_lightning.callbacks import ModelCheckpoint


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")

    return parser


def setup_logging(level):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )


def setup_model(config: Config, iteration):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    common_gen_data = CommonGenDataModule(tokenizer, config)

    kwargs = {
        "learning_rate": config.learning_rate,
        "tokenizer": tokenizer,
        "model": model,
        "hparams": None,
        "log_interval": config.log_interval,
    }
    if config.pretrained_ckpt_path:
        common_gen_model = CommonGenModel.load_from_checkpoint(
            config.pretrained_ckpt_path, **kwargs
        )
    else:
        common_gen_model = CommonGenModel(**kwargs)

    model_name = config.model_name + f"{iteration:02d}"
    validation_output_file = f"val_output_{model_name}.txt"
    callbacks = [
        CoverageCallback(config.enh_type == "pair"),
        LossCallback(config.log_interval),
        TensorBoardCallback(model_name),
        ValidationCallback(validation_output_file, config),
        ModelCheckpoint(f"{config.ckpt_path}/{model_name}/", save_weights_only=True),
        SavePretrainedModelCallback(f"{config.ckpt_path}/{model_name}"),
    ]

    trainer = pl.Trainer(
        default_root_dir=".",
        gpus=config.gpus,
        min_epochs=config.min_epochs,
        max_epochs=config.max_epochs,
        auto_lr_find=False,
        callbacks=callbacks,
        log_every_n_steps=config.log_interval,
        enable_progress_bar=False,
    )

    return trainer, common_gen_model, common_gen_data


def model_loop(trainer, model, data):
    trainer.fit(model, data)
    trainer.validate(model, data)


def main():

    parser = get_arg_parser()
    args = parser.parse_args()

    config = Config.load_config(args.config)

    # fix random seeds
    seed = config.rnd_seed
    torch.manual_seed(seed)
    random.seed(seed)

    setup_logging(config.log_level)
    logger = logging.getLogger("main")

    iterations = config.iterations
    for i in range(iterations):
        logger.info(f"Training model {config.model_name}{i:02d}")
        trainer, model, data = setup_model(config, i)
        if config.train:
            model_loop(trainer, model, data)


if __name__ == "__main__":
    main()
