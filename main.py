import argparse
import random
import torch
import pytorch_lightning as pl
import logging

from transformers import BartTokenizer, BartForConditionalGeneration

from lightning_modules import (
    CommonGenDataModuleFromHub,
    CommonGenDataModuleFromDisk,
    CommonGenModule,
)

from models import BartDoubleHeadsModel

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
    # set up model
    model = None
    if config.model_type == "generation":
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    elif config.model_type == "double_heads":
        model = BartDoubleHeadsModel.from_pretrained("facebook/bart-base")
        model.set_mc_loss_weight(config.mc_loss_weight)
    # set up data loader
    data = None
    if config.data_type == "hub":
        data = CommonGenDataModuleFromHub(tokenizer, config)
    elif config.data_type == "disk":
        data = CommonGenDataModuleFromDisk(tokenizer, config)

    kwargs = {
        "tokenizer": tokenizer,
        "model": model,
        "hparams": {"learning_rate": config.learning_rate},
        "log_interval": config.log_interval,
    }
    if config.pretrained_ckpt_path:
        common_gen_model = CommonGenModule.load_from_checkpoint(
            config.pretrained_ckpt_path, **kwargs
        )
    else:
        common_gen_model = CommonGenModule(**kwargs)

    model_name = config.model_name + f"{iteration:02d}"
    validation_output_file = f"val_output_{model_name}.txt"
    callbacks = [
        LossCallback(config.log_interval),
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

    return trainer, common_gen_model, data


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
