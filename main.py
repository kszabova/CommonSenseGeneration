import argparse
import yaml
import random
import torch
import pytorch_lightning as pl
import logging

from transformers import BartTokenizer, BartForConditionalGeneration

from common_gen_model import CommonGenModel
from common_gen_data_module import CommonGenDataModule

from callbacks import (
    CoverageCallback,
    LossCallback,
    TensorBoardCallback,
    ValidationCallback,
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


def setup_model(config, iteration):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    common_gen_data = CommonGenDataModule(
        config["training"]["hparams"]["batch_size"],
        tokenizer,
        enhancement_type=config["enhancement"]["type"],
        enhancement_file=config["enhancement"]["file_path"],
        csv=config["training"]["data"]["csv_file_path"],
    )

    kwargs = {
        "learning_rate": config["training"]["hparams"]["learning_rate"],
        "tokenizer": tokenizer,
        "model": model,
        "hparams": None,
        "log_interval": config["logging"]["log_interval"],
    }
    if config["pretraining"]["ckpt_path"]:
        common_gen_model = CommonGenModel.load_from_checkpoint(
            config["pretraining"]["ckpt_path"], **kwargs
        )
    else:
        common_gen_model = CommonGenModel(**kwargs)

    model_name = config["output"]["model_name"] + f"{iteration:02d}"
    callbacks = [
        CoverageCallback(config["enhancement"]["type"] == "pair"),
        LossCallback(config["logging"]["log_interval"]),
        TensorBoardCallback(model_name),
        ValidationCallback(
            config["output"]["val_output_name"],
            config["training"]["hparams"]["min_epochs"],
        ),
        ModelCheckpoint(f"./checkpoints/{model_name}/", save_weights_only=True),
    ]

    trainer = pl.Trainer(
        default_root_dir=".",
        gpus=config["infrastructure"]["gpus"],
        min_epochs=config["training"]["hparams"]["min_epochs"],
        max_epochs=config["training"]["hparams"]["max_epochs"],
        auto_lr_find=False,
        callbacks=callbacks,
        log_every_n_steps=config["logging"]["log_interval"],
        enable_progress_bar=False,
    )

    return trainer, common_gen_model, common_gen_data


def model_loop(trainer, model, data):
    trainer.fit(model, data)
    trainer.validate(model, data)


def main():

    parser = get_arg_parser()
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["common_gen_config"]

    # fix random seeds
    seed = config["rnd_seed"]
    torch.manual_seed(seed)
    random.seed(seed)

    setup_logging(config["logging"]["log_level"])
    logger = logging.getLogger("main")

    iterations = config["iterations"]
    for i in range(iterations):
        logger.info(f"Training model {config['output']['model_name']}{i:02d}")
        trainer, model, data = setup_model(config, i)
        model_loop(trainer, model, data)


if __name__ == "__main__":
    main()
