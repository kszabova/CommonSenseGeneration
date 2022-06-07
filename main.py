import argparse
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import logging

from transformers import BartTokenizer, BartForConditionalGeneration

from common_gen_model import CommonGenModel
from common_gen_data_module import CommonGenDataModule

from callbacks import LossCallback, TensorBoardCallback, ValidationCallback


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_epochs", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="baseline")
    parser.add_argument(
        "--val_output",
        type=str,
        default="val_output.txt",
        help="File path where validation output should be stored.",
    )
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--log_interval", type=int, default=5, help="Interval of logging for Trainer."
    )
    parser.add_argument("--gpus", type=int, default=0)

    return parser


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def main():

    parser = get_arg_parser()
    args = parser.parse_args()

    setup_logging()

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    common_gen_data = CommonGenDataModule(args.batch_size, tokenizer)
    common_gen_model = CommonGenModel(
        args.lr, tokenizer, model, None, args.log_interval
    )

    checkpoint = pl.callbacks.ModelCheckpoint(f"./checkpoints/{args.model_name}/")
    callbacks = [
        LossCallback(args.log_interval),
        TensorBoardCallback(args.model_name),
        ValidationCallback(args.val_output, args.min_epochs),
        checkpoint,
    ]

    trainer = pl.Trainer(
        default_root_dir=".",
        gpus=args.gpus,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        auto_lr_find=False,
        callbacks=callbacks,
        log_every_n_steps=args.log_interval,
        enable_progress_bar=False,
    )

    trainer.fit(common_gen_model, common_gen_data)
    trainer.validate(common_gen_model, common_gen_data)


if __name__ == "__main__":
    main()
