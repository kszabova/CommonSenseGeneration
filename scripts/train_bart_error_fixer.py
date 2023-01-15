import argparse
import random
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk
import torch


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="./models")
    return parser


def get_input_tokenize_function(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["input"], padding="max_length", truncation=True)

    return tokenize_function


def get_output_tokenize_function(tokenizer):
    def tokenize_function(examples):
        tokenized = tokenizer(examples["output"], padding="max_length", truncation=True)
        eos_token = tokenizer.eos_token_id
        # shift tokens right
        for input_ids in tokenized["input_ids"]:
            input_ids.insert(0, eos_token)
            input_ids.pop()
        return {
            "decoder_input_ids": tokenized["input_ids"],
            "decoder_attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }

    return tokenize_function


def main():
    parser = get_argparser()
    args = parser.parse_args()

    # set up model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # load data
    dataset = load_from_disk("./data/error_fixer_data.data")
    train_ds = (
        dataset["train"]
        .map(get_input_tokenize_function(tokenizer), batched=True)
        .map(get_output_tokenize_function(tokenizer), batched=True)
    )
    test_ds = (
        dataset["test"]
        .map(get_input_tokenize_function(tokenizer), batched=True)
        .map(get_output_tokenize_function(tokenizer), batched=True)
    )
    small_train_ds = train_ds.select(random.sample(range(len(train_ds)), 10000))

    # set up training
    training_args = TrainingArguments(
        output_dir="error_fixer_trainer",
        evaluation_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_ds,
        eval_dataset=test_ds,
    )

    # train model
    trainer.train()
    trainer.save_model(f"{args.models_dir}/error_fixer_model")


if __name__ == "__main__":
    main()

