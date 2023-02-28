import argparse
import random
import transformers
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk
import torch
import copy


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument("--data_path", type=str, default="./data/error_fixer_data.data")
    parser.add_argument("--model_name", type=str, default="error_fixer_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_concepts", action="store_true")
    return parser


def get_input_tokenize_function(tokenizer, include_concepts=False):
    def tokenize_function(example):
        if include_concepts:
            concepts_string = " ".join(example["concepts"])
            input_string = f"{concepts_string} {tokenizer.cls_token} {example['input']}"
            return tokenizer(input_string, padding="max_length", truncation=True)
        else:
            return tokenizer(example["input"], padding="max_length", truncation=True)

    return tokenize_function


def get_output_tokenize_function(tokenizer):
    def tokenize_function(examples):
        tokenized = tokenizer(examples["output"], padding="max_length", truncation=True)
        eos_token = tokenizer.eos_token_id
        # shift tokens right
        decoder_input_ids = copy.deepcopy(tokenized["input_ids"])
        for input_ids in decoder_input_ids:
            input_ids.insert(0, eos_token)
            input_ids.pop()
        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }

    return tokenize_function


def main():
    parser = get_argparser()
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    transformers.set_seed(args.seed)

    # set up model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # load data
    dataset = load_from_disk(args.data_path)
    train_ds = (
        dataset["train"]
        .map(
            get_input_tokenize_function(tokenizer, args.include_concepts), batched=False
        )
        .map(get_output_tokenize_function(tokenizer), batched=True)
    )
    test_ds = (
        dataset["test"]
        .map(
            get_input_tokenize_function(tokenizer, args.include_concepts), batched=False
        )
        .map(get_output_tokenize_function(tokenizer), batched=True)
    )

    # set up training
    training_args = TrainingArguments(
        output_dir="error_fixer_trainer",
        evaluation_strategy="epoch",
        save_strategy="no",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds,
    )

    # train model
    trainer.train()
    trainer.save_model(f"{args.models_dir}/{args.model_name}")


if __name__ == "__main__":
    main()

