import argparse
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk
import evaluate
import numpy as np


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="./models")
    return parser


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)


def get_tokenize_function(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    return tokenize_function


def main():
    parser = get_argparser()
    args = parser.parse_args()

    # set up model and tokenizer
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # load data
    dataset = load_from_disk("./data/classifier_data.data")
    train_ds = (
        dataset["train"]
        .map(get_tokenize_function(tokenizer), batched=True)
        .rename_column("class", "label")
    )
    test_ds = (
        dataset["test"]
        .map(get_tokenize_function(tokenizer), batched=True)
        .rename_column("class", "label")
    )

    # set up training
    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    # train model
    trainer.train()
    trainer.save_model(f"{args.models_dir}/classifier_model")


if __name__ == "__main__":
    main()

