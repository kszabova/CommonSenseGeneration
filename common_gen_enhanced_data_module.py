import json
import random
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

from datasets import load_dataset


class CommonGenEnhancedDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer, enhancement_type, enhancement_file):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.enhancement = self._setup_enhancement(enhancement_type, enhancement_file)
        self.dataset = load_dataset("common_gen")
        self.setup(None)

    def setup(self, stage):
        self.train = torch.utils.data.Subset(self.dataset["train"], list(range(5)))
        self.validation = torch.utils.data.Subset(
            self.dataset["validation"], list(range(10))
        )
        self.test = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation, batch_size=self.batch_size, collate_fn=self._collate_batch
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, collate_fn=self._collate_batch
        )

    def _collate_batch(self, batch):
        inputs = []
        separator = " " + self.tokenizer.cls_token + " "
        for data in batch:
            concepts = " ".join(data["concepts"])
            sentences = []
            for concept in data["concepts"]:
                sentence = random.choice(self.enhancement.get(concept, [""]))
                sentences.append(sentence)
            concepts += (
                " "
                + self.tokenizer.cls_token
                + " "
                + (random.choice(sentences) if sentences else "")
                + " "
                + self.tokenizer.sep_token
            )
            inputs.append(concepts)

        targets = [data["target"] for data in batch]
        tokenized_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        tokenized_targets = self.tokenizer(targets, padding=True, return_tensors="pt")

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_targets["input_ids"],
        }

    def _setup_enhancement(self, enhancement_type, enhancement_file):
        if enhancement_type == "basic":
            with open(enhancement_file, "r") as file:
                return json.load(file)
        if enhancement_type == "mock":
            return {
                "ski": ["1", "2"],
                "mountain": ["3", "4"],
                "skier": ["4"],
                "wag": ["5", "6", "7"],
                "tail": ["8"],
                "dog": ["9"],
            }
        return {}

