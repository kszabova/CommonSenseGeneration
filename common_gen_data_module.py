from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

from datasets import load_dataset


class CommonGenDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataset = load_dataset("common_gen")
        self.setup(None)

    def setup(self, stage):
        self.train = torch.utils.data.Subset(self.dataset["train"], list(range(50)))
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
        concepts = [" ".join(data["concepts"]) for data in batch]
        targets = [data["target"] for data in batch]
        tokenized_concepts = self.tokenizer(concepts, padding=True, return_tensors="pt")
        tokenized_targets = self.tokenizer(targets, padding=True, return_tensors="pt")

        return {
            "input_ids": tokenized_concepts["input_ids"],
            "attention_mask": tokenized_concepts["attention_mask"],
            "labels": tokenized_targets["input_ids"],
        }
