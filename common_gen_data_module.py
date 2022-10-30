from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import random
import json

from datasets import load_dataset

from csv_dataset import CSVDataset


class CommonGenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        tokenizer,
        enhancement_type=None,
        enhancement_file=None,
        csv=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataset = load_dataset("common_gen")
        self.enhancement_type = enhancement_type
        self.enhancement = self._setup_enhancement(enhancement_type, enhancement_file)
        self.csv = csv
        self.setup(None)

    def setup(self, stage):
        if self.csv:
            self.train = CSVDataset(self.csv)
        else:
            self.train = torch.utils.data.Subset(self.dataset["train"], range(30))
        self.validation = torch.utils.data.Subset(self.dataset["validation"], range(15))
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
        total_keywords = 0
        total_pairs_found = 0
        for data in batch:
            input, pairs_found = self._perform_enhancement_on_input(data["concepts"])
            total_keywords += 1
            total_pairs_found += pairs_found
            inputs.append(input)

        targets = [data["target"] for data in batch]
        tokenized_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        tokenized_targets = self.tokenizer(targets, padding=True, return_tensors="pt")

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_targets["input_ids"],
            "keywords": total_keywords,
            "pairs_found": total_pairs_found,
        }

    def _perform_enhancement_on_input(self, concepts):
        if not self.enhancement_type:
            return " ".join([str(concept) for concept in concepts]), 0
        elif self.enhancement_type == "basic":
            # select a random sentence for a random concept
            input = " ".join(concepts)
            sentences = []
            for concept in concepts:
                concept_sentences = self.enhancement.get(concept, {}).get(
                    "sentences", []
                )
                if concept_sentences:
                    sentences.append(random.choice(concept_sentences))
            input += (
                " "
                + self.tokenizer.cls_token
                + " "
                + (random.choice(sentences) if sentences else "")
                + " "
                + self.tokenizer.sep_token
            )
            return input, 0
        elif self.enhancement_type == "all_keywords":
            # select a random sentence for each of the concepts
            input = " ".join(concepts)
            sentences = []
            for concept in concepts:
                concept_sentences = self.enhancement.get(concept, {}).get(
                    "sentences", []
                )
                if concept_sentences:
                    sentences.append(random.choice(concept_sentences))
            input += (
                "".join(
                    [
                        f" {self.tokenizer.cls_token} {sentence} "
                        for sentence in sentences
                    ]
                )
                + self.tokenizer.sep_token
            )
            return input, 0
        elif self.enhancement_type == "pair":
            # select a sentence that contains at least two of the concepts
            input = " ".join(concepts)
            sentences = []
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    intersection = (
                        self.enhancement.get(concepts[i], {})
                        .get(concepts[j], {})
                        .get("sentences", [])
                    )
                    sentences.extend(intersection)
            sentence = random.choice(sentences) if sentences else ""
            input += (
                " "
                + self.tokenizer.cls_token
                + " "
                + sentence
                + " "
                + self.tokenizer.sep_token
            )
            return input, 1 if sentence else 0

    def _select_sentence_with_multiple_words(self, keywords, sentences, threshold):
        kw_set = set(keywords)
        valid_sentences = []
        for sentence in sentences:
            sentence_set = set(sentence.split())
            isection = kw_set.intersection(sentence_set)
            if len(isection) >= threshold:
                valid_sentences.append(sentence)
        if valid_sentences:
            return random.choice(valid_sentences)
        return None

    def _setup_enhancement(self, enhancement_type, enhancement_file):
        if enhancement_type in ["basic", "all_keywords", "pair"]:
            with open(enhancement_file, "r") as file:
                return json.load(file)
        return {}
