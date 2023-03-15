import json

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset

from utils.config import Config
from utils.input_enhanement import InputEnhancer


class CommonGenDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, config: Config):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.enhancement = self.setup_enhancer(config.enh_type, config.enh_path)
        self.setup_data()

    def setup_enhancer(self, enhancement_type, enhancement_file):
        return InputEnhancer(
            enhancement_type,
            enhancement_file,
            self.tokenizer.cls_token,
            self.tokenizer.sep_token,
        )

    def setup_data(self):
        raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_batch,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_batch,
        )

    def test_dataloader(self):
        if self.test is None:
            return None
        return DataLoader(
            self.test, batch_size=self.config.batch_size, collate_fn=self._collate_batch
        )

    def _perform_enhancement_on_input(self, concepts):
        return self.enhancement.get_enhanced_input(concepts)

    def _collate_batch(self, batch):
        concepts = []
        inputs = []
        for data in batch:
            conc = data["concepts"]
            concepts.append(conc)
            input = self._perform_enhancement_on_input(conc)
            inputs.append(input)

        tokenized_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")

        label_dict = self._get_label_dict(batch)

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "concepts": concepts,
        } | label_dict

    def _get_label_dict(self, batch):
        # return a dictionary containing either {"labels"} or {"lm_labels" and "mc_labels"}
        raise NotImplementedError()


class CommonGenDataModuleFromHub(CommonGenDataModule):
    def __init__(self, tokenizer, config: Config):
        super().__init__(tokenizer, config)

    def setup_data(self):
        dataset = load_dataset("common_gen")
        self.train = dataset["train"]
        self.validation = _select_unique_inputs(dataset["validation"], self.config,)
        self.test = dataset["test"]

    def _get_label_dict(self, batch):
        targets = [data["target"] for data in batch]
        tokenized_targets = self.tokenizer(targets, padding=True, return_tensors="pt")
        return {
            "labels": tokenized_targets["input_ids"],
        }


class CommonGenDataModuleFromDisk(CommonGenDataModule):
    def __init__(self, tokenizer, config: Config):
        super().__init__(tokenizer, config)

    def setup_data(self):
        dataset = load_from_disk(self.config.dataset_path)
        self.train = dataset["train"]
        self.validation = _select_unique_inputs(dataset["test"], self.config)
        self.test = None

    def _get_label_dict(self, batch):
        targets = [data["input"] for data in batch]
        tokenized_targets = self.tokenizer(targets, padding=True, return_tensors="pt")
        mc_labels = [data["contains_all_concepts"] for data in batch]
        return {
            "lm_labels": tokenized_targets["input_ids"],
            "mc_labels": mc_labels,
        }


def _select_unique_inputs(data, config):
    if not config.remove_dev_duplicates:
        return data

    seen_concepts = set()
    unique_examples = []
    references = {}
    for i, ex in enumerate(data):
        concept_str = " ".join(ex["concepts"])
        conceptset_data = references.setdefault(concept_str, [])
        # add reference if the dataset does not contain the "contains_all_concepts" key
        # or if it does and the value is True
        if "contains_all_concepts" not in ex or ex["contains_all_concepts"]:
            reference_key = "target" if "target" in ex else "input"
            conceptset_data.append(ex[reference_key])
        if concept_str in seen_concepts:
            continue
        seen_concepts.add(concept_str)
        unique_examples.append(i)
    with open(config.valid_path, "w") as file:
        file.write(json.dumps(references, indent=4))
    return torch.utils.data.Subset(data, unique_examples)

