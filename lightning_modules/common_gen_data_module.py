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
            collate_fn=self._collate_fn_train,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn_valid,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn_valid,
        )

    def _perform_enhancement_on_input(self, concepts):
        return self.enhancement.get_enhanced_input(concepts)

    def _collate_fn_train(self, batch):
        concepts = []
        inputs = []
        for data in batch:
            conc = data["concepts"]
            concepts.append(conc)
            input = self._perform_enhancement_on_input(conc)
            inputs.append(input)

        tokenized_inputs = self.tokenizer(
            inputs, padding="max_length", return_tensors="pt"
        )

        label_dict = self._get_label_dict(batch)

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "concepts": concepts,
        } | label_dict

    def _collate_fn_valid(self, batch):
        concepts = []
        inputs = []
        targets = []
        for data in batch:
            conc = data["concepts"]
            concepts.append(conc)
            input = self._perform_enhancement_on_input(conc)
            inputs.append(input)
            targets.append(data["target"])

        tokenized_inputs = self.tokenizer(
            inputs, padding="max_length", return_tensors="pt"
        )
        tokenized_targets = self.tokenizer(
            targets, padding="max_length", return_tensors="pt"
        )

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_targets["input_ids"],
            "concepts": concepts,
        }

    def _get_label_dict(self, batch):
        # return a dictionary containing either {"labels"} or {"labels" and "mc_labels"}
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
        tokenized_targets = self.tokenizer(
            targets, padding="max_length", return_tensors="pt"
        )
        return {
            "labels": tokenized_targets["input_ids"],
        }


class CommonGenDataModuleFromDisk(CommonGenDataModule):
    def __init__(self, tokenizer, config: Config):
        super().__init__(tokenizer, config)

    def setup_data(self):
        dataset = load_from_disk(self.config.dataset_path)
        self.train = torch.utils.data.ConcatDataset([dataset["train"], dataset["test"]])
        self.validation = _select_unique_inputs(
            load_dataset("common_gen", split="validation"), self.config,
        )
        self.test = load_dataset("common_gen", split="test")

    def _get_label_dict(self, batch):
        targets = [data["input"] for data in batch]
        tokenized_targets = self.tokenizer(
            targets, padding="max_length", return_tensors="pt"
        )
        mc_labels = [data["contains_all_concepts"] for data in batch]
        return {
            "labels": tokenized_targets["input_ids"],
            "mc_labels": torch.FloatTensor(mc_labels),
            "reference": [data["output"] for data in batch],
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
        conceptset_data.append(ex["target"])
        if concept_str in seen_concepts:
            continue
        seen_concepts.add(ex["concept_set_idx"])
        unique_examples.append(i)
    with open(config.valid_path, "w") as file:
        file.write(json.dumps(references, indent=4))
    return torch.utils.data.Subset(data, unique_examples)

