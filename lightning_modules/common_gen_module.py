import logging

import transformers

import torch
import pytorch_lightning as pl


class CommonGenModule(pl.LightningModule):
    IGNORED_BATCH_KEYS = ["concepts", "reference"]
    logger = logging.getLogger("lightning")

    def __init__(self, model, tokenizer, hparams, log_interval):
        super().__init__()
        self.model: transformers.BartForConditionalGeneration = model
        self.tokenizer = tokenizer
        self.learning_rate = hparams["learning_rate"]
        self.log_interval = log_interval

        self.bleu_data = {}

    def forward(
        self, **kwargs,
    ):
        kwargs = {k: v for k, v in kwargs.items() if k not in self.IGNORED_BATCH_KEYS}
        return self.model(**kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output[0]

        if batch_idx % self.log_interval == 0:
            examples = self._get_examples(batch)

            return {
                "loss": loss,
                "examples": examples,
            }

        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        self.log("loss", training_step_outputs["loss"])
        if "examples" in training_step_outputs:
            srcs = training_step_outputs["examples"]["src"]
            refs = training_step_outputs["examples"]["ref"]
            preds = training_step_outputs["examples"]["pred"]
            for src, ref, pred in zip(srcs, refs, preds):
                self.logger.info(f"SOURCE: {src}")
                self.logger.info(f"REFERENCE: {ref}")
                self.logger.info(f"PREDICTION: {pred}")

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output[0]

        examples = self._get_examples(batch)

        for src, ref, pred in zip(examples["src"], examples["ref"], examples["pred"]):
            self.bleu_data[src] = self.bleu_data.get(src, {"preds": [], "refs": []})
            self.bleu_data[src]["preds"].append(pred)
            self.bleu_data[src]["refs"].append(ref)

        return {
            "val_loss": loss,
            "examples": examples,
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss)
        self.logger.info(f"Validation loss: {val_loss}")

    def test_step(self):
        pass

    def predict_step(self):
        pass

    def _get_bleu_data(self):
        data = []
        for value in self.bleu_data.values():
            refs = value["refs"]
            for pred in set(value["preds"]):
                data.append((pred, refs))
        return data

    def _get_examples(self, batch):
        reference_key = "labels" if "labels" in batch else "lm_labels"
        src_text = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        ref_text = self.tokenizer.batch_decode(
            batch[reference_key], skip_special_tokens=True
        )
        generated_ids = self.model.generate(batch["input_ids"])
        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return {
            "src": src_text,
            "ref": ref_text,
            "pred": generated_text,
            "concepts": batch["concepts"],
        }
