from torchmetrics import BLEUScore
import pytorch_lightning as pl
import torch

import logging


class CommonGenModel(pl.LightningModule):
    logger = logging.getLogger("lightning")

    def __init__(self, learning_rate, tokenizer, model, hparams, log_interval):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate

        self.log_interval = log_interval

        self.total_keywords = 0
        self.total_pairs_found = 0

        self.bleu_data = {}

    # Do a forward pass through the model
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Load batch data
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        concepts = batch["concepts"]
        # Shift the decoder tokens right
        # Decoder starts generating from a <eos> token
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

        # Run the model
        outputs = self(
            src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        logits = outputs[0]
        # Calculate loss
        loss_fx = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fx(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1))

        tb_log = {"train_loss": loss.detach()}

        self.total_keywords += batch.get("keywords", 0)
        self.total_pairs_found += batch.get("pairs_found", 0)

        # Generate sentences
        if batch_idx % self.log_interval == 0:
            src_text = self.tokenizer.batch_decode(src_ids, skip_special_tokens=False)
            ref_text = self.tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
            generated_ids = self.model.generate(src_ids)
            generated_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            bleu_ref = [[ref] for ref in ref_text]
            bleu_score = BLEUScore()(generated_text, bleu_ref)

            return {
                "loss": loss,
                "bleu": bleu_score,
                "examples": {
                    "src": src_text,
                    "ref": ref_text,
                    "pred": generated_text,
                    "concepts": concepts,
                },
                "log": tb_log,
            }

        return {"loss": loss, "log": tb_log}

    def training_step_end(self, training_step_outputs):
        self.log("loss", training_step_outputs["loss"])
        if "bleu" in training_step_outputs:
            self.log("bleu", training_step_outputs["bleu"])
        if "examples" in training_step_outputs:
            srcs = training_step_outputs["examples"]["src"]
            refs = training_step_outputs["examples"]["ref"]
            preds = training_step_outputs["examples"]["pred"]
            for src, ref, pred in zip(srcs, refs, preds):
                self.logger.info(f"SOURCE: {src}")
                self.logger.info(f"REFERENCE: {ref}")
                self.logger.info(f"PREDICTION: {pred}")

    def validation_step(self, batch, batch_idx):

        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        concepts = batch["concepts"]

        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

        # Run the model
        outputs = self(
            src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        logits = outputs[0]

        # Calculate loss
        loss_fx = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = loss_fx(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1))

        # Generate text
        src_text = self.tokenizer.batch_decode(src_ids, skip_special_tokens=False)
        generated_ids = self.model.generate(src_ids)
        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        ref_text = self.tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)

        # Save generated text for BLEU computation
        for src, ref, pred in zip(src_text, ref_text, generated_text):
            self.bleu_data[src] = self.bleu_data.get(src, {"preds": [], "refs": []})
            self.bleu_data[src]["preds"].append(pred)
            self.bleu_data[src]["refs"].append(ref)

        return {
            "val_loss": val_loss,
            "examples": {
                "src": src_text,
                "ref": ref_text,
                "pred": generated_text,
                "concepts": concepts,
            },
        }

    def validation_epoch_end(self, outputs):
        # loss
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        # log to lightning
        self.log("val_loss", val_loss)

        # log to output
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


def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens
