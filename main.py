from torch.utils.data import (
    Dataset,
    DataLoader
)
from torchmetrics import (
    BLEUScore
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import logging

from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    AdamW
)

LOG_EVERY_N_STEPS = 50

class CommonGenModel(pl.LightningModule):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger('lightning')
    
    def __init__(self, learning_rate, tokenizer, model, hparams):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate

        self.bleu_data = {}

    # Do a forward pass through the model
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        src_ids, src_mask = batch['input_ids'], batch['attention_mask']
        tgt_ids = batch['labels']
        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        # Create the loss function
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        # Generate sentences
        if batch_idx % LOG_EVERY_N_STEPS == 0:
            src_text = self.tokenizer.batch_decode(src_ids, skip_special_tokens=True)
            ref_text = self.tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
            generated_ids = self.model.generate(src_ids)
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            bleu_ref = [[ref] for ref in ref_text]
            bleu_score = BLEUScore()(generated_text, bleu_ref)

            return {
                'loss': loss,
                'bleu': bleu_score,
                'examples': {
                    'src': src_text,
                    'ref': ref_text,
                    'pred': generated_text
                }
            }

        return {'loss': loss}

    def training_step_end(self, training_step_outputs):
        self.log('loss', training_step_outputs['loss'])
        if 'bleu' in training_step_outputs:
            self.log('bleu', training_step_outputs['bleu'])
        if 'examples' in training_step_outputs:
            srcs = training_step_outputs['examples']['src']
            refs = training_step_outputs['examples']['ref']
            preds = training_step_outputs['examples']['pred']
            for src, ref, pred in zip(srcs, refs, preds):
                self.logger.info(f"SOURCE: {src}")
                self.logger.info(f"REFERENCE: {ref}")
                self.logger.info(f"PREDICTION: {pred}")

    def validation_step(self, batch, batch_idx):

        src_ids, src_mask = batch['input_ids'], batch['attention_mask']
        tgt_ids = batch['labels']

        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
        
        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        # Generate text
        input_text = self.tokenizer.batch_decode(src_ids, skip_special_tokens=True)
        generated_ids = self.model.generate(src_ids)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        ref_text = self.tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
        # Save for BLEU computation
        for src, ref, pred in zip(input_text, ref_text, generated_text):
            self.bleu_data[src] = self.bleu_data.get(src, {'preds': [], 'refs': []})
            self.bleu_data[src]['preds'].append(pred)
            self.bleu_data[src]['refs'].append(ref)

        return {'loss': val_loss}

    def validation_epoch_end(self, outputs):
        # loss
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # bleu
        epoch_bleu_data = self._get_bleu_data()
        preds = [data[0] for data in epoch_bleu_data]
        targets = [data[1] for data in epoch_bleu_data]
        val_bleu = BLEUScore()(preds, targets)

        # log to lightning
        self.log('val_loss', val_loss)
        self.log('val_bleu', val_bleu)

        # log to output
        self.logger.info(f"Validation loss: {val_loss}")
        self.logger.info(f"Validation BLEU: {val_bleu}")

        # reset predictions
        self.bleu_data = {}

    def test_step(self):
        pass

    def predict_step(self):
        pass

    def generate_text(self, text, eval_beams, early_stopping = True, max_len = 40):
        ''' Function to generate text '''
        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams= eval_beams,
            max_length = max_len,
            early_stopping = early_stopping
        )
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]

    def _get_bleu_data(self):
        data = []
        for value in self.bleu_data.values():
            refs = value['refs']
            for pred in value['preds']:
                data.append((pred, refs))
        return data


class CommonGenDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.batch_size = batch_size
        self.dataset = load_dataset('common_gen')
        self.setup(None)

    def setup(self, stage):
        self.train = self.dataset['train']
        self.validation = self.dataset['validation']
        self.test = self.dataset['test']

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self._collate_batch, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size, collate_fn=self._collate_batch)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self._collate_batch)

    def _collate_batch(self, batch):
        concepts = [" ".join(data['concepts']) for data in batch]
        targets = [data['target'] for data in batch]
        tokenized_concepts = self.tokenizer(concepts, padding=True, return_tensors='pt')
        tokenized_targets = self.tokenizer(targets, padding=True, return_tensors='pt')

        return {
            'input_ids': tokenized_concepts['input_ids'],
            'attention_mask': tokenized_concepts['attention_mask'],
            'labels': tokenized_targets['input_ids']
        }


class LossCallback(Callback):
    logger = logging.getLogger('metrics')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if batch_idx % LOG_EVERY_N_STEPS == 0:
            self.logger.info(f"STEP {trainer.global_step} Loss: {trainer.callback_metrics['loss']}")
            self.logger.info(f"STEP {trainer.global_step} BLEU: {trainer.callback_metrics['bleu']}")


def shift_tokens_right(input_ids, pad_token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
  index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens


def main():
    tokenizer = BartTokenizer.from_pretrained(
        'facebook/bart-base'
    )
    model = BartForConditionalGeneration.from_pretrained(
        'facebook/bart-base'
    )
    common_gen_data = CommonGenDataModule(3, tokenizer)
    common_gen_model = CommonGenModel(2e-5, tokenizer, model, None)

    checkpoint = pl.callbacks.ModelCheckpoint('./checkpoints/')
    trainer = pl.Trainer(
        default_root_dir='.',
        gpus=1,
        max_epochs=10,
        min_epochs=100,
        auto_lr_find=False,
        callbacks=[LossCallback(), checkpoint],
        log_every_n_steps=LOG_EVERY_N_STEPS,
        enable_progress_bar=False
    )

    trainer.fit(common_gen_model, common_gen_data)
    trainer.validate(common_gen_model, common_gen_data)


if __name__ == "__main__":
    main()
