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
logging.basicConfig(level='INFO')
LOGGER = logging.getLogger("lightning-logger")

class CommonGenModel(pl.LightningModule):
    
    def __init__(self, learning_rate, tokenizer, model, hparams):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate

        self.predictions = []
        self.targets = []

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
                LOGGER.info(f"SOURCE: {src}")
                LOGGER.info(f"REFERENCE: {ref}")
                LOGGER.info(f"PREDICTION: {pred}")

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
        generated_ids = self.model.generate(src_ids)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        ref_text = self.tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
        self.predictions.extend([pred for pred in generated_text])
        self.targets.extend([[ref] for ref in ref_text])

        return {'loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_bleu = BLEUScore()(self.predictions, self.targets)
        self.log('val_loss', val_loss)
        self.log('val_bleu', val_bleu)

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
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if batch_idx % LOG_EVERY_N_STEPS == 0:
            print(f"-------------------\n" +\
                f"Step {trainer.global_step}:\n" +\
                f"Loss: {trainer.callback_metrics['loss']}\n" +\
                f"BLEU: {trainer.callback_metrics['bleu']}\n" +\
                "-------------------")


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
        gpus=0,
        max_epochs=2,
        min_epochs=2,
        auto_lr_find=False,
        callbacks=[LossCallback(), checkpoint],
        log_every_n_steps=LOG_EVERY_N_STEPS
    )

    trainer.fit(common_gen_model, common_gen_data)
    trainer.validate(common_gen_model, common_gen_data)


if __name__ == "__main__":
    main()
