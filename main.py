import argparse
from ast import parse
import os
from torch.utils.data import (
    Dataset,
    DataLoader
)
from torchmetrics import (
    BLEUScore
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import logging

from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    AdamW
)

class CommonGenModel(pl.LightningModule):
    logger = logging.getLogger('lightning')
    
    def __init__(self, learning_rate, tokenizer, model, hparams, log_interval):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate

        self.log_interval = log_interval

        self.bleu_data = {}

    # Do a forward pass through the model
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Load batch data
        src_ids, src_mask = batch['input_ids'], batch['attention_mask']
        tgt_ids = batch['labels']
        # Shift the decoder tokens right
        # This is important, idk why
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

        # Run the model 
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        logits = outputs[0]
        # Calculate loss
        loss_fx = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fx(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1))

        tb_log = {'train_loss': loss.detach()}

        # Generate sentences
        if batch_idx % self.log_interval == 0:
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
                },
                'log': tb_log
            }

        return {'loss': loss, 'log': tb_log}

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
        
        # Run the model
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        logits = outputs[0]

        # Calculate loss
        loss_fx = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = loss_fx(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1))

        # Generate text
        src_text = self.tokenizer.batch_decode(src_ids, skip_special_tokens=True)
        generated_ids = self.model.generate(src_ids)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        ref_text = self.tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)

        # Save generated text    for BLEU computation
        for src, ref, pred in zip(src_text, ref_text, generated_text):
            self.bleu_data[src] = self.bleu_data.get(src, {'preds': [], 'refs': []})
            self.bleu_data[src]['preds'].append(pred)
            self.bleu_data[src]['refs'].append(ref)

        return {
            'val_loss': val_loss,
            'examples': {
                'src': src_text,
                'ref': ref_text,
                'pred': generated_text
            }
        }

    def validation_epoch_end(self, outputs):
        # loss
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # bleu
        epoch_bleu_data = self._get_bleu_data()
        preds = [data[0] for data in epoch_bleu_data]
        targets = [data[1] for data in epoch_bleu_data]
        val_bleu = BLEUScore(1)(preds, targets)

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

    # def generate_text(self, text, eval_beams, early_stopping = True, max_len = 40):
    #     ''' Function to generate text '''
    #     generated_ids = self.model.generate(
    #         text["input_ids"],
    #         attention_mask=text["attention_mask"],
    #         use_cache=True,
    #         decoder_start_token_id = self.tokenizer.pad_token_id,
    #         num_beams= eval_beams,
    #         max_length = max_len,
    #         early_stopping = early_stopping
    #     )
    #     return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]

    def _get_bleu_data(self):
        data = []
        for value in self.bleu_data.values():
            refs = value['refs']
            for pred in set(value['preds']):
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
        self.train = torch.utils.data.Subset(self.dataset['train'], list(range(50)))
        self.validation = torch.utils.data.Subset(self.dataset['validation'], list(range(10)))
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

    def __init__(self, log_interval) -> None:
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if batch_idx % self.log_interval == 0:
            self.logger.info(f"STEP {trainer.global_step} Loss: {trainer.callback_metrics['loss']}")
            self.logger.info(f"STEP {trainer.global_step} BLEU: {trainer.callback_metrics['bleu']}")


class TensorBoardCallback(Callback):

    def __init__(self, model_name) -> None:
        super().__init__()
        self.model_name = model_name
        self.tb_logger = TensorBoardLogger(
            'tb_logs',
            name=self.model_name
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.tb_logger.log_metrics(outputs['log'], batch_idx)


class SaveGeneratedSentencesCallback(Callback):

    def __init__(self, output_file, min_epochs) -> None:
        super().__init__()
        self.output_file = output_file
        self.min_epoch_idx = min_epochs - 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch >= self.min_epoch_idx:
            if batch_idx == 0:
                try:
                    os.remove(self.output_file)
                except OSError:
                    pass
            with open(self.output_file, 'a') as file:
                inputs = outputs['examples']['src']
                references = outputs['examples']['ref']
                predictions = outputs['examples']['pred']
                for input, ref, pred in zip(inputs, references, predictions):
                    file.write(f'INPUT: {input}\nREFERENCE: {ref}\nPREDICTION: {pred}\n\n')


def shift_tokens_right(input_ids, pad_token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
  index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_epochs', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='baseline')
    parser.add_argument('--val_output', type=str, default='val_output.txt', help='File path where validation output should be stored.')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--log_interval', type=int, default=5, help="Interval of logging for Trainer.")
    parser.add_argument('--gpus', type=int, default=0)
    
    return parser


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def main():

    parser = get_arg_parser()
    args = parser.parse_args()

    setup_logging()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    common_gen_data = CommonGenDataModule(args.batch_size, tokenizer)
    common_gen_model = CommonGenModel(args.lr, tokenizer, model, None, args.log_interval)

    checkpoint = pl.callbacks.ModelCheckpoint(f'./checkpoints/{args.model_name}/')
    callbacks = [
        LossCallback(args.log_interval),
        TensorBoardCallback(args.model_name),
        SaveGeneratedSentencesCallback(args.val_output, args.min_epochs),
        checkpoint
    ]

    trainer = pl.Trainer(
        default_root_dir='.',
        gpus=args.gpus,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        auto_lr_find=False,
        callbacks=callbacks,
        log_every_n_steps=args.log_interval,
        enable_progress_bar=False
    )

    trainer.fit(common_gen_model, common_gen_data)
    trainer.validate(common_gen_model, common_gen_data)


if __name__ == "__main__":
    main()
