from torch.utils.data import (
    Dataset,
    DataLoader
)
import pytorch_lightning as pl
import torch

from datasets import load_dataset
from transformers import (
    BartTokenizer
)

dataset = load_dataset("common_gen")
print(dataset.keys())

word_sep_token = '<|KEYWORD|>'
kw_sep_token = '<|PROMPT|>'
tokenizer = BartTokenizer.from_pretrained(
    "facebook/bart-base"
)

def collate_batch(batch):

    concepts = [" ".join(data['concepts']) for data in batch]
    targets = [data['target'] for data in batch]
    tokenized_concepts = tokenizer(concepts, padding=True, return_tensors='pt')
    tokenized_targets = tokenizer(targets, padding=True, return_tensors='pt')

    return {
        'input_ids': tokenized_concepts['input_ids'],
        'attention_mask': tokenized_concepts['attention_mask'],
        'labels': tokenized_targets['input_ids']
    }

# dataloader = DataLoader(dataset["train"], collate_fn=collate_batch, batch_size=3, shuffle=True)
# for batch in dataloader:
#     print(batch)
#     break

# class CommonSenseModel(pl.LightningModule):
    
#     def __init__(self, learning_rate, tokenizer, model, hparams):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.model = model
#         self.learning_rate = learning_rate

#     # Do a forward pass through the model
#     def forward(self, input_ids, **kwargs):
#         return self.model(input_ids, **kwargs)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
#         return optimizer

#     def training_step(self, batch, batch_idx):
#         # Load the data into variables
#         src_ids, src_mask = batch[0], batch[1]
#         tgt_ids = batch[2]
#         # Shift the decoder tokens right (but NOT the tgt_ids)
#         #decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

#         # Run the model and get the logits
#         outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
#         lm_logits = outputs[0]
#         # Create the loss function
#         ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
#         # Calculate the loss on the un-shifted tokens
#         loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

#         return {'loss':loss}


class CommonGenDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-base"
        )
        self.batch_size = batch_size
        self.dataset = load_dataset('common_gen')
        self.setup(None)

    def setup(self, stage):
        self.train = self.dataset['train']
        self.validation = self.dataset['validation']
        self.test = self.dataset['test']

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self._collate_batch, shuffle=True)

    def validation_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size, collate_fn=self._collate_batch, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self._collate_batch, shuffle=True)

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


data_module = CommonGenDataModule(3)
train = data_module.train_dataloader()
for batch in train:
    print(batch)
    break
