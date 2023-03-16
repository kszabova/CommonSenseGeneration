import torch

from transformers import BartPretrainedModel, BartConfig, BartModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from transformers.models.bart.modeling_bart import (
    BartClassificationHead,
    shift_tokens_right,
)


class BartDoubleHeadsModel(BartPretrainedModel):
    """
    This class combines BartForConditionalGeneration and BartForSequenceClassification
    to be able to perform both tasks at the same time.

    The code is based on the implementations of the two classes above:
    https://github.com/huggingface/transformers/blob/v4.8.1/src/transformers/models/bart/modeling_bart.py#L1217
    https://github.com/huggingface/transformers/blob/v4.8.1/src/transformers/models/bart/modeling_bart.py#L1382
    """

    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        self.lm_head = torch.nn.Linear(
            config.d_model, self.model.shared.num_embeddings, bias=False
        )
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            1,  # number of labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int):
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros(
                (1, new_num_tokens - old_num_tokens),
                device=self.final_logits_bias.device,
            )
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if mc_labels is None and input_ids is not None:
            mc_labels = torch.ones(
                (input_ids.shape[0], 1, 1), dtype=torch.long, device=input_ids.device
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        # generation
        lm_logits = self.lm_head(hidden_states) + self.final_logits_bias
        lm_loss = None
        if labels is not None:
            lm_loss_fct = torch.nn.CrossEntropyLoss()
            # zero-out logits for incomplete sentences so that they don't contribute to loss
            logits_zerod = lm_logits * mc_labels.view(-1, 1, 1)
            labels_zerod = labels * mc_labels.view(-1, 1).long()
            lm_loss = lm_loss_fct(
                logits_zerod.view(-1, logits_zerod.shape[-1]), labels_zerod.view(-1)
            )
            lm_loss = lm_loss

        # classification
        # during training, we assume that mc_logits are never None
        # they can be None during inference (when calling generate()), in which case we don't need to compute the loss
        mc_logits = None
        if input_ids is not None:
            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError(
                    "All examples must have the same number of <eos> tokens."
                )
            sentence_representation = hidden_states[eos_mask, :].view(
                hidden_states.size(0), -1, hidden_states.size(-1)
            )[:, -1, :]
            mc_logits = self.classification_head(sentence_representation)
        mc_loss = None
        if mc_labels is not None and mc_logits is not None:
            mc_loss_fct = torch.nn.MSELoss()
            mc_loss = mc_loss_fct(mc_logits.view(-1), mc_labels.view(-1))

        output_lm = (lm_logits,) + outputs[1:]
        output_mc = (mc_logits,) + outputs[1:]
        loss = (
            lm_loss + mc_loss if lm_loss is not None and mc_loss is not None else None
        )

        if not return_dict:
            output = output_lm + output_mc
            return ((loss,) + output) if lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past

