from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import Qwen2Model, Qwen2PreTrainedModel
from transformers.utils import ModelOutput

from .arguments import SpecialToken


@dataclass
class SNDOutputWithPast(ModelOutput):
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    last_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    embeddings: Optional[torch.FloatTensor] = None


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph_hidden_size = config.model_args.graph_hidden_size
        self.intermediate_size = config.hidden_size * 2
        self.hidden_size = config.hidden_size
        self.gate_proj = nn.Linear(
            self.graph_hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.graph_hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2ModelForSNDPubEmbedding(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_args = config.model_args if hasattr(config, "model_args") else None

        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if self.model_args and self.model_args.use_graph:
            self.graph_proj = self.init_graph_proj(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def init_graph_proj(self, config):
        return Qwen2MLP(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        graph_embeddings: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SNDOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inputs_embeds = self.model.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.clone()

        # graph projection
        if self.model_args and self.model_args.use_graph:
            graph_mask = input_ids == self.graph_token_id

            # convert graph embeddings to the same dtype as the model
            graph_embeddings = (
                graph_embeddings.to(inputs_embeds.dtype)
                if graph_embeddings is not None
                else None
            )

            projected_graph_embeddings = self.graph_proj(graph_embeddings)

            # replace the graph token embeddings with the projected graph embeddings
            inputs_embeds[graph_mask] = projected_graph_embeddings

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs.last_hidden_state
        sentence_embeddings = self.last_token_pool(last_hidden_states, attention_mask)

        return SNDOutputWithPast(
            past_key_values=outputs.past_key_values,
            last_hidden_states=last_hidden_states,
            attentions=outputs.attentions,
            embeddings=sentence_embeddings,
        )

    # copied from https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct
    def last_token_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def add_special_tokens(self, tokenizer):
        self.graph_token_id = tokenizer.convert_tokens_to_ids(SpecialToken.GRAPH_TOKEN)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)
