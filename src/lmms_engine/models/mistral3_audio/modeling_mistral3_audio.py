from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    Mistral3ForConditionalGeneration,
    PreTrainedModel,
    Qwen2AudioEncoder,
)
from transformers.activations import ACT2FN
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3CausalLMOutputWithPast,
    Mistral3MultiModalProjector,
)
from transformers.utils import is_torchdynamo_compiling

from .configuration_mistral3_audio import Mistral3AudioConfig


@dataclass
class Mistral3AudioCausalLMOutputWithPast(Mistral3CausalLMOutputWithPast):
    """
    Base class for Mistral3 causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class Mistral3AudioProjector(nn.Module):
    def __init__(self, config: Mistral3AudioConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.d_model,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, x):
        hidden_states = self.linear_1(x)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Mistral3AudioForConditionalGeneration(Mistral3ForConditionalGeneration):
    _tied_weights_keys = ["language_model.lm_head.weight"]
    config_class = Mistral3AudioConfig

    def __init__(self, config: Mistral3AudioConfig):
        PreTrainedModel.__init__(self, config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.audio_tower = Qwen2AudioEncoder(config.audio_config)
        self.audio_modal_projector = Mistral3AudioProjector(config)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in self.language_model._tied_weights_keys
            ]

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        self.post_init()

    def prepare_audio_values(self, audio_values, audio_attention_mask):
        (
            audio_feat_lengths,
            audio_output_lengths,
        ) = self.audio_tower._get_feat_extract_output_lengths(
            audio_attention_mask.sum(-1)
        )
        batch_size, _, max_mel_seq_len = audio_values.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(
                0,
                max_seq_len,
                dtype=audio_feat_lengths.dtype,
                device=audio_feat_lengths.device,
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device,
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(
            audio_values, attention_mask=audio_attention_mask
        )
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.audio_modal_projector(selected_audio_feature)
        return audio_features, audio_output_lengths

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: torch.Tensor = None,
        image_sizes_videos: torch.Tensor = None,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ) -> Union[Tuple, Mistral3AudioCausalLMOutputWithPast]:
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
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                image_sizes=image_sizes,
            )

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(
                -1
            )
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            if (
                not is_torchdynamo_compiling()
                and inputs_embeds[special_image_mask].numel() != image_features.numel()
            ):
                n_image_tokens = (input_ids == self.config.image_token_index).sum()
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        if pixel_values_videos is not None:
            image_features_videos = self.get_image_features(
                pixel_values=pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                image_sizes=image_sizes_videos,
            )

            special_video_mask = (input_ids == self.config.video_token_index).unsqueeze(
                -1
            )
            special_video_mask = special_video_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            if (
                not is_torchdynamo_compiling()
                and inputs_embeds[special_video_mask].numel()
                != image_features_videos.numel()
            ):
                n_video_tokens = (input_ids == self.config.video_token_index).sum()
                n_video_features = (
                    image_features_videos.shape[0] * image_features_videos.shape[1]
                )
                raise ValueError(
                    f"Video features and Video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            image_features_videos = image_features_videos.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_video_mask, image_features_videos
            )

        if audio_values is not None:
            audio_features, audio_output_lengths = self.prepare_audio_values(
                audio_values, audio_attention_mask
            )
            n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
            n_audio_features = audio_output_lengths.sum()
            if n_audio_tokens != n_audio_features:
                raise ValueError(
                    f"Audio features and image tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                )
            audio_mask = (
                (input_ids == self.config.audio_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            audio_features = audio_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            # Audio feature is in (bs, max_seq_len, hidden_size)
            # If directly masked scatter, the embed will be place one by one (order is incorret)
            # We remove the padded values first
            unpadded_audio_features = [
                audio_feat[:audio_output_length]
                for audio_feat, audio_output_length in zip(
                    audio_features, audio_output_lengths
                )
            ]
            # Concat the audio features
            # Should exactly have audio_mask.sum() values
            unpadded_audio_features = torch.concatenate(unpadded_audio_features, dim=0)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask, unpadded_audio_features
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(
                    logits.device
                )
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Mistral3AudioCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
