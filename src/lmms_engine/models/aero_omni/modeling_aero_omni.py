from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..aero import AeroForConditionalGeneration
from ..aero.modeling_aero import AeroCausalLMOutputWithPast
from .configuration_aero_omni import AeroOmniConfig


class AeroOmniForConditionalGeneration(AeroForConditionalGeneration):
    config_class = AeroOmniConfig

    def __init__(self, config: AeroOmniConfig):
        super().__init__(config)
        # Original Vocab Size
        self.vocab_size = config.text_config.vocab_size
        # Additional Audio Vocab
        self.audio_vocab_size = config.code_book_size * config.num_codebooks

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        audio_input_ids: torch.LongTensor = None,
        audio_values: torch.FloatTensor = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        audio_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        codec_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
    ) -> Union[Tuple, AeroCausalLMOutputWithPast]:
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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if (audio_input_ids is None) ^ (audio_inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of audio_input_ids or audio_inputs_embeds"
            )

        # Concat the two inputs
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if audio_input_ids is not None:
            audio_inputs_embeds = self.get_input_embeddings()(audio_input_ids)
        inputs_embeds = (inputs_embeds + audio_inputs_embeds) / 2.0

        # Embed audio features
        if audio_values is not None:
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
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
                batch_size, max_seq_len
            )
            # Create mask
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(
                batch_size, 1, 1, max_seq_len
            ).expand(batch_size, 1, max_seq_len, max_seq_len)
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

        n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
        flops = self.calc_gpt_flops(attention_mask, n_audio_tokens)
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
            labels=labels,
        )

        logits = outputs[0]
        # Audio Logits should start from the vocab size
        audio_logits = logits[..., self.vocab_size :, :].contiguous()
        logits = logits[..., : self.vocab_size, :].contiguous()
        loss = outputs.get("loss", None)
        if labels is not None and loss is None:
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

        return AeroCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            flops=flops,
            audio_hidden_states=audio_features if audio_values is not None else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        past_key_values=None,
        audio_input_ids=None,
        inputs_embeds=None,
        audio_inputs_embeds=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        audio_values=None,
        audio_attention_mask=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if (audio_input_ids is None) ^ (audio_inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of audio_input_ids or audio_inputs_embeds"
            )
        # Concat the two inputs
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if audio_input_ids is not None:
            audio_inputs_embeds = self.get_input_embeddings()(audio_input_ids)
        inputs_embeds = (inputs_embeds + audio_inputs_embeds) / 2.0

        # Embed audio features
        if audio_values is not None:
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
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
                batch_size, max_seq_len
            )
            # Create mask
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(
                batch_size, 1, 1, max_seq_len
            ).expand(batch_size, 1, max_seq_len, max_seq_len)
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

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )
