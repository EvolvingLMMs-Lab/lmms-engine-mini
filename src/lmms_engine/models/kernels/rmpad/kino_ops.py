from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from lmms_engine.models.kino.modeling_kino import LlavaOnevisionCausalLMOutputWithPast

from .utils import _unpad_input


def forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    image_sizes: Optional[torch.LongTensor] = None,
    pixel_values_videos: torch.FloatTensor = None,
    image_sizes_videos: Optional[torch.LongTensor] = None,
    audio_values: torch.FloatTensor = None,
    audio_attention_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    vision_aspect_ratio: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int = 0,
) -> Union[Tuple, LlavaOnevisionCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


    Returns:
        [`~LlavaOnevisionCausalLMOutputWithPast`] (if `return_dict=True`) or a `tuple`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> import torch
    >>> from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

    >>> model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype="float16", device_map="cuda:0")
    >>> processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

    >>> conversation = [
    ...     {
    ...       "role": "user",
    ...       "content": [
    ...           {"type": "text", "text": "What is shown in this image?"},
    ...           {"type": "image"},
    ...         ],
    ...     },
    ... ]
    >>> prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    >>> image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> raw_image = Image.open(requests.get(image_file, stream=True).raw)
    >>> inputs = processor(text=prompt, images=raw_image, return_tensors='pt').to(0, torch.float16)

    >>> output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    >>> processor.batch_decode(output, skip_special_tokens=True)[0]
    "user\n\nWhat is shown in this image?\nassistant\ncat"
    ```"""
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
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )
    vision_aspect_ratio = (
        vision_aspect_ratio
        if vision_aspect_ratio is not None
        else self.config.vision_aspect_ratio
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if (
        pixel_values is not None or pixel_values_videos is not None
    ) and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both `pixel_values`/`pixel_values_videos` and `inputs_embeds` at the same time, "
            "and must specify either one"
        )
    if not self.use_vision_tower:
        assert (
            pixel_values is None and pixel_values_videos is None
        ), "Vision tower is not used, can not process images"

    if input_ids is None:
        assert (
            False
        ), "input_ids is None, please provide input_ids. To use rmpad with kino, please provide input ids. This is only used in training"

    # Unpad the input ids here
    input_ids, indices, cu_seq_lens, _ = _unpad_input(
        input_ids, attention_mask=attention_mask
    )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # Images are processed with Anyres
    if pixel_values is not None:
        if self.config.vision_aspect_ratio == "navit":
            image_features = self.get_navit_features(
                pixel_values,
                image_sizes,
            )
        else:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=self.image_newline,
                vision_aspect_ratio=vision_aspect_ratio,
            )
        n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (input_ids == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # Video are simply embedded and further pooled to decrease seq len
    if pixel_values_videos is not None:
        if self.config.vision_aspect_ratio == "navit":
            video_features = self.get_navit_features(
                pixel_values_videos,
                image_sizes_videos,
            )
        else:
            video_features = self.get_video_features(
                pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_newline = (
                self.image_newline[None, None, :]
                .repeat(video_features.shape[0], 1, 1)
                .to(video_features.device)
            )
            video_features = torch.cat((video_features, image_newline), dim=1)
            video_features = video_features.flatten(0, 1)

        n_video_tokens = (input_ids == self.config.video_token_index).sum().item()
        n_video_features = video_features.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        special_video_mask = (
            (input_ids == self.config.video_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

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
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
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

    n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
    n_video_tokens = (input_ids == self.config.video_token_index).sum().item()
    n_visual_tokens = n_image_tokens + n_video_tokens
    n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
    flops = self.calc_gpt_flops(attention_mask, n_audio_tokens, n_visual_tokens)
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
        cu_seq_lens=cu_seq_lens,
        indices=indices,
    )

    logits = outputs[0]
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

    return LlavaOnevisionCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        flops=flops,
        image_hidden_states=image_features if pixel_values is not None else None,
        video_hidden_states=video_features if pixel_values_videos is not None else None,
        audio_hidden_states=audio_features if audio_values is not None else None,
    )
