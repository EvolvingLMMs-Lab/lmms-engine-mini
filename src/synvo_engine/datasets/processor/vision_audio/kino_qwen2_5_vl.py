from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ....models.qwen2_5_vl_audio import KinoQwen2_5_VLProcessor
from ....models.qwen2_5_vl_audio.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs
from .kino_processor import KinoDataProcessor


class KinoQwen2_5_DataProcessor(KinoDataProcessor):
    def _build_processor(self):
        processor = KinoQwen2_5_VLProcessor.from_pretrained(self.config.processor_name)
        if self.config.max_pixels:
            processor.image_processor.max_pixels = self.config.max_pixels
        if self.config.min_pixels:
            processor.image_processor.min_pixels = self.config.min_pixels
        return processor

    def process(
        self,
        images: List[Image.Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt=True,
        add_generation_prompt=False,  # Whether add a generation prompt at the end
        **kwargs,
    ):
        """
        A wrapper method to process single data
        """

        output_kwargs = self.processor._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        video_inputs = {}
        audio_inputs = {}

        if images is not None:
            image_inputs = self.processor.image_processor(
                images, return_tensors="pt", **output_kwargs["images_kwargs"]
            )
            image_inputs["image_sizes"] = image_inputs.pop("image_grid_thw")
            merge_size = self.processor.image_processor.merge_size
            num_image_tokens = [
                (image_size[-2] * image_size[-1]).item() // (merge_size**2)
                for image_size in image_inputs["image_sizes"]
            ]
        else:
            num_image_tokens = None

        if videos is not None:
            raise NotImplementedError

        if audios is not None:
            audio_inputs = self.processor.audio_processor(
                audios,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                return_tensors="pt",
                **kwargs,
            )
            audio_inputs["audio_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on
            audio_inputs["audio_values"] = audio_inputs.pop("input_features")
            input_lengths = (audio_inputs["audio_attention_mask"].sum(-1) - 1) // 2 + 1
            num_audio_tokens = (input_lengths - 2) // 2 + 1
        else:
            num_audio_tokens = None

        inputs = self.get_qwen_template_labels(
            hf_messages,
            num_image_tokens,
            num_audio_tokens,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        if images is not None:
            inputs["pixel_values"] = image_inputs["pixel_values"]
            inputs["image_grid_thw"] = image_inputs["image_sizes"]
        if audios is not None:
            inputs["audio_values"] = audio_inputs["audio_values"]
            inputs["audio_attention_mask"] = audio_inputs["audio_attention_mask"]

        return inputs

    def get_qwen_template_labels(
        self,
        hf_messages,
        num_image_tokens: List[int],
        num_audio_tokens: List[int],
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
    ):
        image_token_index = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
        special_tokens = self.processor.tokenizer.additional_special_tokens
        special_tokens.extend(["<|im_start|>", "<|im_end|>"])
        unmask_tokens_idx = [
            self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens
        ]
        input_id, target = [], []
        start_from = 0
        if add_system_prompt:
            input_id += self.processor.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}],
            )
            target += [-100] * len(input_id)
        for message in hf_messages:
            role = message["role"]
            # Cautions, qwen2_5 vl tokenizer wrap into a list
            encode_id = self.processor.apply_chat_template([message], tokenize=True)[0]
            if image_token_index in encode_id:
                encode_id, used_images = self._expand_encode_id_image_tokens(
                    encode_id, num_image_tokens, start_from
                )
                start_from += used_images
            elif self.audio_token_id in encode_id:
                encode_id, used_audio = self._expand_encode_id_audio_tokens(
                    encode_id, num_audio_tokens
                )
            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                # Adopted from llava-ov that mask out the assistant
                encode_id[:3] = [-100] * 3
                target += encode_id

        if add_generation_prompt:
            generation_tokens = self.processor.tokenizer.encode(
                "<|im_start|>assistant\n"
            )
            input_id += generation_tokens
            target += [-100] * len(generation_tokens)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                target[idx] = -100
            if encode_id == self.audio_token_id:
                target[idx] = -100

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )

    @property
    def chat_template_no_system(self):
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% set image_count = namespace(value=0) %}"
            "{% set video_count = namespace(value=0) %}"
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{% if message['content'] is string %}"
            "{{ message['content'] }}<|im_end|>\n"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if 'audio' in content or 'audio_url' in content %}"
            "{% set audio_count.value = audio_count.value + 1 %}"
            "Audio {{ audio_count.value }}: <|AUDIO|>\n"
            "{% elif content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
            "{% set image_count.value = image_count.value + 1 %}"
            "{% if add_vision_id %}"
            "Picture {{ image_count.value }}: "
            "{% endif %}"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            "{% elif content['type'] == 'video' or 'video' in content %}"
            "{% set video_count.value = video_count.value + 1 %}"
            "{% if add_vision_id %}"
            "Video {{ video_count.value }}: "
            "{% endif %}"
            "<|vision_start|><|video_pad|><|vision_end|>\n"
            "{% elif 'text' in content %}"
            "{{ content['text'] }}"
            "{% endif %}"
            "{% endfor %}"
            "<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on
