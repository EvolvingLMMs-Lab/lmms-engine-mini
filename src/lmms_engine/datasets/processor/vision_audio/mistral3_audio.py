from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers.image_utils import load_image, make_batched_videos

from ....models.mistral3_audio import Mistral3AudioProcessor
from ....models.mistral3_audio.processing_mistral3_audio import (
    PixtralProcessorKwargs,
    is_image_or_image_url,
)
from .kino_qwen2_5_vl import KinoQwen2_5_DataProcessor


class Mistral3AudioDataProcessor(KinoQwen2_5_DataProcessor):
    def _build_processor(self):
        processor = Mistral3AudioProcessor.from_pretrained(self.config.processor_name)
        processor.tokenizer.chat_template = self.chat_template_no_system
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
            PixtralProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            if is_image_or_image_url(images):
                images = [images]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                pass
            elif (
                isinstance(images, (list, tuple))
                and isinstance(images[0], (list, tuple))
                and is_image_or_image_url(images[0][0])
            ):
                images = [image for sublist in images for image in sublist]
            else:
                raise ValueError(
                    "Invalid input images. Please provide a single image, a list of images, or a list of lists of images."
                )
            images = [load_image(im) if isinstance(im, str) else im for im in images]
            image_inputs = self.processor.image_processor(
                images, patch_size=self.patch_size, **output_kwargs["images_kwargs"]
            )
        else:
            image_inputs = {}

        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            image_sizes = iter(image_inputs["image_sizes"])
            num_image_tokens = []
            for image_size in image_sizes:
                if not isinstance(image_size, list):
                    image_size = image_size.tolist()
                height, width = image_size
                num_height_tokens = height // (
                    self.patch_size * self.spatial_merge_size
                )
                num_width_tokens = width // (self.patch_size * self.spatial_merge_size)
                num_image_tokens.append(num_height_tokens * num_width_tokens)
        else:
            num_image_tokens = None

        if videos is not None:
            videos = make_batched_videos(videos)
            video_inputs = self.processor.image_processor(
                videos, patch_size=self.patch_size, **output_kwargs["images_kwargs"]
            )
            image_sizes = iter(video_inputs["image_sizes"])
            num_video_tokens = []
            for image_size in image_sizes:
                if not isinstance(image_size, list):
                    image_size = image_size.tolist()
                height, width = image_size
                num_height_tokens = height // (
                    self.patch_size * self.spatial_merge_size
                )
                num_width_tokens = width // (self.patch_size * self.spatial_merge_size)
                num_video_tokens.append(num_height_tokens * num_width_tokens)
            video_inputs["pixel_values_videos"] = video_inputs.pop("pixel_values")
            video_inputs["image_sizes_videos"] = video_inputs.pop("image_sizes")
        else:
            video_inputs = {}
            num_video_tokens = None

        if audios is not None:
            audio_inputs = self.processor.audio_processor(
                audios,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                **kwargs,
            )
            audio_inputs["audio_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on
            audio_inputs["audio_values"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to audio_features for clarification
            # Computes the output length of the convolutional layers and the output length of the audio encoder
            input_lengths = (audio_inputs["audio_attention_mask"].sum(-1) - 1) // 2 + 1
            num_audio_tokens = (input_lengths - 2) // 2 + 1
        else:
            audio_inputs = {}
            num_audio_tokens = None

        inputs = self.get_mistral_template_labels(
            hf_messages,
            num_image_tokens,
            num_audio_tokens,
            num_video_tokens,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        if images is not None:
            inputs["pixel_values"] = image_inputs["pixel_values"]
            inputs["image_sizes"] = image_inputs["image_sizes"]
        if audios is not None:
            inputs["audio_values"] = audio_inputs["audio_values"]
            inputs["audio_attention_mask"] = audio_inputs["audio_attention_mask"]
        if videos is not None:
            for key, value in video_inputs.items():
                inputs[key] = value

        print(self.processor.batch_decode(inputs["input_ids"]))

        return inputs

    def get_mistral_template_labels(
        self,
        hf_messages,
        num_image_tokens: List[int],
        num_audio_tokens: List[int],
        num_video_tokens: List[int],
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
    ):
        special_tokens = self.processor.tokenizer.additional_special_tokens
        unmask_tokens_idx = [
            self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens
        ]
        input_id, target = [], []
        image_start_from = 0
        audio_start_from = 0
        video_start_from = 0
        if add_system_prompt:
            input_id += self.processor.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}],
            )
            target += [-100] * len(input_id)
            print(f"Input Id here {input_id}")
        for message in hf_messages:
            role = message["role"]
            encode_id = self.processor.apply_chat_template([message], tokenize=True)
            print(f"Encode Id here {encode_id}")
            # Should be 3 if instead of if else, so that can expand for each case
            if self.image_token_id in encode_id:
                encode_id, used_images = self._expand_encode_id_image_tokens(
                    encode_id, num_image_tokens, image_start_from
                )
                image_start_from += used_images
            if self.audio_token_id in encode_id:
                encode_id, used_audio = self._expand_encode_id_audio_tokens(
                    encode_id, num_audio_tokens, audio_start_from
                )
                audio_start_from += used_audio
            if self.video_token_id in encode_id:
                encode_id, used_video = self._expand_encode_id_video_tokens(
                    encode_id, num_video_tokens, video_start_from
                )
                video_start_from += used_video

            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            # Mistral got no generation prompt

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == self.image_token_id:
                target[idx] = -100
            if encode_id == self.audio_token_id:
                target[idx] = -100
            if encode_id == self.video_token_id:
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
            '{%- set today = strftime_now("%Y-%m-%d") %}'
            "{%- set loop_messages = messages %}"
            "{%- endif %}"
            "{%- for message in loop_messages %}"
            "   {%- if message['role'] == 'user' %}"
            "       {%- if message['content'] is string %}\n"
            "           {{- '[INST]' + message['content'] + '[/INST]' }}"
            "       {%- else %}"
            "           {{- '[INST]' }}"
            "           {%- for block in message['content'] %}"
            "               {%- if block['type'] == 'text' %}"
            "                   {{- block['text'] }}"
            "               {%- elif block['type'] == 'image' or block['type'] == 'image_url' %}"
            "                   {{- '[IMG]' }}"
            "               {%- elif block['type'] == 'video' or 'video' in block or block['type'] == 'video_url' %}"
            "                   {{- '[VIDEO]' }}"
            "               {%- elif block['type'] == 'audio' or block['type'] == 'audio_url' or 'audio' in block %}"
            "                   {{- '[AUDIO]' }}"
            "               {%- endif %}"
            "           {%- endfor %}"
            "           {{- '[/INST]' }}"
            "       {%- endif %}"
            "   {%- elif message['role'] == 'system' %}"
            "       {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}"
            "   {%- elif message['role'] == 'assistant' %}"
            "       {{- message['content'] + eos_token }}"
            "   {%- endif %}"
            "{%- endfor %}"
        )
        # fmt on
