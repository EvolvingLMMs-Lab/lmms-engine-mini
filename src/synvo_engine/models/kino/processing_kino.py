# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for LLaVa-Onevision.
"""

import math
import os
from typing import Iterable, List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import select_best_resolution
from transformers.image_utils import (
    ImageInput,
    VideoInput,
    get_image_size,
    to_numpy_array,
)
from transformers.models.auto import AutoFeatureExtractor, AutoImageProcessor
from transformers.models.llava_onevision.processing_llava_onevision import (
    LlavaOnevisionProcessorKwargs,
)
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class KinoProcessor(ProcessorMixin):
    r"""
    Constructs a Kino processor which wraps a video processor, an audio processor, LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

    [`LlavaNextProcessor`] offers all the functionalities of [`LlavaOnevisionVideoProcessor`], [`LlavaOnevisionImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaOnevisionVideoProcessor.__call__`], [`~LlavaNextProcessor.__call__`] and [`~LlavaNextProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaOnevisionImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`LlavaOnevisionVideoProcessor`], *optional*):
            The video processor is a required input.
        num_image_tokens (`int`, *optional*):
            Number of image tokens for one imagethat will be returned by vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Shoudl be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
    """

    attributes = ["image_processor", "tokenizer", "video_processor", "audio_processor"]
    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "vision_feature_select_strategy",
        "image_token",
        "video_token",
        "audio_token",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoImageProcessor"
    audio_processor_class = "WhisperFeatureExtractor"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        audio_processor=None,
        num_image_tokens=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        video_token="<video>",
        audio_token="<|AUDIO|>",
        **kwargs,
    ):
        self.num_image_tokens = num_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = (
            tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        )
        self.video_token = (
            tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        )
        self.audio_token = (
            tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        )
        if chat_template is None:
            chat_template = self.default_chat_template
        super().__init__(
            image_processor,
            tokenizer,
            video_processor,
            audio_processor,
            chat_template=chat_template,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        audios: Union[np.ndarray, List[np.ndarray]] = None,
        videos: VideoInput = None,
        sampling_rate: Optional[int] = None,
        **kwargs: Unpack[LlavaOnevisionProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            LlavaOnevisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        image_inputs = {}
        video_inputs = {}
        audio_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(
                images, **output_kwargs["images_kwargs"]
            )
            if self.vision_feature_select_strategy == "navit":
                image_inputs["image_sizes"] = image_inputs.pop("image_grid_thw")
                merge_size = self.image_processor.merge_size
                num_image_tokens = [
                    (image_size[-2] * image_size[-1]).item() // (merge_size**2)
                    for image_size in image_inputs["image_sizes"]
                ]
                text = self.expand_image_tokens_navit(
                    text, num_image_tokens, self.image_token
                )
            else:
                image_sizes = iter(image_inputs["image_sizes"])
                height, width = get_image_size(
                    to_numpy_array(image_inputs["pixel_values"][0][0]),
                    channel_dim=output_kwargs["images_kwargs"].get("data_format"),
                )
                text = self._expand_image_tokens(
                    text, image_sizes, height, width, self.image_token
                )

        if videos is not None:
            video_inputs = self.video_processor(
                videos, **output_kwargs["videos_kwargs"]
            )

            if self.vision_feature_select_strategy == "navit":
                raise NotImplementedError(
                    "Navit strategy haven't implemented for video yet."
                )
            else:
                one_video = to_numpy_array(video_inputs["pixel_values_videos"][0])
                height, width = get_image_size(
                    one_video[0],
                    channel_dim=output_kwargs["images_kwargs"].get("data_format"),
                )
                num_frames = one_video.shape[0]  # frame dim is always after batch dim
                patches_height_width = int(math.sqrt(self.num_image_tokens))
                pooled_height_width = math.ceil(patches_height_width / 2)
                num_video_tokens = (
                    num_frames * pooled_height_width * pooled_height_width
                ) + 1  # +1 for newline token
                text = [
                    sample.replace(
                        self.video_token, self.video_token * num_video_tokens
                    )
                    for sample in text
                ]

        if audios is not None:
            audio_inputs = self.audio_processor(
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
            text = [
                sample.replace(self.audio_token, self.audio_token * num_audio_token)
                for sample, num_audio_token in zip(text, num_audio_tokens)
            ]

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs, **audio_inputs}
        )

    def _expand_image_tokens(
        self,
        text: List[TextInput],
        image_sizes: Iterable[Union[List[int], int]],
        height: int,
        width: int,
        special_token: str,
        num_frames: int = 1,
    ):
        prompt_strings = []
        for sample in text:
            while special_token in sample:
                image_size_list = next(image_sizes)
                original_size = (
                    image_size_list[0] if num_frames != 1 else image_size_list
                )
                if not isinstance(original_size, (list, tuple)):
                    # cast to list to avoid numerical precision errors when calculating unpadding
                    original_size = original_size.tolist()
                orig_height, orig_width = original_size
                num_image_tokens = self._get_number_of_features(
                    orig_height, orig_width, height, width
                )
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                sample = sample.replace(
                    special_token, "<placeholder>" * num_image_tokens * num_frames, 1
                )
            prompt_strings.append(sample)
        text = [
            sample.replace("<placeholder>", special_token) for sample in prompt_strings
        ]
        return text

    def expand_image_tokens_navit(
        self,
        text: List[TextInput],
        num_image_tokens: List[int],
        special_token: str,
    ):
        prompt_strings = []
        current_img_idx = 0
        for sample in text:
            while special_token in sample:
                num_image_token = num_image_tokens[current_img_idx]
                sample = sample.replace(
                    special_token, "<placeholder>" * num_image_token, 1
                )
                current_img_idx += 1
            prompt_strings.append(sample)
        text = [
            sample.replace("<placeholder>", special_token) for sample in prompt_strings
        ]
        return text

    def _get_number_of_features(
        self, orig_height: int, orig_width: int, height: int, width: int
    ) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = (
            height_best_resolution // height,
            width_best_resolution // width,
        )

        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height,
            orig_width,
            patches_height,
            patches_width,
            scale_height,
            scale_width,
        )

        # The base patch covers the entire image (no CLS for SigLIP)
        base_features = self.num_image_tokens
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens

    def _get_unpadded_features(
        self, height, width, patches_height, patches_width, scale_height, scale_width
    ):
        """
        Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
        because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
        patches an image is divided into and get the number of features from that.
        """
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(height * (current_width / width))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(width * (current_height / height))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        ratio = math.sqrt(current_height * current_width / (9 * patches_height**2))
        if ratio > 1.1:
            unpadded_features = int(current_height // ratio) * int(
                current_width // ratio
            )
            newline_features = int(current_height // ratio)

        return (unpadded_features, newline_features)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # override to save video-config in a separate config file
    # override to save audio-config in a separate config file
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
        os.makedirs(save_directory, exist_ok=True)
        video_processor_path = os.path.join(save_directory, "video_processor")
        self.video_processor.save_pretrained(video_processor_path)
        audio_processor_path = os.path.join(save_directory, "audio_processor")
        self.audio_processor.save_pretrained(audio_processor_path)

        video_processor_present = "video_processor" in self.attributes
        if video_processor_present:
            self.attributes.remove("video_processor")

        audio_processor_present = "audio_processor" in self.attributes
        if audio_processor_present:
            self.attributes.remove("audio_processor")

        outputs = super().save_pretrained(save_directory, **kwargs)

        if video_processor_present:
            self.attributes += ["video_processor"]
        if audio_processor_present:
            self.attributes += ["audio_processor"]
        return outputs

    # override to load video-config from a separate config file
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if return_unused_kwargs a tuple is returned where the second element is 'unused_kwargs'
        if isinstance(processor, tuple):
            processor = processor[0]

        try:
            video_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path, subfolder="video_processor"
            )
            processor.video_processor = video_processor
        except EnvironmentError:
            # this means users are using prev version of saved processor where we had only one preprocessor_config.json
            # for loading back that should work and load a LlavaOnevisionVideoProcessor class
            logger.info(
                "You are loading `LlavaOnevisionProcessor` but the indicated `path` doesn't contain a folder called "
                "`video_processor`. It is strongly recommended to load and save the processor again so the video processor is saved "
                "in a separate config."
            )

        try:
            audio_processor = AutoFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path, subfolder="audio_processor"
            )
            processor.audio_processor = audio_processor
        except EnvironmentError:
            logger.info(
                "You are loading `WhisperFeatureExtractor` but the indicated `path` doesn't contain a folder called "
                "`audio_processor`. It is strongly recommended to load and save the processor again so the audio processor is saved "
                "in a separate config."
            )

        return processor

    @property
    def default_chat_template(self):
        """
        This default vicuna template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content is a list of strings and audios.
        * If the content element is an audio, the template will output a sequence of <|AUDIO|> tokens

        Example:

        ```python
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
                {"type": "text", "text": "What's that sound?"},
            ]},
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"},
                {"type": "text", "text": "How about this one?"},
            ]},
        ]

        result = template.render(messages=messages, add_generation_prompt=True)
        ```
        """
        # fmt: off
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% set image_count = namespace(value=0) %}"
            "{% set video_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
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
                            "<image>\n"
                        "{% elif content['type'] == 'video' or 'video' in content %}"
                            "{% set video_count.value = video_count.value + 1 %}"
                            "{% if add_vision_id %}"
                                "Video {{ video_count.value }}: "
                            "{% endif %}"
                            "<video>\n"
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
