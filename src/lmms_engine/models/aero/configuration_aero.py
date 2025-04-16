# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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


from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class KinoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaOnevisionForConditionalGeneration`]. It is used to instantiate an
    Llava-NeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
    model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SiglipVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.
        video_token_index (`int`, *optional*, defaults to 151647):
            The video token index to encode the video prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"full"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            The index of the layer to select the vision feature.
        vision_aspect_ratio (`str`, *optional*, defaults to `"anyres_max_9"`):
            Aspect ratio used when processong image features. The default value is "anyres_max_9".
        image_grid_pinpoints (`List`, *optional*):
            A list of possible resolutions to use for processing high resolution images. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    Example:

    ```python
    >>> from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionConfig, SiglipVisionConfig, Qwen2Config

    >>> # Initializing a CLIP-vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = Qwen2Config()

    >>> # Initializing a Llava-Next llava-hf/llava-onevision-qwen2-7b-ov-hf style configuration
    >>> configuration = LlavaOnevisionConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-hf/llava-onevision-qwen2-7b-ov-hf style configuration
    >>> model = LlavaOnevisionForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kino"
    sub_configs = {
        "text_config": AutoConfig,
        "audio_config": AutoConfig,
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        audio_config=None,
        image_token_index=151646,
        video_token_index=151647,
        audio_token_index=151648,
        projector_hidden_act="gelu",
        projector_type="mlp",
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        vision_aspect_ratio="anyres_max_9",
        image_grid_pinpoints=None,
        tie_word_embeddings=False,
        use_rmpad=False,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.audio_token_index = audio_token_index
        self.projector_hidden_act = projector_hidden_act
        self.projector_type = projector_type

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.vision_aspect_ratio = vision_aspect_ratio
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [
                [384, 384],
                [384, 768],
                [384, 1152],
                [384, 1536],
                [384, 1920],
                [384, 2304],
                [768, 384],
                [768, 768],
                [768, 1152],
                [768, 1536],
                [768, 1920],
                [768, 2304],
                [1152, 384],
                [1152, 768],
                [1152, 1152],
                [1152, 1536],
                [1152, 1920],
                [1152, 2304],
                [1536, 384],
                [1536, 768],
                [1536, 1152],
                [1536, 1536],
                [1536, 1920],
                [1536, 2304],
                [1920, 384],
                [1920, 768],
                [1920, 1152],
                [1920, 1536],
                [1920, 1920],
                [1920, 2304],
                [2304, 384],
                [2304, 768],
                [2304, 1152],
                [2304, 1536],
                [2304, 1920],
                [2304, 2304],
            ]
        )
        # For navit, there is no image aspect ratio
        if self.vision_aspect_ratio == "navit":
            self.image_grid_pinpoints = None
        else:
            self.image_grid_pinpoints = image_grid_pinpoints

        if isinstance(vision_config, dict):
            if self.vision_aspect_ratio == "navit":
                vision_config = Qwen2VLVisionConfig(**vision_config)
            else:
                vision_config["model_type"] = (
                    vision_config["model_type"]
                    if "model_type" in vision_config
                    else "siglip_vision_model"
                )
                vision_config = CONFIG_MAPPING[vision_config["model_type"]](
                    **vision_config
                )

        self.vision_config = vision_config
        self.use_rmpad = use_rmpad

        if isinstance(text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "qwen2"
            )
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = AutoConfig.from_pretrained(
                "llava-hf/llava-onevision-qwen2-7b-ov-hf"
            ).text_config

        self.text_config = text_config

        if isinstance(audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"]
                if "model_type" in audio_config
                else "qwen2_audio_encoder"
            )
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["qwen2_audio_encoder"](
                d_model=1280,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                num_mel_bins=128,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )

        self.audio_config = audio_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
