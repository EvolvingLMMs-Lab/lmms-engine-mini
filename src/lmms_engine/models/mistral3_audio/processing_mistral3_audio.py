from typing import List, Optional, Union

from transformers import PixtralProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import (
    ImageInput,
    is_valid_image,
    load_image,
    make_batched_videos,
)
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    _validate_images_text_input_order,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class PixtralProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


class Mistral3AudioProcessor(PixtralProcessor):
    attributes = ["image_processor", "tokenizer", "audio_processor"]
    valid_kwargs = [
        "chat_template",
        "patch_size",
        "spatial_merge_size",
        "image_token",
        "image_break_token",
        "image_end_token",
    ]
    image_processor_class = "AutoImageProcessor"
    audio_processor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        audio_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        chat_template=None,
        image_token="[IMG]",  # set the default and let users change if they have peculiar special tokens in rare cases
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
        video_token="[VIDEO]",
        audio_token="[AUDIO]",
        **kwargs,
    ):
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token = image_token
        self.image_break_token = image_break_token
        self.image_end_token = image_end_token
        self.video_token = video_token
        self.audio_token = audio_token
        if chat_template is None:
            chat_template = self.default_chat_template
        super().__init__(
            image_processor,
            tokenizer,
            audio_processor,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        audios=None,
        videos=None,
        sampling_rate: Optional[int] = None,
        **kwargs: Unpack[PixtralProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
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
            image_inputs = self.image_processor(
                images, patch_size=self.patch_size, **output_kwargs["images_kwargs"]
            )
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            image_sizes = iter(image_inputs["image_sizes"])
            num_image_tokens = []
            for image_size in image_sizes:
                height, width = image_size
                num_height_tokens = height // (
                    self.patch_size * self.spatial_merge_size
                )
                num_width_tokens = width // (self.patch_size * self.spatial_merge_size)
                num_image_tokens.append(num_height_tokens * num_width_tokens)
            prompt_strings = self.expand_special_token(
                prompt_strings, num_image_tokens, self.image_token
            )

        if videos is not None:
            videos = make_batched_videos(videos)
            video_inputs = self.image_processor(
                videos, patch_size=self.patch_size, **output_kwargs["images_kwargs"]
            )
            image_sizes = iter(video_inputs["image_sizes"])
            num_video_tokens = []
            for image_size in image_sizes:
                height, width = image_size
                num_height_tokens = height // (
                    self.patch_size * self.spatial_merge_size
                )
                num_width_tokens = width // (self.patch_size * self.spatial_merge_size)
                num_video_tokens.append(num_height_tokens * num_width_tokens)
            prompt_strings = self.expand_special_token(
                prompt_strings, num_video_tokens, self.video_token
            )
            video_inputs["pixel_values_videos"] = video_inputs.pop("pixel_values")
            video_inputs["image_sizes_videos"] = video_inputs.pop("image_sizes")
        else:
            video_inputs = {}

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
            text = self.expand_audio_tokens(text, num_audio_tokens, self.audio_token)
        else:
            audio_inputs = {}

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs, **audio_inputs},
            tensor_type=output_kwargs["common_kwargs"]["return_tensors"],
        )

    def expand_special_token(
        self,
        text: List[TextInput],
        num_special_tokens: List[int],
        special_token: str,
    ):
        prompt_strings = []
        current_audio_idx = 0
        for sample in text:
            while special_token in sample:
                num_audio_token = num_special_tokens[current_audio_idx]
                sample = sample.replace(
                    special_token, "<placeholder>" * num_audio_token, 1
                )
                current_audio_idx += 1
            prompt_strings.append(sample)
        text = [
            sample.replace("<placeholder>", special_token) for sample in prompt_strings
        ]
        return text

    @property
    def default_chat_template(self):
        # fmt off
        return (
            '{%- set today = strftime_now("%Y-%m-%d") %}\n'
            "{%- set default_system_message = You are a helpful assistant"
            "{{- bos_token }}\n\n"
            "{%- if messages[0]['role'] == 'system' %}\n    "
            "{%- set system_message = messages[0]['content'] %}\n"
            "{%- set loop_messages = messages[1:] %}\n"
            "{%- else %}\n    "
            "{%- set system_message = default_system_message %}\n"
            "{%- set loop_messages = messages %}\n"
            "{%- endif %}\n"
            "{{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}\n\n"
            "{%- for message in loop_messages %}\n    "
            "{%- if message['role'] == 'user' %}\n\t    "
            "{%- if message['content'] is string %}\n"
            "{{- '[INST]' + message['content'] + '[/INST]' }}\n\t"
            "{%- else %}\n\t\t    "
            "{{- '[INST]' }}\n\t\t    "
            "{%- for block in message['content'] %}\n\t\t\t    "
            "{%- if block['type'] == 'text' %}\n\t\t\t\t    "
            "{{- block['text'] }}\n\t\t\t    "
            "{%- elif block['type'] == 'image' or block['type'] == 'image_url' %}\n\t\t\t\t    "
            "{{- '[IMG]' }}\n\t\t\t\t"
            "{% elif block['type'] == 'video' or 'video' in block or block['type'] == 'video_url' %}\n\t\t\t\t    "
            "{{- '[VIDEO]' }}\n\t\t\t\t"
            "{%- elif block['type'] == 'audio' or block['type'] == 'audio_url' or 'audio' in block %}\n\t\t\t\t    "
            "{{- '[AUDIO]' }}\n\t\t\t\t"
            "{%- else %}\n\t\t\t\t    "
            "{{- raise_exception('Only text and image blocks are supported in message content!') }}\n\t\t\t\t"
            "{%- endif %}\n\t\t\t"
            "{%- endfor %}\n\t\t    "
            "{{- '[/INST]' }}\n\t\t"
            "{%- endif %}\n    "
            "{%- elif message['role'] == 'system' %}\n        "
            "{{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    "
            "{%- elif message['role'] == 'assistant' %}\n        "
            "{{- message['content'] + eos_token }}\n    "
            "{%- else %}\n        "
            "{{- raise_exception('Only user, system and assistant roles are supported!') }}\n    "
            "{%- endif %}\n"
            "{%- endfor %}"
        )
        # fmt on
