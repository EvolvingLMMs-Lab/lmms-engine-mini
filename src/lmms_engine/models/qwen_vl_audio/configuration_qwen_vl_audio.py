from transformers import Qwen2AudioConfig, Qwen2VLConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen2VLAudioConfig(PretrainedConfig):
    # TODO: Finish docs
    r"""
    This is the configuration class to store the configuration of a [`<xxx>`]. It is used to instantiate an
    <xxx> model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Qwen2-Audio.

    e.g. [xxx](xxx.com)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.

    Example:

    ```python
    ```"""
    model_type = "qwen_vl_audio"
    sub_configs = {
        "qwen_vl_config": Qwen2VLConfig,
        "qwen_audio_config": Qwen2AudioConfig,
    }

    def __init__(
        self,
        qwen_vl_config: Qwen2VLConfig = None,
        qwen_audio_config: Qwen2AudioConfig = None,
        audio_token_index: int = 151657,
        **kwargs,
    ):
        if isinstance(qwen_vl_config, dict):
            qwen_vl_config["model_type"] = (
                qwen_vl_config["model_type"]
                if "model_type" in qwen_vl_config
                else "qwen2_vl"
            )
            qwen_vl_config = CONFIG_MAPPING[qwen_vl_config["model_type"]](
                **qwen_vl_config
            )
        elif qwen_vl_config is None:
            # By default use 7B config :D
            qwen_vl_config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        self.qwen_vl_config = qwen_vl_config

        if isinstance(qwen_audio_config, dict):
            qwen_audio_config["model_type"] = (
                qwen_audio_config["model_type"]
                if "model_type" in qwen_audio_config
                else "qwen2_audio"
            )
            qwen_audio_config = CONFIG_MAPPING[qwen_audio_config["model_type"]](
                **qwen_audio_config
            )
        elif qwen_audio_config is None:
            qwen_audio_config = AutoConfig.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct"
            )

        self.qwen_audio_config = qwen_audio_config

        # Because we init from qwen2 vl tokenizer, so the special tokens of audio are added to the tokenizer of Qwen2-VL
        # and will be different to the tokenizer of Qwen2-Audio.
        self.audio_token_index = audio_token_index
        super().__init__(**kwargs)
