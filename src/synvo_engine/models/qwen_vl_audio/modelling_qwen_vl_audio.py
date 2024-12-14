import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioMultiModalProjector,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from .configuration_qwen_vl_audio import Qwen2VLAudioConfig

QWEN2VLAUDIO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2VLAudioConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2Audio Model outputting raw hidden-states without any specific head on top.",
    QWEN2VLAUDIO_START_DOCSTRING,
)
class Qwen2VLAudioPreTrainedModel(PreTrainedModel):
    config_class = Qwen2VLAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Qwen2AudioAttention",
        "Qwen2VLDecoderLayer",
        "Qwen2VLVisionBlock",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        # important: this ported version of Qwen2Audio isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = self.config.qwen_vl_config.initializer_range

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2VLAudioForConditionalGeneration(
    Qwen2VLAudioPreTrainedModel, GenerationMixin
):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2VLAudioConfig):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(
            config.qwen_vl_config.vision_config
        )
        self.model = Qwen2VLModel(config.qwen_vl_config)
        self.vocab_size = config.qwen_vl_config.vocab_size
        self.lm_head = nn.Linear(
            config.qwen_vl_config.hidden_size,
            config.qwen_vl_config.vocab_size,
            bias=False,
        )
        self.rope_deltas = None  # cache rope_deltas here
        self.audio_tower = AutoModel.from_config(config.qwen_audio_config.audio_config)
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(
            config.qwen_audio_config
        )

        # Initialize weights and apply final processing
        self.post_init()
