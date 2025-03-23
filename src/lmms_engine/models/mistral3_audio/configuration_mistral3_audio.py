from transformers import AutoConfig, Mistral3Config
from transformers.models.auto import CONFIG_MAPPING


class Mistral3AudioConfig(Mistral3Config):
    model_type = "mistral3"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": AutoConfig,
        "audio_config": AutoConfig,
    }
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        audio_config=None,
        image_token_index=10,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
        multimodal_projector_bias=False,
        spatial_merge_size=2,
        video_token_index=None,
        audio_token_index=None,
        **kwargs,
    ):
        super().__init__(
            vision_config=vision_config,
            text_config=text_config,
            image_token_index=image_token_index,
            projector_hidden_act=projector_hidden_act,
            vision_feature_layer=vision_feature_layer,
            multimodal_projector_bias=multimodal_projector_bias,
            spatial_merge_size=spatial_merge_size,
            **kwargs,
        )

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
        self.video_token_index = video_token_index
        self.audio_token_index = audio_token_index


__all__ = ["Mistral3AudioConfig"]
