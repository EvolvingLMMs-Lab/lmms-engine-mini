import importlib

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    PretrainedConfig,
)
from transformers.modeling_utils import PreTrainedModel


def register_model(
    model_type: str, model_config: PretrainedConfig, model_class: PreTrainedModel
):
    AutoConfig.register(model_type, model_config)
    AutoModelForCausalLM.register(model_config, model_class)


class ModelFactory:
    @staticmethod
    def create_model(model_name, **kwargs):
        config = AutoConfig.from_pretrained(model_name, **kwargs)

        if type(config) in AutoModelForCausalLM._model_mapping.keys():
            model_class = AutoModelForCausalLM
        elif type(config) in AutoModelForVision2Seq._model_mapping.keys():
            model_class = AutoModelForVision2Seq
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        return model_class
